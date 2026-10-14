import time
from dataclasses import replace

import pytest

from gateway.admission import QueueFull, RequestTooLarge, TokenBudget
from gateway.engine import Sequence, StepCost
from gateway.scheduler import Scheduler


def seq(n, gen):
    return Sequence(prompt_ids=[3] * n, max_tokens=gen, ignore_eos=True)


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def drain(sched, items, limit=10_000):
    for _ in range(limit):
        if all(s.finished for s in items):
            return
        sched.step()
    raise AssertionError("did not finish")


def test_budget_take_and_give_back():
    b = TokenBudget(100)
    assert b.take(60)
    assert not b.take(50)
    b.give_back(60)
    assert b.take(100)
    assert TokenBudget(None).take(10**9)


def test_rejects_oversized_and_full_queue(runner, settings):
    sched = Scheduler(runner, replace(settings, max_queue=2), budget=1000)
    with pytest.raises(RequestTooLarge):
        sched.submit(seq(500, 100))
    sched.submit(seq(4, 4))
    sched.submit(seq(4, 4))
    with pytest.raises(QueueFull):
        sched.submit(seq(4, 4))
    assert sched.metrics.totals["rejected"] == 1


def test_rebalance_splits_padded_lane_and_merges_back(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, join_wait_ms=0.0, max_lanes=2), budget=None)
    short = [seq(4, 30) for _ in range(3)]
    long = [seq(120, 30) for _ in range(3)]
    for s in short + long:
        sched.submit(s)
    sched._admit()
    assert len(sched.lanes) == 1
    sched.rebalance()
    assert len(sched.lanes) == 2
    widths = sorted(b.width for b in sched.lanes.values())
    assert widths == [4, 120]
    assert sorted(len(b.seqs) for b in sched.lanes.values()) == [3, 3]
    drain(sched, short + long)
    assert all(len(s.output_ids) == 30 for s in short + long)
    assert sched.metrics.totals["lane_splits"] == 1


def test_split_lanes_generate_same_tokens(runner, settings):
    items = [seq(4, 12), seq(90, 12), seq(6, 12), seq(85, 12)]
    solo = Scheduler(runner, replace(settings, max_wait_ms=0.0, bucketing=False), budget=None)
    copies = [Sequence(prompt_ids=list(s.prompt_ids), max_tokens=12, ignore_eos=True) for s in items]
    for c in copies:
        solo.submit(c)
    drain(solo, copies)

    split = Scheduler(runner, replace(settings, max_wait_ms=0.0, max_lanes=2, rebalance_every=1), budget=None)
    for s in items:
        split.submit(s)
    drain(split, items)
    assert split.metrics.totals["lane_splits"] >= 1
    assert [s.output_ids for s in items] == [c.output_ids for c in copies]


def test_cost_model_keeps_one_lane_when_split_is_not_worth_it(runner, settings):
    cheap_padding = StepCost(fixed_s=0.02, per_slot_s=1e-7)
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, join_wait_ms=0.0), budget=None, costs=cheap_padding)
    for s in [seq(4, 30), seq(120, 30)]:
        sched.submit(s)
    sched._admit()
    sched.rebalance()
    assert len(sched.lanes) == 1


def test_best_cut_and_padding():
    from gateway.engine import best_cut, padding

    assert padding([3, 5, 10]) == 7 + 5
    short, long, saved = best_cut([100, 3, 98, 4])
    assert short == [1, 3] and long == [0, 2]
    assert saved == padding([100, 3, 98, 4]) - padding([3, 4]) - padding([100, 98])


def test_unbucketed_takes_fifo(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, bucketing=False), budget=None)
    items = [seq(10, 8), seq(200, 100), seq(10, 8)]
    for s in items:
        sched.submit(s)
    assert sched._pick() == (0, items)


def test_budget_reserves_padded_footprint(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, bucketing=False), budget=300)
    a, b, c = seq(50, 50), seq(100, 40), seq(10, 10)
    for s in (a, b, c):
        sched.submit(s)
    assert sched._admit()
    assert sched.running == [a, b]
    assert sched.reserved == {0: 2 * 140}
    assert sched.budget.used == 280
    assert sched._pick() == (0, [])


def test_prefill_cap_splits_microbatches(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, bucketing=False, max_prefill_tokens=512), budget=None)
    items = [seq(200, 4) for _ in range(4)]
    for s in items:
        sched.submit(s)
    assert len(sched._pick()[1]) == 2


def test_requests_join_running_batch(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0, join_wait_ms=0.0), budget=None)
    first = seq(8, 20)
    sched.submit(first)
    sched.step()
    assert sched.rows == 1
    late = seq(30, 5)
    sched.submit(late)
    sched.step()
    assert sched.rows == 2
    drain(sched, [first, late])
    assert len(first.output_ids) == 20 and len(late.output_ids) == 5
    sched.step()
    assert not sched.lanes
    assert not sched.reserved


def test_end_to_end_threaded(runner, settings):
    sched = Scheduler(runner, settings, budget=5000)
    sched.start()
    try:
        items = [seq(n, g) for n, g in [(5, 3), (40, 7), (9, 12), (70, 2), (3, 30)]]
        for s in items:
            sched.submit(s)
            time.sleep(0.01)
        assert wait_for(lambda: all(s.finished for s in items))
        for s in items:
            assert s.finish_reason == "length"
            assert len(s.output_ids) == s.max_tokens
        assert wait_for(lambda: not sched.lanes)
        assert sched.budget.used == 0
        totals = sched.metrics.totals
        assert totals["generated_tokens"] == sum(s.max_tokens for s in items)
        assert totals["requests"]["length"] == 5
    finally:
        sched.stop()


def test_cancel_waiting_and_running(runner, settings):
    sched = Scheduler(runner, replace(settings, max_wait_ms=0.0), budget=None)
    a, b = seq(5, 50), seq(5, 50)
    sched.submit(a)
    sched.step()
    sched.submit(b)
    sched.cancel(b)
    assert b.finish_reason == "cancelled"
    sched.cancel(a)
    sched.step()
    assert a.finish_reason == "cancelled"
    assert not sched.lanes


def test_queue_timeout(runner, settings):
    sched = Scheduler(runner, replace(settings, queue_timeout_s=0.01, max_wait_ms=10_000), budget=None)
    s = seq(5, 5)
    sched.submit(s)
    time.sleep(0.03)
    sched.step()
    assert s.finish_reason == "error"
