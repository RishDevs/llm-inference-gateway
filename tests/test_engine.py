import random

import torch

from gateway.engine import Sequence, sample


def make(rng, n, gen):
    return Sequence(prompt_ids=[rng.randrange(2, 300) for _ in range(n)], max_tokens=gen, ignore_eos=True)


def clone(s):
    return Sequence(prompt_ids=list(s.prompt_ids), max_tokens=s.max_tokens, ignore_eos=True)


def run_alone(runner, seq):
    batch, emitted, _ = runner.prefill([seq])
    runner.settle(emitted)
    batch = runner.evict(batch)
    while batch is not None:
        emitted, _ = runner.decode(batch)
        runner.settle(emitted)
        batch = runner.evict(batch)
    return seq.output_ids


def test_prefill_batch_matches_one_at_a_time(runner):
    rng = random.Random(7)
    seqs = [make(rng, n, g) for n, g in [(5, 6), (17, 3), (9, 9), (31, 4)]]
    expected = [run_alone(runner, clone(s)) for s in seqs]

    batch, emitted, _ = runner.prefill(seqs)
    runner.settle(emitted)
    batch = runner.evict(batch)
    while batch is not None:
        emitted, _ = runner.decode(batch)
        runner.settle(emitted)
        batch = runner.evict(batch)

    assert [s.output_ids for s in seqs] == expected


def test_join_and_evict_mid_generation_matches(runner):
    rng = random.Random(3)
    first = [make(rng, 12, 10), make(rng, 4, 3)]
    late = [make(rng, 40, 6), make(rng, 2, 8)]
    expected = [run_alone(runner, clone(s)) for s in first + late]

    running, emitted, _ = runner.prefill(first)
    runner.settle(emitted)
    for _ in range(4):
        emitted, _ = runner.decode(running)
        runner.settle(emitted)
        running = runner.evict(running)

    joining, emitted, _ = runner.prefill(late)
    runner.settle(emitted)
    running = runner.merge(running, joining)
    while running is not None:
        emitted, _ = runner.decode(running)
        runner.settle(emitted)
        running = runner.evict(running)

    assert [s.output_ids for s in first + late] == expected


def test_evict_trims_leading_padding(runner):
    rng = random.Random(5)
    long, short = make(rng, 30, 1), make(rng, 3, 5)
    batch, emitted, _ = runner.prefill([long, short])
    runner.settle(emitted)
    assert batch.width == 30
    batch = runner.evict(batch)
    assert batch.size == 1
    assert batch.width == 3
    assert batch.cache.layers[0].keys.shape[-2] == 3


def test_padding_accounting(runner):
    rng = random.Random(1)
    seqs = [make(rng, 4, 3), make(rng, 12, 3)]
    batch, emitted, stats = runner.prefill(seqs)
    assert (stats.slots, stats.useful) == (24, 16)
    runner.settle(emitted)
    emitted, stats = runner.decode(batch)
    assert stats.width == 13
    assert (stats.slots, stats.useful) == (26, 5 + 13)


def test_eos_stops_unless_ignored(runner):
    seq = Sequence(prompt_ids=[5, 6, 7], max_tokens=10)
    _, emitted, _ = runner.prefill([seq])
    runner.eos_ids = frozenset({emitted[0][1]})
    runner.settle(emitted)
    assert seq.finish_reason == "stop"
    assert len(seq.output_ids) == 1


def test_sample_greedy_and_top_p():
    logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]])
    temps = torch.tensor([0.0, 1.0])
    tight = torch.tensor([1.0, 0.01])
    assert sample(logits, temps, tight).tolist() == [1, 0]
