from __future__ import annotations

import itertools
import logging
import threading
import time

import torch

from .admission import QueueFull, RequestTooLarge, TokenBudget
from .engine import Batch, ModelRunner, Sequence, StepCost, best_cut, padding
from .metrics import Metrics

log = logging.getLogger("gateway.scheduler")


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


class Scheduler:
    def __init__(
        self,
        runner: ModelRunner,
        settings,
        metrics: Metrics | None = None,
        budget: int | None = None,
        costs: StepCost | None = None,
    ):
        self.runner = runner
        self.settings = settings
        self.metrics = metrics or Metrics()
        self.budget = TokenBudget(budget)
        self.costs = costs or StepCost(fixed_s=0.0, per_slot_s=1.0)
        self._lane_ids = itertools.count(1)
        self._steps = 0
        self.waiting: list[Sequence] = []
        self.lanes: dict[int, Batch] = {}
        self.reserved: dict[int, int] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def rows(self) -> int:
        return sum(b.size for b in self.lanes.values())

    @property
    def running(self) -> list[Sequence]:
        return [s for b in self.lanes.values() for s in b.seqs]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._halt.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def submit(self, seq: Sequence) -> Sequence:
        s = self.settings
        if seq.prompt_len == 0:
            raise RequestTooLarge("prompt is empty")
        if seq.need > s.max_model_len:
            raise RequestTooLarge(f"prompt + max_tokens = {seq.need} exceeds max_model_len {s.max_model_len}")
        if not self.budget.unlimited and seq.need > self.budget.capacity:
            raise RequestTooLarge(f"request needs {seq.need} tokens, budget is {self.budget.capacity}")
        with self._lock:
            if s.admission and len(self.waiting) >= s.max_queue:
                self.metrics.on_reject()
                raise QueueFull(s.retry_after_s)
            self.waiting.append(seq)
        self._wake.set()
        return seq

    def cancel(self, seq: Sequence) -> None:
        seq.cancelled = True
        with self._lock:
            if seq in self.waiting:
                self.waiting.remove(seq)
                seq.finish("cancelled")
                self.metrics.on_finish(seq)

    def step(self) -> bool:
        self._expire()
        joined = self._admit()
        for key in list(self.lanes):
            self._decode(key)
        self._steps += 1
        if self._steps % self.settings.rebalance_every == 0:
            self.rebalance()
        self._publish()
        return joined or bool(self.lanes)

    def _loop(self) -> None:
        while not self._halt.is_set():
            try:
                if not self.step():
                    self._idle()
            except Exception:
                log.exception("engine step failed")
                self._fail_everything(500, "engine error")
        self._fail_everything(503, "server shutting down")

    def _idle(self) -> None:
        with self._lock:
            oldest = self.waiting[0].arrived if self.waiting else None
        if oldest is None:
            timeout = 0.5
        else:
            timeout = max(0.0, oldest + self.settings.max_wait_ms / 1000 - time.monotonic())
        if timeout > 0:
            self._wake.wait(timeout)
        self._wake.clear()

    def _expire(self) -> None:
        limit = self.settings.queue_timeout_s
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            stale = [q for q in self.waiting if now - q.arrived > limit]
            for q in stale:
                self.waiting.remove(q)
        for q in stale:
            q.fail(503, "timed out waiting for capacity")
            self.metrics.on_finish(q)

    @staticmethod
    def join_padding(lane: Batch, seq: Sequence) -> int:
        width = lane.width
        if seq.prompt_len <= width:
            return width - seq.prompt_len
        return (seq.prompt_len - width) * lane.size

    def route(self, seq: Sequence) -> int:
        if not self.settings.bucketing or not self.lanes:
            return 0 if not self.lanes else next(iter(self.lanes))
        return min(self.lanes, key=lambda k: (self.join_padding(self.lanes[k], seq), k))

    def rebalance(self) -> None:
        if not self.settings.bucketing or not self.lanes:
            return
        factor = self.settings.lane_split_factor
        if len(self.lanes) < self.settings.max_lanes:
            key = max(self.lanes, key=lambda k: padding(self.lanes[k].cached))
            lane = self.lanes[key]
            if lane.size >= 2:
                short, long, saved = best_cut(lane.cached)
                if saved and self.costs.worth_splitting(saved, factor):
                    self._split(key, short, long)
                    return
        if len(self.lanes) >= 2:
            keys = sorted(self.lanes, key=lambda k: self.lanes[k].width)
            pairs = list(zip(keys, keys[1:]))
            added = {
                (a, b): padding(self.lanes[a].cached + self.lanes[b].cached)
                - padding(self.lanes[a].cached)
                - padding(self.lanes[b].cached)
                for a, b in pairs
            }
            a, b = min(added, key=added.get)
            if self.costs.worth_merging(added[(a, b)], factor):
                self._merge_lanes(a, b)

    def _split(self, key: int, short: list[int], long: list[int]) -> None:
        lane = self.lanes.pop(key)
        held = self.reserved.pop(key, 0)
        fresh = next(self._lane_ids)
        self.lanes[key] = self.runner.take(lane, long)
        self.lanes[fresh] = self.runner.take(lane, short)
        lane.cache = None
        self.reserved[key] = held
        self.reserved[fresh] = 0
        self._resize(key)
        self._resize(fresh)
        self.metrics.on_lanes("split")

    def _merge_lanes(self, a: int, b: int) -> None:
        first, second = self.lanes[a], self.lanes[b]
        footprint = (first.size + second.size) * max(first.horizon, second.horizon)
        held = self.reserved.get(a, 0) + self.reserved.get(b, 0)
        if footprint > held and not self.budget.take(footprint - held):
            return
        if footprint < held:
            self.budget.give_back(held - footprint)
        del self.lanes[b]
        self.reserved.pop(b, None)
        self.lanes[a] = self.runner.merge(first, second)
        self.reserved[a] = footprint
        self.metrics.on_lanes("merge")

    def _pool(self, seed: Sequence, target: int) -> list[Sequence]:
        if not self.settings.bucketing or len(self.lanes) <= 1:
            return list(self.waiting)
        return [q for q in self.waiting if self.route(q) == target]

    def _ready(self, seed: Sequence, free_rows: int) -> bool:
        s = self.settings
        wait_ms = s.join_wait_ms if self.lanes else s.max_wait_ms
        age_ms = (time.monotonic() - seed.arrived) * 1000
        return len(self.waiting) >= free_rows or age_ms >= wait_ms

    def _pick(self) -> tuple[int, list[Sequence]]:
        s = self.settings
        with self._lock:
            free_rows = s.max_batch_size - self.rows
            if not self.waiting or free_rows <= 0:
                return 0, []
            seed = self.waiting[0]
            if not self._ready(seed, free_rows):
                return 0, []

            key = self.route(seed)
            pool = self._pool(seed, key)
            lane = self.lanes.get(key)
            lane_rows = lane.size if lane is not None else 0
            held = self.reserved.get(key, 0)

            picked: list[Sequence] = []
            width = 0
            horizon = lane.horizon if lane is not None else 0
            for q in pool:
                if len(picked) >= free_rows:
                    break
                w = max(width, q.prompt_len)
                h = max(horizon, q.need)
                if s.admission and picked and (len(picked) + 1) * w > s.max_prefill_tokens:
                    continue
                grown = (lane_rows + len(picked) + 1) * h
                if not self.budget.fits(grown - held):
                    if not picked:
                        return key, []
                    continue
                picked.append(q)
                width, horizon = w, h

            footprint = (lane_rows + len(picked)) * horizon
            if not self.budget.take(footprint - held):
                return key, []
            self.reserved[key] = footprint
            chosen = set(id(q) for q in picked)
            self.waiting = [q for q in self.waiting if id(q) not in chosen]
        return key, picked

    def _admit(self) -> bool:
        key, picked = self._pick()
        if not picked:
            return False
        self.metrics.on_microbatch(len(picked))
        try:
            joining, emitted, stats = self.runner.prefill(picked)
        except Exception as err:
            self._crash(picked, err)
            self._resize(key)
            return True
        self.metrics.on_slots("prefill", stats.slots, stats.useful)
        self._settle(emitted)
        self.lanes[key] = self.runner.merge(self.lanes.get(key), joining)
        self._shrink(key)
        return True

    def _decode(self, key: int) -> None:
        self.lanes[key].drop_cancelled()
        self._shrink(key)
        batch = self.lanes.get(key)
        if batch is None:
            return
        try:
            emitted, stats = self.runner.decode(batch)
        except Exception as err:
            del self.lanes[key]
            self._crash(batch.seqs, err)
            self._resize(key)
            return
        self.metrics.on_decode_step(stats.rows)
        self.metrics.on_slots("decode", stats.slots, stats.useful)
        self._settle(emitted)
        self._shrink(key)

    def _settle(self, emitted) -> None:
        self.runner.settle(emitted)
        self.metrics.on_tokens(len(emitted))
        for seq, _ in emitted:
            if seq.finished:
                self.metrics.on_finish(seq)

    def _shrink(self, key: int) -> None:
        batch = self.lanes.get(key)
        if batch is not None:
            for seq in batch.seqs:
                if seq.finished:
                    self.metrics.on_finish(seq)
            batch = self.runner.evict(batch)
            if batch is None:
                del self.lanes[key]
            else:
                self.lanes[key] = batch
        self._resize(key)

    def _resize(self, key: int) -> None:
        batch = self.lanes.get(key)
        target = batch.size * batch.horizon if batch is not None else 0
        held = self.reserved.get(key, 0)
        if target < held:
            self.budget.give_back(held - target)
            held = target
        if held:
            self.reserved[key] = held
        else:
            self.reserved.pop(key, None)

    def _crash(self, seqs: list[Sequence], err: Exception) -> None:
        if is_oom(err):
            self.metrics.on_oom()
            log.error("out of memory with %d sequences in flight", len(seqs) + self.rows)
            status, message = 503, "out of memory"
        else:
            log.exception("forward pass failed", exc_info=err)
            status, message = 500, "forward pass failed"
        for seq in seqs:
            if not seq.finished:
                seq.fail(status, message)
            self.metrics.on_finish(seq)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fail_everything(self, status: int, message: str) -> None:
        with self._lock:
            pending, self.waiting = self.waiting, []
        pending += self.running
        self.lanes.clear()
        for seq in pending:
            seq.fail(status, message)
            self.metrics.on_finish(seq)
        self.budget.give_back(sum(self.reserved.values()))
        self.reserved.clear()

    def _publish(self) -> None:
        with self._lock:
            depth = len(self.waiting)
        self.metrics.set_state(depth, self.rows, len(self.lanes), self.budget.used, self.budget.capacity)
