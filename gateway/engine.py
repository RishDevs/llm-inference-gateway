from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.profiler import record_function

_ids = itertools.count()


@dataclass(frozen=True)
class Done:
    reason: str


@dataclass(frozen=True)
class Failed:
    status: int
    message: str


@dataclass
class Sequence:
    prompt_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False
    loop: object = None
    outbox: object = None
    rid: int = field(default_factory=lambda: next(_ids))
    arrived: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    finished_at: float | None = None
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    cancelled: bool = False
    reported: bool = False
    admitted: int = 0
    preemptions: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def need(self) -> int:
        return self.prompt_len + self.max_tokens

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def push(self, item) -> None:
        if self.loop is None or self.outbox is None:
            return
        try:
            self.loop.call_soon_threadsafe(self.outbox.put_nowait, item)
        except RuntimeError:
            pass

    def finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finish_reason = reason
        self.finished_at = time.monotonic()
        self.push(Done(reason))

    def fail(self, status: int, message: str) -> None:
        if self.finished:
            return
        self.finish_reason = "error"
        self.finished_at = time.monotonic()
        self.push(Failed(status, message))


@dataclass
class Batch:
    seqs: list[Sequence]
    cache: object = None
    mask: torch.Tensor | None = None
    next_pos: torch.Tensor | None = None
    last: torch.Tensor | None = None
    temps: torch.Tensor | None = None
    top_ps: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return len(self.seqs)

    @property
    def width(self) -> int:
        return 0 if self.mask is None else self.mask.shape[1]

    @property
    def done(self) -> bool:
        return all(s.finished for s in self.seqs)

    @property
    def horizon(self) -> int:
        return max((s.need for s in self.seqs), default=0)

    @property
    def cached(self) -> list[int]:
        return [s.prompt_len + len(s.output_ids) - 1 for s in self.seqs]

    def drop_cancelled(self) -> None:
        for s in self.seqs:
            if s.cancelled and not s.finished:
                s.finish("cancelled")


@dataclass(frozen=True)
class StepCost:
    fixed_s: float
    per_slot_s: float

    def worth_splitting(self, saved_slots: int, factor: float = 1.0) -> bool:
        return saved_slots * self.per_slot_s > self.fixed_s * factor

    def worth_merging(self, added_slots: int, factor: float = 1.0) -> bool:
        return added_slots * self.per_slot_s < 0.5 * self.fixed_s * factor


def padding(lengths: list[int]) -> int:
    return len(lengths) * max(lengths) - sum(lengths) if lengths else 0


def best_cut(lengths: list[int]) -> tuple[list[int], list[int], int]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    ranked = [lengths[i] for i in order]
    whole = padding(ranked)
    best_k, best_saved = 0, 0
    for k in range(1, len(ranked)):
        saved = whole - padding(ranked[:k]) - padding(ranked[k:])
        if saved > best_saved:
            best_k, best_saved = k, saved
    return sorted(order[:best_k]), sorted(order[best_k:]), best_saved


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class StepStats:
    rows: int = 0
    width: int = 0
    slots: int = 0
    useful: int = 0


def _layers(cache):
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(f"unsupported cache type {type(cache).__name__}; transformers>=4.56 is required")
    return layers


def map_kv(cache, fn) -> None:
    for layer in _layers(cache):
        layer.keys = fn(layer.keys)
        layer.values = fn(layer.values)


def zip_kv(dst, src, fn) -> None:
    for a, b in zip(_layers(dst), _layers(src)):
        a.keys = fn(a.keys, b.keys)
        a.values = fn(a.values, b.values)


def stack_left(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    width = max(a.shape[-2], b.shape[-2])
    shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:-2]) + (width, a.shape[-1])
    out = a.new_zeros(shape)
    out[: a.shape[0], ..., width - a.shape[-2]:, :] = a
    out[a.shape[0]:, ..., width - b.shape[-2]:, :] = b
    return out


def stack_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    width = max(a.shape[1], b.shape[1])
    out = a.new_zeros((a.shape[0] + b.shape[0], width))
    out[: a.shape[0], width - a.shape[1]:] = a
    out[a.shape[0]:, width - b.shape[1]:] = b
    return out


class ModelRunner:
    def __init__(self, model, pad_id: int, eos_ids, device, dtype):
        self.model = model
        self.pad_id = pad_id
        self.eos_ids = frozenset(eos_ids)
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def config(self):
        return self.model.config

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence]) -> tuple[Batch, list, StepStats]:
        rows = len(seqs)
        width = max(s.context_len for s in seqs)
        ids = torch.full((rows, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((rows, width), dtype=torch.long)
        for i, s in enumerate(seqs):
            n = s.context_len
            ids[i, width - n:] = torch.tensor(s.prompt_ids + s.output_ids, dtype=torch.long)
            mask[i, width - n:] = 1
        pos = (mask.cumsum(-1) - 1).clamp_(min=0)

        ids, mask, pos = ids.to(self.device), mask.to(self.device), pos.to(self.device)
        with record_function("gateway.prefill"):
            out = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                use_cache=True,
                logits_to_keep=1,
            )
        batch = Batch(
            seqs=list(seqs),
            cache=out.past_key_values,
            mask=mask,
            next_pos=pos[:, -1] + 1,
            temps=torch.tensor([s.temperature for s in seqs], device=self.device, dtype=torch.float32),
            top_ps=torch.tensor([s.top_p for s in seqs], device=self.device, dtype=torch.float32),
        )
        stats = StepStats(rows, width, rows * width, sum(s.context_len for s in seqs))
        picked = sample(out.logits[:, -1, :], batch.temps, batch.top_ps)
        return batch, self._advance(batch, picked), stats

    @torch.inference_mode()
    def decode(self, batch: Batch) -> tuple[list, StepStats]:
        rows = batch.size
        ones = torch.ones((rows, 1), dtype=batch.mask.dtype, device=self.device)
        batch.mask = torch.cat([batch.mask, ones], dim=1)
        with record_function("gateway.decode"):
            out = self.model(
                input_ids=batch.last.view(rows, 1),
                attention_mask=batch.mask,
                position_ids=batch.next_pos.view(rows, 1),
                past_key_values=batch.cache,
                use_cache=True,
            )
        batch.cache = out.past_key_values
        batch.next_pos = batch.next_pos + 1
        stats = StepStats(rows, batch.width, rows * batch.width, sum(batch.cached) + rows)
        picked = sample(out.logits[:, -1, :], batch.temps, batch.top_ps)
        return self._advance(batch, picked), stats

    @record_function("gateway.merge")
    def merge(self, running: Batch | None, joining: Batch) -> Batch:
        if running is None or running.size == 0:
            return joining
        zip_kv(running.cache, joining.cache, stack_left)
        running.mask = stack_mask(running.mask, joining.mask)
        running.seqs = running.seqs + joining.seqs
        running.next_pos = torch.cat([running.next_pos, joining.next_pos])
        running.last = torch.cat([running.last, joining.last])
        running.temps = torch.cat([running.temps, joining.temps])
        running.top_ps = torch.cat([running.top_ps, joining.top_ps])
        joining.cache = None
        return running

    @record_function("gateway.evict")
    def evict(self, batch: Batch) -> Batch | None:
        keep = [i for i, s in enumerate(batch.seqs) if not s.finished]
        if len(keep) == batch.size:
            return batch
        if not keep:
            batch.cache = None
            return None
        return self.take(batch, keep)

    def take(self, batch: Batch, rows: list[int]) -> Batch:
        cached = batch.cached
        lead = batch.width - max(cached[i] for i in rows)
        idx = torch.tensor(rows, device=self.device)
        cache = copy.copy(batch.cache)
        cache.layers = [copy.copy(layer) for layer in batch.cache.layers]
        map_kv(cache, lambda t: t[:, :, lead:, :].index_select(0, idx))
        return Batch(
            seqs=[batch.seqs[i] for i in rows],
            cache=cache,
            mask=batch.mask[:, lead:].index_select(0, idx),
            next_pos=batch.next_pos.index_select(0, idx),
            last=batch.last.index_select(0, idx),
            temps=batch.temps.index_select(0, idx),
            top_ps=batch.top_ps.index_select(0, idx),
        )

    def _time_decode(self, rows: int, width: int, steps: int) -> float:
        seqs = [Sequence(prompt_ids=[self.pad_id + 1] * width, max_tokens=steps + 2, ignore_eos=True) for _ in range(rows)]
        batch, emitted, _ = self.prefill(seqs)
        self.settle(emitted)
        samples = []
        for _ in range(steps):
            sync(self.device)
            t = time.perf_counter()
            emitted, _ = self.decode(batch)
            sync(self.device)
            samples.append(time.perf_counter() - t)
            self.settle(emitted)
        batch.cache = None
        samples.sort()
        return samples[len(samples) // 2]

    def calibrate(self, rows: int = 16, short: int = 32, long: int = 768, steps: int = 5) -> StepCost:
        self._time_decode(rows, short, 2)
        quick = self._time_decode(rows, short, steps)
        slow = self._time_decode(rows, long, steps)
        per_slot = max((slow - quick) / (rows * (long - short)), 0.0)
        fixed = max(quick - per_slot * rows * short, 1e-4)
        return StepCost(fixed, per_slot)

    def _advance(self, batch: Batch, picked: torch.Tensor) -> list[tuple[Sequence, int]]:
        batch.last = picked
        tokens = picked.tolist()
        emitted = []
        now = time.monotonic()
        for s, tok in zip(batch.seqs, tokens):
            if s.finished:
                continue
            s.output_ids.append(tok)
            if s.first_token_at is None:
                s.first_token_at = now
            emitted.append((s, tok))
        return emitted

    def settle(self, emitted: list[tuple[Sequence, int]]) -> None:
        for s, tok in emitted:
            s.push(tok)
            if s.cancelled:
                s.finish("cancelled")
            elif not s.ignore_eos and tok in self.eos_ids:
                s.finish("stop")
            elif len(s.output_ids) >= s.max_tokens:
                s.finish("length")


def sample(logits: torch.Tensor, temps: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    greedy = logits.argmax(dim=-1)
    hot = temps > 0
    if not bool(hot.any()):
        return greedy
    scaled = logits / temps.clamp(min=1e-5).unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    sorted_p, order = probs.sort(dim=-1, descending=True)
    over = (sorted_p.cumsum(dim=-1) - sorted_p) > top_ps.unsqueeze(1)
    sorted_p = sorted_p.masked_fill(over, 0.0)
    choice = torch.multinomial(sorted_p, 1)
    drawn = order.gather(1, choice).squeeze(1)
    return torch.where(hot, drawn, greedy)
