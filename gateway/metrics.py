from __future__ import annotations

import threading

import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30, 60)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("gateway_requests", "Finished requests by outcome", ["outcome"], registry=r)
        self.ttft = Histogram("gateway_ttft_seconds", "Arrival to first token", buckets=LATENCY_BUCKETS, registry=r)
        self.e2e = Histogram("gateway_e2e_seconds", "Arrival to last token", buckets=LATENCY_BUCKETS, registry=r)
        self.tokens = Counter("gateway_generated_tokens", "Tokens sent to clients", registry=r)
        self.slots = Counter("gateway_token_slots", "Token positions computed", ["phase", "kind"], registry=r)
        self.microbatch_size = Histogram(
            "gateway_prefill_microbatch_rows", "Rows per prefill micro-batch", buckets=(1, 2, 4, 8, 16, 32, 64), registry=r
        )
        self.batch_rows = Histogram(
            "gateway_decode_batch_rows", "Rows per decode step", buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 128), registry=r
        )
        self.lane_events = Counter("gateway_lane_events", "Lane splits and merges", ["kind"], registry=r)
        self.preemptions = Counter("gateway_preemptions", "Sequences paused to free KV memory", registry=r)
        self.ooms = Counter("gateway_oom", "CUDA out-of-memory errors caught", registry=r)
        self.queue_depth = Gauge("gateway_queue_depth", "Requests waiting for a batch", registry=r)
        self.running = Gauge("gateway_running_rows", "Sequences being decoded", registry=r)
        self.lane_count = Gauge("gateway_lanes", "Length lanes currently decoding", registry=r)
        self.budget_used = Gauge("gateway_budget_used_tokens", "Reserved KV tokens", registry=r)
        self.budget_capacity = Gauge("gateway_budget_capacity_tokens", "Token budget (-1 = unlimited)", registry=r)

        self._lock = threading.Lock()
        self.totals = {
            "requests": {},
            "generated_tokens": 0,
            "prefill_slots": 0,
            "prefill_useful": 0,
            "decode_slots": 0,
            "decode_useful": 0,
            "microbatches": 0,
            "decode_steps": 0,
            "decode_rows": 0,
            "ooms": 0,
            "preemptions": 0,
            "lane_splits": 0,
            "lane_merges": 0,
            "rejected": 0,
        }
        self.state = {"queue_depth": 0, "running_rows": 0, "lanes": 0, "budget_used": 0, "budget_capacity": -1}

    def set_state(self, queue_depth: int, running_rows: int, lanes: int, used: int, capacity: int | None) -> None:
        cap = -1 if capacity is None else capacity
        self.queue_depth.set(queue_depth)
        self.running.set(running_rows)
        self.lane_count.set(lanes)
        self.budget_used.set(used)
        self.budget_capacity.set(cap)
        with self._lock:
            self.state.update(
                queue_depth=queue_depth, running_rows=running_rows, lanes=lanes, budget_used=used, budget_capacity=cap
            )

    def on_microbatch(self, rows: int) -> None:
        self.microbatch_size.observe(rows)
        with self._lock:
            self.totals["microbatches"] += 1

    def on_decode_step(self, rows: int) -> None:
        self.batch_rows.observe(rows)
        with self._lock:
            self.totals["decode_steps"] += 1
            self.totals["decode_rows"] += rows

    def on_slots(self, phase: str, slots: int, useful: int) -> None:
        self.slots.labels(phase, "useful").inc(useful)
        self.slots.labels(phase, "padding").inc(slots - useful)
        with self._lock:
            self.totals[f"{phase}_slots"] += slots
            self.totals[f"{phase}_useful"] += useful

    def on_tokens(self, n: int) -> None:
        self.tokens.inc(n)
        with self._lock:
            self.totals["generated_tokens"] += n

    def on_finish(self, seq) -> None:
        if seq.reported:
            return
        seq.reported = True
        outcome = seq.finish_reason or "unknown"
        self.requests.labels(outcome).inc()
        if seq.first_token_at is not None:
            self.ttft.observe(seq.first_token_at - seq.arrived)
        if seq.finished_at is not None and seq.output_ids:
            self.e2e.observe(seq.finished_at - seq.arrived)
        with self._lock:
            bucket = self.totals["requests"]
            bucket[outcome] = bucket.get(outcome, 0) + 1

    def on_reject(self) -> None:
        self.requests.labels("rejected").inc()
        with self._lock:
            self.totals["rejected"] += 1

    def on_lanes(self, kind: str) -> None:
        self.lane_events.labels(kind).inc()
        with self._lock:
            self.totals[f"lane_{kind}s"] += 1

    def on_preempt(self) -> None:
        self.preemptions.inc()
        with self._lock:
            self.totals["preemptions"] += 1

    def on_oom(self) -> None:
        self.ooms.inc()
        with self._lock:
            self.totals["ooms"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.totals.items()}
            data.update(self.state)
        if torch.cuda.is_available():
            data["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
            data["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        return data

    def render(self) -> bytes:
        return generate_latest(self.registry)
