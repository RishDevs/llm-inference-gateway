from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from gateway.config import Settings
from gateway.engine import ModelRunner, Sequence, StepCost, sync
from gateway.model import load
from gateway.scheduler import Scheduler

from .workload import mixed_jobs


class TimedRunner(ModelRunner):
    def __init__(self, base: ModelRunner):
        super().__init__(base.model, base.pad_id, base.eos_ids, base.device, base.dtype)
        self.spent = defaultdict(float)
        self.steps: list[tuple[int, int, int, float]] = []

    def _timed(self, name, fn, *args):
        sync(self.device)
        t = time.perf_counter()
        out = fn(*args)
        sync(self.device)
        dt = time.perf_counter() - t
        self.spent[name] += dt
        return out, dt

    def prefill(self, seqs):
        out, _ = self._timed("prefill", super().prefill, seqs)
        return out

    def decode(self, batch):
        (emitted, stats), dt = self._timed("decode", super().decode, batch)
        self.steps.append((stats.rows, stats.slots, stats.useful, dt))
        return emitted, stats

    def merge(self, running, joining):
        out, _ = self._timed("merge", super().merge, running, joining)
        return out

    def evict(self, batch):
        out, _ = self._timed("evict", super().evict, batch)
        return out


def drive(sched: Scheduler, jobs, users: int) -> float:
    pending = list(jobs)
    active: list[Sequence] = []
    start = time.perf_counter()
    while pending or active:
        active = [s for s in active if not s.finished]
        while pending and len(active) < users:
            job = pending.pop(0)
            seq = Sequence(prompt_ids=list(job.prompt_ids), max_tokens=job.max_tokens, ignore_eos=True)
            sched.submit(seq)
            active.append(seq)
        sched.step()
    return time.perf_counter() - start


def run(base: ModelRunner, settings: Settings, jobs, users: int, costs: StepCost, profiler=None):
    runner = TimedRunner(base)
    sched = Scheduler(runner, settings, budget=None, costs=costs)
    if profiler is not None:
        with profiler:
            wall = drive(sched, jobs, users)
    else:
        wall = drive(sched, jobs, users)
    t = sched.metrics.totals
    prefill_pad = t["prefill_slots"] - t["prefill_useful"]
    kv_pad = t["decode_slots"] - t["decode_useful"]
    return {
        "bucketing": settings.bucketing,
        "wall_s": wall,
        "generated": t["generated_tokens"],
        "throughput_tok_s": t["generated_tokens"] / wall,
        "time_s": dict(runner.spent),
        "microbatches": t["microbatches"],
        "decode_steps": t["decode_steps"],
        "mean_decode_rows": t["decode_rows"] / max(t["decode_steps"], 1),
        "prefill_slots": t["prefill_slots"],
        "prefill_padding": prefill_pad,
        "kv_slots": t["decode_slots"],
        "kv_padding": kv_pad,
        "padded_tokens": prefill_pad + kv_pad,
        "kv_padding_share": kv_pad / max(t["decode_slots"], 1),
        "lanes_opened": next(sched._lane_ids) - 1,
    }, runner.steps


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Profile padding waste with and without length-bucketed batching")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device")
    p.add_argument("--requests", type=int, default=160)
    p.add_argument("--users", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/profile")
    p.add_argument("--trace-requests", type=int, default=40)
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args(argv)

    loaded = load(args.model, args.device)
    base = ModelRunner(loaded.model, loaded.pad_id, loaded.eos_ids, loaded.device, loaded.dtype)
    jobs = mixed_jobs(loaded.tokenizer, args.requests, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    costs = base.calibrate()
    plain = Settings(model=args.model, max_batch_size=args.batch, max_wait_ms=0.0, join_wait_ms=0.0, bucketing=False)
    run(base, plain, jobs[: args.users], args.users, costs)

    activities = [ProfilerActivity.CPU]
    if loaded.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    prof = profile(activities=activities)
    run(base, plain, jobs[: args.trace_requests], args.users, costs, profiler=prof)
    fifo, _ = run(base, plain, jobs, args.users, costs)
    sort_key = "cuda_time_total" if loaded.device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    (out / "fifo_top_ops.txt").write_text(table)
    if not args.no_trace:
        prof.export_chrome_trace(str(out / "fifo_trace.json"))

    bucketed, _ = run(base, replace(plain, bucketing=True), jobs, args.users, costs)

    for result in (fifo, bucketed):
        decode_s = result["time_s"].get("decode", 0.0)
        waste = result["kv_padding"] * costs.per_slot_s
        result["kv_padding_time_s"] = waste
        result["kv_padding_share_of_decode_time"] = waste / max(decode_s, 1e-9)
        prefill_s = result["time_s"].get("prefill", 0.0)
        result["prefill_padding_time_s"] = prefill_s * result["prefill_padding"] / max(result["prefill_slots"], 1)
        total_pad_s = waste + result["prefill_padding_time_s"]
        result["padding_share_of_engine_time"] = total_pad_s / max(sum(result["time_s"].values()), 1e-9)

    report = {
        "model": args.model,
        "device": str(loaded.device),
        "requests": args.requests,
        "users": args.users,
        "batch": args.batch,
        "fifo": fifo,
        "bucketed": bucketed,
        "step_cost": {"fixed_s": costs.fixed_s, "per_kv_slot_s": costs.per_slot_s},
        "padded_token_reduction": 1 - bucketed["padded_tokens"] / max(fifo["padded_tokens"], 1),
        "throughput_ratio": bucketed["throughput_tok_s"] / fifo["throughput_tok_s"],
    }
    (out / "profile.json").write_text(json.dumps(report, indent=2))

    print(table)
    for name, r in (("FIFO", fifo), ("bucketed", bucketed)):
        print(
            f"{name:>9}: {r['throughput_tok_s']:7.1f} tok/s  padded tokens {r['padded_tokens']:>9}  "
            f"(prefill {r['prefill_padding']}, kv {r['kv_padding']})  kv padding share of decode time "
            f"{r['kv_padding_share_of_decode_time']:.1%}  mean decode rows {r['mean_decode_rows']:.1f}  lanes {r['lanes_opened']}"
        )
    print(f"step cost: {costs.fixed_s * 1e3:.2f} ms fixed + {costs.per_slot_s * 1e6:.3f} us per KV slot")
    print(f"padded tokens cut by {report['padded_token_reduction']:.1%}, throughput ratio {report['throughput_ratio']:.2f}x")
    print(f"wrote {out / 'profile.json'}")


if __name__ == "__main__":
    main()
