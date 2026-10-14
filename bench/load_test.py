from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

from .workload import OUTPUT_BUCKETS, PROMPT_BUCKETS, Job, mixed_jobs


@dataclass
class Record:
    concurrency: int
    prompt_tokens: int
    max_tokens: int
    sent: float
    first: float | None = None
    done: float | None = None
    status: int = 0
    tokens: int = 0
    error: str | None = None
    phase: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and self.done is not None


async def fire(client: httpx.AsyncClient, url: str, model: str, job: Job, rec: Record) -> Record:
    payload = {
        "model": model,
        "prompt": job.prompt_ids,
        "max_tokens": job.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    chunks = 0
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=payload) as resp:
            rec.status = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
                rec.error = resp.text[:200]
                rec.done = time.perf_counter()
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                body = json.loads(data)
                if "error" in body:
                    rec.error = str(body["error"])[:200]
                    break
                if body.get("choices"):
                    if rec.first is None:
                        rec.first = time.perf_counter()
                    chunks += 1
                usage = body.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    rec.tokens = usage["completion_tokens"]
    except (httpx.HTTPError, json.JSONDecodeError) as err:
        rec.error = f"{type(err).__name__}: {err}"[:200]
    rec.done = time.perf_counter()
    if rec.tokens == 0:
        rec.tokens = chunks
    return rec


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def summarize(records: list[Record], window: tuple[float, float] | None = None) -> dict:
    good = [r for r in records if r.ok and r.first is not None]
    statuses: dict[str, int] = {}
    for r in records:
        key = str(r.status) if r.error is None or r.status != 200 else "stream_error"
        statuses[key] = statuses.get(key, 0) + 1
    if not good:
        return {"requests": len(records), "completed": 0, "statuses": statuses}
    start = window[0] if window else min(r.sent for r in good)
    end = window[1] if window else max(r.done for r in good)
    span = max(end - start, 1e-9)
    tokens = sum(r.tokens for r in good)
    ttft = [r.first - r.sent for r in good]
    e2e = [r.done - r.sent for r in good]
    itl = [(r.done - r.first) / (r.tokens - 1) for r in good if r.tokens > 1]
    return {
        "requests": len(records),
        "completed": len(good),
        "statuses": statuses,
        "duration_s": span,
        "output_tokens": tokens,
        "throughput_tok_s": tokens / span,
        "requests_per_s": len(good) / span,
        "ttft_p50_s": pct(ttft, 50),
        "ttft_p99_s": pct(ttft, 99),
        "e2e_p50_s": pct(e2e, 50),
        "e2e_p99_s": pct(e2e, 99),
        "itl_p50_s": pct(itl, 50),
        "itl_p99_s": pct(itl, 99),
    }


async def closed_loop(url: str, model: str, jobs: list[Job], concurrency: int, warmup: int) -> list[Record]:
    queue: asyncio.Queue[tuple[int, Job]] = asyncio.Queue()
    for i, job in enumerate(jobs):
        queue.put_nowait((i, job))
    records: list[Record | None] = [None] * len(jobs)
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)

    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        async def worker():
            while True:
                try:
                    i, job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = Record(concurrency, len(job.prompt_ids), job.max_tokens, time.perf_counter())
                records[i] = await fire(client, url, model, job, rec)
                if rec.status == 429:
                    await asyncio.sleep(0.25)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return [r for r in records[warmup:] if r is not None]


async def burst(url: str, model: str, jobs: list[Job], base: int, peak: int, phases: tuple[float, float, float]) -> list[Record]:
    calm, spike, after = phases
    total = calm + spike + after
    records: list[Record] = []
    cursor = 0
    limits = httpx.Limits(max_connections=peak + 8, max_keepalive_connections=peak + 8)
    t0 = time.perf_counter()

    def target(elapsed: float) -> tuple[int, str]:
        if elapsed < calm:
            return base, "steady"
        if elapsed < calm + spike:
            return peak, "burst"
        return base, "recovery"

    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        async def worker(idx: int):
            nonlocal cursor
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= total:
                    return
                active, phase = target(elapsed)
                if idx >= active:
                    await asyncio.sleep(0.05)
                    continue
                job = jobs[cursor % len(jobs)]
                cursor += 1
                rec = Record(active, len(job.prompt_ids), job.max_tokens, time.perf_counter(), phase=phase)
                await fire(client, url, model, job, rec)
                records.append(rec)
                if rec.status in (429, 503):
                    await asyncio.sleep(0.25)

        await asyncio.gather(*(worker(i) for i in range(peak)))
    return records


def server_stats(url: str) -> dict | None:
    try:
        r = httpx.get(f"{url}/stats", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Closed-loop and burst load generator for OpenAI-style completion servers")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--tokenizer")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=ints, default=(1, 8, 16, 32))
    p.add_argument("--requests-per-user", type=int, default=6)
    p.add_argument("--min-requests", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-buckets", type=ints, default=PROMPT_BUCKETS)
    p.add_argument("--output-buckets", type=ints, default=OUTPUT_BUCKETS)
    p.add_argument("--burst", action="store_true")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--peak", type=int, default=64)
    p.add_argument("--phases", type=lambda t: tuple(float(x) for x in t.split(",")), default=(30.0, 60.0, 30.0))
    args = p.parse_args(argv)

    tok = load_tokenizer(args.tokenizer or args.model)
    result = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "mode": "burst" if args.burst else "steady",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "levels": [],
    }

    if args.burst:
        jobs = mixed_jobs(tok, 4000, args.seed, args.prompt_buckets, args.output_buckets)
        recs = asyncio.run(burst(args.url, args.model, jobs, args.base, args.peak, args.phases))
        result["burst"] = {
            "base": args.base,
            "peak": args.peak,
            "phases": list(args.phases),
            "overall": summarize(recs),
            "by_phase": {ph: summarize([r for r in recs if r.phase == ph]) for ph in ("steady", "burst", "recovery")},
        }
        result["records"] = [asdict(r) for r in recs]
        print(json.dumps(result["burst"]["by_phase"], indent=2))
    else:
        for level in args.concurrency:
            count = max(args.min_requests, level * args.requests_per_user)
            jobs = mixed_jobs(tok, count + level, args.seed + level, args.prompt_buckets, args.output_buckets)
            recs = asyncio.run(closed_loop(args.url, args.model, jobs, level, warmup=level))
            summary = summarize(recs)
            result["levels"].append({"concurrency": level, "summary": summary, "records": [asdict(r) for r in recs]})
            tput = summary.get("throughput_tok_s") or 0.0
            ttft = summary.get("ttft_p50_s") or 0.0
            e2e = summary.get("e2e_p99_s") or 0.0
            print(f"[{args.label}] c={level:>3}  {tput:8.1f} tok/s  ttft p50 {ttft * 1000:7.0f} ms  e2e p99 {e2e:6.2f} s  {summary['statuses']}")

    result["server_stats"] = server_stats(args.url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
