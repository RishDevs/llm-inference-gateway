from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import config as cfg
from .admission import QueueFull, RequestTooLarge, size_budget
from .engine import Done, Failed, ModelRunner, Sequence
from .metrics import Metrics
from .protocol import CompletionRequest, TextStream, chunk, completion, error_body, usage_chunk
from .scheduler import Scheduler

log = logging.getLogger("gateway")

FINISH_MAP = {"stop": "stop", "length": "length", "cancelled": "abort"}


def build_engine(settings: cfg.Settings):
    from .model import load

    loaded = load(settings.model, settings.device)
    runner = ModelRunner(loaded.model, loaded.pad_id, loaded.eos_ids, loaded.device, loaded.dtype)
    budget = size_budget(settings, loaded.model.config, loaded.dtype, loaded.device)
    costs = runner.calibrate() if settings.bucketing else None
    scheduler = Scheduler(runner, settings, Metrics(), budget, costs)
    log.info(
        "loaded %s on %s (%s), preset=%s, token budget=%s",
        settings.model, loaded.device, loaded.dtype, settings.preset, budget if budget is not None else "unlimited",
    )
    if costs is not None:
        log.info("decode step cost: %.2f ms fixed + %.3f us per KV slot", costs.fixed_s * 1e3, costs.per_slot_s * 1e6)
    return scheduler, loaded.tokenizer


def create_app(settings: cfg.Settings, scheduler: Scheduler | None = None, tokenizer=None) -> FastAPI:
    state = {"scheduler": scheduler, "tokenizer": tokenizer}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["scheduler"] is None:
            state["scheduler"], state["tokenizer"] = await asyncio.to_thread(build_engine, settings)
        state["scheduler"].start()
        yield
        state["scheduler"].stop()

    app = FastAPI(title="llm-inference-gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": settings.name, "preset": settings.preset}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": settings.name, "object": "model", "owned_by": "local"}]}

    @app.get("/metrics")
    async def metrics():
        return Response(state["scheduler"].metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    async def stats():
        sched = state["scheduler"]
        snap = sched.metrics.snapshot()
        snap["preset"] = settings.preset
        snap["settings"] = {
            "max_batch_size": settings.max_batch_size,
            "max_lanes": settings.max_lanes,
            "lane_split_factor": settings.lane_split_factor,
            "max_wait_ms": settings.max_wait_ms,
            "join_wait_ms": settings.join_wait_ms,
            "max_prefill_tokens": settings.max_prefill_tokens,
            "bucketing": settings.bucketing,
            "admission": settings.admission,
            "reserve": settings.reserve,
        }
        return snap

    @app.post("/stats/reset")
    async def reset_stats():
        state["scheduler"].metrics.reset_samples()
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        sched: Scheduler = state["scheduler"]
        tok = state["tokenizer"]
        if isinstance(body.prompt, str):
            prompt_ids = tok.encode(body.prompt)
        else:
            prompt_ids = list(body.prompt)

        loop = asyncio.get_running_loop()
        seq = Sequence(
            prompt_ids=prompt_ids,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            ignore_eos=body.ignore_eos,
            loop=loop,
            outbox=asyncio.Queue(),
        )
        try:
            sched.submit(seq)
        except QueueFull as err:
            return JSONResponse(
                error_body("server is at capacity, retry later", "rate_limit"),
                status_code=429,
                headers={"Retry-After": str(err.retry_after)},
            )
        except RequestTooLarge as err:
            return JSONResponse(error_body(str(err), "invalid_request"), status_code=400)

        first = await _first_item(seq, request, sched)
        if first is None:
            return Response(status_code=499)
        if isinstance(first, Failed):
            return JSONResponse(error_body(first.message, "server_error"), status_code=first.status)

        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        stream = TextStream(tok, body.stop_list())

        if not body.stream:
            text, reason = await _collect(seq, first, stream, sched)
            return completion(rid, settings.name, text, reason, _usage(seq))

        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            _sse(seq, first, stream, sched, rid, settings.name, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _first_item(seq: Sequence, request: Request, sched: Scheduler):
    while True:
        try:
            return await asyncio.wait_for(seq.outbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if await request.is_disconnected():
                sched.cancel(seq)
                return None


def _usage(seq: Sequence) -> dict:
    n = len(seq.output_ids)
    return {"prompt_tokens": seq.prompt_len, "completion_tokens": n, "total_tokens": seq.prompt_len + n}


async def _collect(seq: Sequence, first, stream: TextStream, sched: Scheduler):
    item = first
    while True:
        if isinstance(item, int):
            stream.feed(item)
            if stream.stopped:
                sched.cancel(seq)
                return stream.emitted, "stop"
        elif isinstance(item, Done):
            stream.close()
            return stream.emitted, FINISH_MAP.get(item.reason, item.reason)
        elif isinstance(item, Failed):
            return stream.emitted, "error"
        item = await seq.outbox.get()


async def _sse(seq: Sequence, first, stream: TextStream, sched: Scheduler, rid: str, model: str, include_usage: bool):
    item = first
    done = False
    try:
        while True:
            if isinstance(item, int):
                piece = stream.feed(item)
                if piece:
                    yield chunk(rid, model, piece)
                if stream.stopped:
                    sched.cancel(seq)
                    yield chunk(rid, model, "", "stop")
                    break
            elif isinstance(item, Done):
                tail = stream.close()
                yield chunk(rid, model, tail, FINISH_MAP.get(item.reason, item.reason))
                break
            elif isinstance(item, Failed):
                yield f"data: {json.dumps({'error': {'message': item.message, 'code': item.status}})}\n\n"
                break
            item = await seq.outbox.get()
        done = True
        if include_usage:
            yield usage_chunk(rid, model, _usage(seq))
        yield "data: [DONE]\n\n"
    finally:
        if not done or not seq.finished:
            sched.cancel(seq)


def parse_args(argv=None) -> cfg.Settings:
    p = argparse.ArgumentParser(description="LLM inference gateway")
    p.add_argument("--preset", choices=sorted(cfg.PRESETS), default=None)
    p.add_argument("--model")
    p.add_argument("--served-name")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--device")
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-batch-size", type=int)
    p.add_argument("--max-lanes", type=int)
    p.add_argument("--lane-split-factor", type=float)
    p.add_argument("--max-wait-ms", type=float)
    p.add_argument("--join-wait-ms", type=float)
    p.add_argument("--max-prefill-tokens", type=int)
    p.add_argument("--token-budget", type=int)
    p.add_argument("--reserve", choices=["grow", "worst"])
    p.add_argument("--watermark", type=float)
    p.add_argument("--max-queue", type=int)
    p.add_argument("--queue-timeout-s", type=float)
    p.add_argument("--bucketing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--admission", action=argparse.BooleanOptionalAction, default=None)
    args = vars(p.parse_args(argv))

    settings = cfg.from_env()
    preset = args.pop("preset")
    if preset:
        settings = cfg.apply_preset(settings, preset)
    overrides = {k: v for k, v in args.items() if v is not None}
    return replace(settings, **overrides).validate()


def main(argv=None) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = parse_args(argv)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
