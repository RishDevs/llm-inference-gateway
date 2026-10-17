import json
from dataclasses import replace

from fastapi.testclient import TestClient

from gateway.config import Settings, apply_preset, from_env
from gateway.protocol import TextStream
from gateway.scheduler import Scheduler
from gateway.server import create_app, parse_args


def client_for(runner, tokenizer, settings, budget=None):
    sched = Scheduler(runner, settings, budget=budget)
    return TestClient(create_app(settings, sched, tokenizer)), sched


def events(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(line[6:])
    return out


def test_streaming_completion(runner, tokenizer, settings):
    client, _ = client_for(runner, tokenizer, settings)
    with client:
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "hello",
                "max_tokens": 6,
                "temperature": 0,
                "stream": True,
                "ignore_eos": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert r.status_code == 200
        data = events(r.text)
        assert data[-1] == "[DONE]"
        bodies = [json.loads(d) for d in data[:-1]]
        assert bodies[-1]["usage"]["completion_tokens"] == 6
        finals = [b for b in bodies if b["choices"] and b["choices"][0]["finish_reason"]]
        assert finals[-1]["choices"][0]["finish_reason"] == "length"


def test_non_streaming_and_token_prompt(runner, tokenizer, settings):
    client, _ = client_for(runner, tokenizer, settings)
    with client:
        r = client.post("/v1/completions", json={"prompt": [5, 6, 7, 8], "max_tokens": 4, "temperature": 0, "ignore_eos": True})
        assert r.status_code == 200
        body = r.json()
        assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}


def test_too_large_is_400(runner, tokenizer, settings):
    client, _ = client_for(runner, tokenizer, settings)
    with client:
        r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 10_000})
        assert r.status_code == 400


def test_queue_full_is_429(runner, tokenizer, settings):
    tight = replace(settings, max_queue=0)
    client, _ = client_for(runner, tokenizer, tight)
    with client:
        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        assert r.status_code == 429
        assert r.headers["retry-after"] == "1"


def test_health_stats_metrics(runner, tokenizer, settings):
    client, _ = client_for(runner, tokenizer, settings)
    with client:
        assert client.get("/health").json()["status"] == "ok"
        client.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3, "ignore_eos": True})
        stats = client.get("/stats").json()
        assert stats["generated_tokens"] == 3
        assert "gateway_generated_tokens_total" in client.get("/metrics").text


class Chars:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_text_stream_stop_string():
    ts = TextStream(Chars(), ["END"])
    out = "".join(ts.feed(ord(c)) for c in "abcENDxyz")
    assert out == "abc"
    assert ts.stopped


def test_presets_and_cli():
    naive = apply_preset(Settings(), "naive")
    assert naive.max_batch_size == 1 and not naive.admission
    s = parse_args(["--preset", "unbucketed", "--max-batch-size", "8", "--model", "m"])
    assert s.preset == "unbucketed" and not s.bucketing and s.max_batch_size == 8 and s.model == "m"
    env = from_env({"GATEWAY_PRESET": "unguarded", "GATEWAY_MAX_QUEUE": "9", "GATEWAY_BUCKETING": "false"})
    assert env.max_batch_size == 256 and env.max_queue == 9 and env.bucketing is False


def test_latency_breakdown_and_reset(runner, tokenizer, settings):
    client, _ = client_for(runner, tokenizer, settings)
    with client:
        client.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3, "ignore_eos": True})
        lat = client.get("/stats").json()["server_latency"]
        assert lat["samples"] == 1
        assert lat["queue_p50_s"] >= 0 and lat["prefill_p50_s"] >= 0
        assert abs(lat["queue_p50_s"] + lat["prefill_p50_s"] - lat["ttft_p50_s"]) < 1e-6
        assert client.post("/stats/reset").json()["status"] == "ok"
        assert client.get("/stats").json()["server_latency"]["samples"] == 0
