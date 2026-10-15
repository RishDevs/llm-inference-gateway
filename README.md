# LLM Inference Gateway

A serving layer that sits in front of Qwen2.5-1.5B and serves many users at once on a single free T4 GPU. It groups requests into micro-batches, streams tokens as they're generated, and limits how much work reaches the GPU so traffic bursts don't crash it.

## What it does

- **Dynamic micro-batching:** new requests are prefilled together as a micro-batch and merged into the running batch at the next decode step. Finished requests leave right away, so nobody waits for someone else's long answer.
- **Length-bucketed batching:** the running set can split into length lanes so short sequences don't carry the padding of long ones. A cost model calibrated on the GPU at startup decides when a split pays for its extra forward pass.
- **Token streaming:** OpenAI-compatible `/v1/completions` with Server-Sent Events.
- **Token-budget admission control and backpressure:** every batch reserves its padded KV-cache footprint against a budget sized from free GPU memory. Prefill size is capped. When the queue is full, the server returns `429` with `Retry-After`.
- **Benchmarks:** closed-loop and burst load tests against a request-at-a-time baseline and against vLLM on the same GPU, plus a PyTorch Profiler run that measures padding waste.

## Architecture

```
client ──HTTP/SSE──▶ FastAPI (asyncio) ──submit──▶ waiting queue
                         ▲                              │  admission: queue cap, token budget, prefill cap
                         │ tokens                       ▼
                 per-request asyncio.Queue ◀── engine thread: prefill micro-batch → merge into lane
                                                       → decode each lane → evict finished → rebalance lanes
                                                       │
                                               Qwen2.5-1.5B, fp16, PyTorch SDPA
```

Details are in [docs/PROJECT.md](docs/PROJECT.md).

## Results

Measured on a Google Colab **Tesla T4 (15 GB)** with Qwen2.5-1.5B-Instruct in fp16. Every request generates exactly `max_tokens` (`ignore_eos`), the client and server share one machine, and each number comes from a single run.

### 32 concurrent users, mixed prompts (64–512 tokens in, 32–256 out)

| | Naive (batch 1) | Gateway | vLLM |
|---|---|---|---|
| Throughput | 23.7 tok/s | **282.1 tok/s (11.9×)** | 723.2 tok/s |
| Median time to first token | 103.6 s | **158 ms** | 117 ms |
| p99 end-to-end latency | 139.3 s | **24.7 s (−82%)** | 8.8 s |
| Requests served | 192 / 192 | 192 / 192 | 192 / 192 |

The gateway reaches **39% of vLLM's throughput** on the same GPU. The gap comes from vLLM's paged KV cache (no padding, no copying when requests join or leave), CUDA graphs, and custom kernels.

### Long, mixed prompts (128–2048 tokens in, 64–512 out), 32 users

| | One batch | Length lanes |
|---|---|---|
| Throughput | 114.5 tok/s | **134.5 tok/s (1.17×)** |
| Median time to first token | 375 ms | 281 ms |
| p99 end-to-end latency | 109.6 s | 90.3 s (−18%) |

With the short-prompt workload, lanes made no difference (279.9 vs 282.1 tok/s): the cost model only splits when padding is expensive enough to pay for the extra forward pass.

### Burst: 32 → 128 → 32 users, prompts up to 3,072 tokens

| | Admission control on | Admission control off |
|---|---|---|
| Peak GPU memory | **7.0 GB / 15** | 14.1 GB / 15 |
| OOM crashes | 0 | 0 |
| Throughput during the spike | 45 tok/s | 91 tok/s |

Admission control kept memory at half the GPU while the unguarded server came within about 1 GB of the limit. The cost is throughput during spikes: reserving every request's worst-case length turns extra load away with `429`. Reserving memory as tokens grow, with preemption, is the planned fix.

## Quick start

### Local (CPU or Apple Silicon)

```bash
git clone https://github.com/RishDevs/llm-inference-gateway.git
cd llm-inference-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m gateway.server --model Qwen/Qwen2.5-0.5B-Instruct
```

```bash
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"The capital of France is","max_tokens":32,"temperature":0,"stream":true}'
```

### Google Colab (free T4)

Open [`notebooks/colab_benchmark.ipynb`](notebooks/colab_benchmark.ipynb) in Colab, choose a T4 runtime, and run all cells. It runs every benchmark and the vLLM comparison, and writes `results/summary.md`.

### Docker (any machine with an NVIDIA GPU)

```bash
docker compose up --build
```

## Server presets

| Preset | What it is |
|---|---|
| `gateway` | Micro-batching + length lanes + admission control (default) |
| `unbucketed` | Same, but one running batch with no length lanes |
| `naive` | One request at a time: batch size 1, no admission control |
| `unguarded` | Large batches with admission control off, used in the burst test |

```bash
python -m gateway.server --preset naive
```

Every setting can also be passed as a flag (`--max-batch-size 16`) or an environment variable (`GATEWAY_MAX_BATCH_SIZE=16`).

## Benchmarks

```bash
bash scripts/run_benchmarks.sh
```

This runs the steady-state sweep (1/8/16/32 users) for `naive`, `unbucketed` and `gateway`, then the 32→64 user burst test, then the profiler. It writes JSON, plots and `results/summary.md`.

## Docs

- [docs/PROJECT.md](docs/PROJECT.md): design and setup
- [docs/SIGNIFICANCE.md](docs/SIGNIFICANCE.md): what the project is and why it matters
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): code walkthrough, file by file
- [docs/DEFENSE.md](docs/DEFENSE.md): talking points, design decisions, Q&A

## License

[MIT](LICENSE)
