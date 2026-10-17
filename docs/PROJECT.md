# LLM Inference Gateway: Project Documentation

**Status:** implemented. Every component below exists in the repo and is covered by tests. Benchmark numbers are produced by running `scripts/run_benchmarks.sh` on a GPU; this document doesn't quote results.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [What it measures](#2-what-it-measures)
3. [Architecture](#3-architecture)
4. [Components](#4-components)
5. [Server presets and baselines](#5-server-presets-and-baselines)
6. [Benchmark methodology](#6-benchmark-methodology)
7. [Tech stack and cost](#7-tech-stack-and-cost)
8. [Repository layout](#8-repository-layout)
9. [Setup and running](#9-setup-and-running)
10. [Configuration reference](#10-configuration-reference)
11. [Design decisions](#11-design-decisions)
12. [Known constraints](#12-known-constraints)

---

## 1. What this project is

A serving layer for one LLM on one GPU. It accepts OpenAI-style completion requests over HTTP and schedules them onto the model in batches. It streams tokens back over Server-Sent Events and refuses work it can't fit in memory.

| Function | How |
|---|---|
| Dynamic micro-batching | Waiting requests are prefilled together as a micro-batch, then merged into a running batch at the next step boundary. Finished requests are evicted from the batch right away. |
| Length-bucketed batching | The running set can be split into up to `max_lanes` lanes by context length, so short sequences don't carry the padding of long ones. A cost model calibrated at startup decides when splitting or merging lanes pays off. |
| Token streaming | Each token goes to its client as soon as it's sampled: `POST /v1/completions` with `stream: true`. |
| Token-budget admission control | Each lane reserves `rows × (width + 1)` KV slots, what its next step needs, against a budget sized from free GPU memory. When the budget is short, the newest request is preempted and later resumed by re-prefilling its prompt plus generated tokens. A prefill micro-batch can't exceed `max_prefill_tokens` padded tokens. |
| Backpressure | When `max_queue` requests are already waiting, new ones get `429` with `Retry-After`. Requests that wait longer than `queue_timeout_s` get `503`. |
| Benchmarks | A closed-loop sweep, a burst test and a PyTorch Profiler run, compared against a request-at-a-time baseline and vLLM. |

**Model:** Qwen2.5-1.5B-Instruct in fp16 by default. Any Hugging Face causal LM whose cache is a `DynamicCache` with per-layer `keys`/`values` tensors works.

**Out of scope:** multi-GPU serving, quantization, LoRA, authentication, chat templates (the API is raw completions).

---

## 2. What it measures

| Metric | Definition | Where it comes from |
|---|---|---|
| Throughput | Output tokens per second over the measurement window | `bench/load_test.py` |
| TTFT | Request sent → first streamed chunk received, p50 and p99 | `bench/load_test.py` |
| End-to-end latency | Request sent → last chunk received, p50 and p99 | `bench/load_test.py` |
| Inter-token latency | (last chunk − first chunk) / (tokens − 1), p50 and p99 | `bench/load_test.py` |
| OOM count | CUDA out-of-memory errors caught by the engine | `/stats` → `ooms` |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` | `/stats` → `gpu_peak_bytes` |
| Padded tokens | Prefill: `rows × width − Σ prompt_len`. Decode: `rows × cache_width − Σ cached_len`, summed per step | `/stats`, `bench/profile_decode.py` |
| Padding share of decode time | KV padding slots × calibrated cost per slot ÷ total decode time | `bench/profile_decode.py` |

`bench/report.py` turns these into `results/summary.md` with these headline rows at 32 users:
- micro-batching vs naive
- bucketed vs unbucketed
- gateway vs vLLM
- TTFT naive vs gateway
- p99 reduction
- padding figures from the profiler

---

## 3. Architecture

```
 client ── POST /v1/completions ──▶ FastAPI handler (asyncio event loop)
                                       │ tokenize, build Sequence(outbox = asyncio.Queue)
                                       │ Scheduler.submit()  ── 400 too large / 429 queue full
                                       ▼
                               waiting list (locked)
                                       │
 ┌──────────────────── engine thread (owns the GPU) ─────────────────────┐
 │ loop:                                                                  │
 │   expire stale waiting requests ─────────────────────────────▶ 503     │
 │   admit: pick a micro-batch for one lane (budget, prefill cap,         │
 │          free rows) → prefill → merge into the lane → evict finished   │
 │   decode: one forward pass per lane → sample → push tokens → evict     │
 │   every N steps: rebalance lanes (split or merge using the cost model) │
 └────────────────────────────────────────────────────────────────────────┘
                                       │ loop.call_soon_threadsafe(outbox.put_nowait, token)
                                       ▼
 FastAPI handler: incremental detokenize → stop strings → SSE chunks → data: [DONE]
```

- **Threads:** the asyncio event loop handles HTTP and streaming. One engine thread runs every GPU call. They communicate only through the locked waiting list, per-request `asyncio.Queue`s, and a `cancelled` flag.
- **Batch state:** each lane is a `Batch`: a list of sequences plus a left-padded KV cache `[rows, kv_heads, width, head_dim]` per layer, an attention mask `[rows, width]`, and per-row next position, last token, temperature and top-p.

---

## 4. Components

### 4.1 API server: `gateway/server.py`, `gateway/protocol.py`

- `POST /v1/completions` takes these fields:
  - `prompt` (string or list of token ids)
  - `max_tokens`, `temperature`, `top_p`
  - `stream`, `stop`
  - `ignore_eos`
  - `stream_options.include_usage`
- `GET /health`, `GET /v1/models`, `GET /metrics` (Prometheus), `GET /stats` (JSON snapshot used by the benchmarks).
- The handler waits for the first item from the engine *before* opening the stream. Queue timeouts and OOMs therefore come back as real HTTP status codes (503), not as a 200 with an error inside.
- `TextStream` decodes the full token list on each new token and holds back text that ends in an incomplete UTF-8 character. It also holds back the last `len(stop) − 1` characters, so a stop string split across tokens is still caught.
- When a client disconnects, the request is cancelled and its row is evicted on the next step.

### 4.2 Scheduler: `gateway/scheduler.py`

One `step()` does the following:

1. `_expire`: fails requests that have waited longer than `queue_timeout_s`.
2. `_admit`: if there are free rows and the oldest waiting request is ready, routes it to a lane, builds a micro-batch from waiting requests routed to the same lane, reserves budget, prefills, merges, and evicts anything that already finished.
   - *Ready* means either the oldest has waited `max_wait_ms` (idle) or `join_wait_ms` (busy), or enough requests are waiting to fill the free rows.
3. `_decode`: one forward pass per lane.
4. Every `rebalance_every` steps, `rebalance`:
   - splits the most-padded lane at the length cut that saves the most padding, if that saving is worth more than one extra forward pass;
   - otherwise merges the two adjacent lanes whose combined padding grows the least, if that growth costs less than half a forward pass.

The oldest waiting request always seeds the next micro-batch, so nothing starves.

### 4.3 Engine: `gateway/engine.py`

- **`prefill(seqs)`:** left-pads prompts into `[rows, width]` and sets `position_ids = cumsum(mask) − 1` so padding doesn't shift positions. Runs the model with `logits_to_keep=1`, so only the last position's logits are computed (vocab is 151,936, so full prefill logits would be huge). Returns a new `Batch` and the first token for each row.
- **`decode(batch)`:** appends a `1` column to the mask and feeds each row's last token at its own position. HF's `DynamicCache` appends the new K/V.
- **`merge(a, b)`:** builds one zero tensor of the combined shape and copies each batch into it right-aligned (`stack_left`). One copy per layer.
- **`evict(batch)` / `take(batch, rows)`:** `index_select`s the kept rows and drops leading columns that are now padding for every remaining row. One copy per layer.
- **`calibrate()`:** times decode steps at two cache widths to fit `step_time ≈ fixed + per_slot × rows × width`.
- **`sample()`:** greedy where `temperature == 0`; otherwise temperature plus top-p via sort/cumsum/multinomial, vectorized across rows.

### 4.4 Admission: `gateway/admission.py`

- `TokenBudget`: a thread-safe counter of reserved KV slots.
- `size_budget`: `(free_gpu_bytes × kv_memory_fraction − max_prefill_tokens × activation_bytes_per_token) / kv_bytes_per_token`.
  - For Qwen2.5-1.5B, `kv_bytes_per_token = 2 × 28 layers × 2 kv_heads × 128 head_dim × 2 bytes = 28,672`.
- In the default `grow` mode a lane reserves `rows × (width + 1)` before each decode step, exactly what that step holds, and preempts the newest request if the budget can't cover it. In `worst` mode (`worstcase` preset) it reserves `rows × max(prompt + max_tokens)` up front instead.
- The per-token cost includes the temporary attention copy (KV heads expanded to all query heads), not just the KV cache itself.

### 4.5 Metrics: `gateway/metrics.py`

Prometheus counters and histograms (TTFT, end-to-end latency, tokens, padding slots, micro-batch size, decode batch size, lane events, OOMs, rejections), plus a plain dict mirror served at `/stats`.

### 4.6 Benchmarks: `bench/`

| File | What it does |
|---|---|
| `workload.py` | Deterministic mixed-length jobs. Prompt lengths are drawn from 64/128/256/512-token buckets and outputs from 32/64/128/256-token buckets, with each length uniform within the upper half of its bucket. Prompts are sent as token ids, so lengths are exact. |
| `load_test.py` | Closed-loop sweep (N users, each sends its next request when the last finishes, first N requests discarded as warm-up). Burst mode steps from `base` to `peak` users and back. |
| `profile_decode.py` | Runs the real scheduler in-process with 32 simulated users, with lanes off and on. Records a PyTorch Profiler trace and op table for the FIFO run, times each engine phase, counts padded tokens, and converts KV padding into time using the calibrated per-slot cost. |
| `report.py` | Builds `results/summary.md` and three plots. |

---

## 5. Server presets and baselines

| Preset | Batch size | Lanes | Admission | Purpose |
|---|---|---|---|---|
| `naive` | 1 | 1 | off | Request-at-a-time baseline, same code path |
| `unbucketed` | 32 | 1 | on | Micro-batching without length bucketing |
| `worstcase` | 32 | up to 3 | on, worst-case reservation | Admission control without preemption |
| `gateway` | 32 | up to 3, cost-model driven | on | Full system |
| `unguarded` | 256 | up to 3 | off | Burst test without admission control |
| vLLM | its own scheduler | — | its own | Reference engine, same model, `--dtype half` |

The naive baseline is the same server with batch size 1. Tokenizer, sampling, streaming and HTTP are identical, so the comparison isolates batching.

---

## 6. Benchmark methodology

- **Same inputs for every system:** same model and dtype, same token-id prompts, `temperature = 0`, `ignore_eos = true`. Every request generates exactly `max_tokens`, so output length doesn't depend on the system.
- **Steady state:** 1, 8, 16 and 32 users, 6 measured requests per user (minimum 24), after one warm-up request per user.
- **Burst:** 32 users for 30 s, 64 users for 60 s, back to 32 for 30 s. The burst uses longer prompts (up to 1536 tokens) and outputs (up to 384) to put pressure on memory. It runs for `gateway` and `unguarded`. Status codes, server-side OOM count and peak GPU memory are recorded.
- **Profiler:** 160 requests, 32 simulated users, batch 32, lanes off vs on, on the same model and GPU.
- **Same host:** the server and the client run on the same machine over `localhost`, so network latency is excluded.

---

## 7. Tech stack and cost

| Layer | Choice | License | Cost |
|---|---|---|---|
| Language | Python ≥ 3.10 | PSF | $0 |
| Inference | PyTorch (SDPA attention), Hugging Face `transformers` ≥ 4.56 | BSD-3 / Apache-2.0 | $0 |
| API | FastAPI, uvicorn, pydantic | MIT / BSD | $0 |
| Metrics | prometheus-client | Apache-2.0 | $0 |
| Load testing | httpx, numpy | BSD | $0 |
| Plots | matplotlib | PSF-based | $0 |
| Reference | vLLM | Apache-2.0 | $0 |
| Model | Qwen2.5-1.5B-Instruct | Apache-2.0 | $0 |
| GPU | Google Colab free tier (T4, 16 GB) | — | $0 |
| CI | GitHub Actions (public repo, CPU tests) | — | $0 |
| Container | Docker, `pytorch/pytorch` CUDA runtime image | — | $0 |

Colab Pro's L4/A100 work too but aren't needed. Report the GPU with every result, because numbers from different GPUs aren't comparable. Kaggle (T4 ×2, about 30 GPU-hours a week) is the free fallback.

---

## 8. Repository layout

```
gateway/
  config.py        Settings dataclass, presets, env parsing
  model.py         device/dtype choice, model loading, KV and activation size math
  engine.py        Sequence, Batch, ModelRunner (prefill, decode, merge, evict, take, calibrate), sampling
  scheduler.py     admission, micro-batch selection, lanes, rebalancing, failure handling
  admission.py     TokenBudget, budget sizing, QueueFull / RequestTooLarge
  metrics.py       Prometheus metrics + /stats snapshot
  protocol.py      request schema, incremental detokenizer, SSE chunk builders
  server.py        FastAPI app, routes, CLI
bench/
  workload.py      mixed-length job generator
  load_test.py     closed-loop and burst load generator
  profile_decode.py  profiler and padding analysis
  report.py        summary.md + plots
scripts/run_benchmarks.sh   full benchmark sequence
notebooks/colab_benchmark.ipynb
tests/             engine equivalence, scheduler, server tests (tiny random model, CPU)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

---

## 9. Setup and running

### 9.1 Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m gateway.server --model Qwen/Qwen2.5-0.5B-Instruct
```

The tests use a randomly initialized 2-layer Qwen2 model on CPU, so they download nothing. On Apple Silicon the server runs on `mps`.

### 9.2 Colab (free T4)

1. Open `notebooks/colab_benchmark.ipynb` and choose **Runtime → Change runtime type → T4 GPU**.
2. Run all cells. It installs the package and runs `scripts/run_benchmarks.sh` (naive, unbucketed, gateway, both burst runs, profiler). Then it installs vLLM, benchmarks it on port 8001, builds the report, and downloads `results.zip`.

vLLM is installed last because it pins its own torch build. On a T4 it must run with `--dtype half`, since bf16 needs compute capability 8.0.

### 9.3 Docker

```bash
docker compose up --build
```

Needs an NVIDIA GPU and the NVIDIA Container Toolkit. The model cache is a named volume. Choose the preset with `GATEWAY_PRESET`.

### 9.4 Manual request

```bash
curl -N localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"Explain KV caching in one sentence.","max_tokens":64,"temperature":0,"stream":true}'
```

---

## 10. Configuration reference

Every field can be set with a CLI flag (`--max-batch-size`) or an env var (`GATEWAY_MAX_BATCH_SIZE`). `--preset` applies first, then explicit flags override it.

| Setting | Default | Meaning |
|---|---|---|
| `model` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model id |
| `max_model_len` | 2048 | Max `prompt + max_tokens` per request |
| `max_batch_size` | 32 | Max sequences decoding at once, across all lanes |
| `max_lanes` | 3 | Max length lanes |
| `lane_split_factor` | 1.5 | Safety margin on the cost model: a split must save 1.5× a forward pass |
| `rebalance_every` | 8 | Steps between lane rebalances |
| `max_wait_ms` | 30 | When idle, how long the first request waits for others to batch with |
| `join_wait_ms` | 15 | When busy, how long new requests wait to join as a group |
| `max_prefill_tokens` | 8192 | Max padded tokens in one prefill micro-batch |
| `token_budget` | 0 (auto) | KV slot budget; 0 sizes it from free GPU memory |
| `kv_memory_fraction` | 0.85 | Share of free memory the budget may use |
| `reserve` | `grow` | `grow`: reserve one step ahead and preempt when short. `worst`: reserve each request's `prompt + max_tokens` up front |
| `watermark` | 0.05 | In `grow` mode, share of the budget kept free when admitting new requests |
| `max_queue` | 48 | Waiting requests before 429 |
| `queue_timeout_s` | 30 | Max wait before 503 |
| `bucketing` | true | Enable lanes and lane-grouped prefill |
| `admission` | true | Enable budget, prefill cap and queue limit |

---

## 11. Design decisions

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Batching granularity | Join at step boundaries, evict finished rows | Run each micro-batch to completion | Run-to-completion makes every row wait for the longest one. In early local tests it barely beat naive. |
| KV layout | HF `DynamicCache`, left-padded, one tensor per layer per lane | Paged KV with a custom attention kernel | Paged attention needs custom kernels. Left padding works with stock SDPA and keeps the code reviewable. Padding is then the cost to manage, and the lanes manage it. |
| Length bucketing | Lanes split and merged by a calibrated cost model | Fixed buckets; no bucketing | Fixed buckets cut padding but add a forward pass per bucket. Whether that pays off depends on the GPU and model, so the engine measures it instead of guessing. |
| Memory reservation | Reserve one step ahead, preempt the newest request when short | Reserve each request's worst case up front (`worstcase` preset) | Worst-case reservation over-reserves because most requests are far from their limit. Reserve-ahead uses only what's needed; preemption by recompute keeps outputs identical. |
| Concurrency | Single process: asyncio for HTTP, one engine thread | Separate API and engine processes | No inter-process communication or serialization. The GIL is the tradeoff, and it's watched via CPU usage. |
| Baseline | Same server with batch size 1 | Separate `model.generate` server | Isolates batching as the only variable. |
| Streaming | SSE, OpenAI-compatible | WebSocket | The same client can benchmark the gateway and vLLM. |
| Attention | PyTorch SDPA | FlashAttention-2 | FA2 doesn't support the T4. |

---

## 12. Known constraints

- **GQA under a padding mask:** HF's SDPA path expands the KV heads to all query heads when a padding mask is present (6× for Qwen2.5-1.5B). That makes padded slots expensive. It's the reason lanes exist and why the profiler measures padding time.
- **DynamicCache growth:** the cache appends with `torch.cat`, so every decode step copies the whole cache. vLLM avoids this with preallocated paged blocks. This is part of the gap to vLLM.
- **Python per-step overhead:** HF's forward pass runs Python for every layer on every step. vLLM uses CUDA graphs. This also contributes to the gap.
- **Colab has 2 vCPUs:** the load generator, the HTTP server, detokenization and the engine loop share them. Watch CPU usage during runs.
- **Burst OOM depends on workload:** whether `unguarded` actually runs out of memory depends on prompt lengths and GPU memory. The burst test uses long prompts for that reason. The claim to check is zero OOMs with admission on.
- **vLLM on T4:** check that the installed vLLM version supports compute capability 7.5. Older guidance says T4 runs the V0 engine, and some models need SM80. Pin an earlier vLLM if the latest refuses the GPU.
