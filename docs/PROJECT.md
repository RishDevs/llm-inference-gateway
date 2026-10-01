# LLM Inference Gateway — Project Documentation

**Status:** Design stage. No code exists yet. This document describes what the project will build, how it will be built, and how to run it at zero cost. Sections marked **Open decision** list the free options and their tradeoffs. Nothing in them has been chosen yet.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Goals and how they are measured](#2-goals-and-how-they-are-measured)
3. [Architecture](#3-architecture)
4. [Components](#4-components)
5. [Baselines](#5-baselines)
6. [Benchmark methodology](#6-benchmark-methodology)
7. [Tech stack and cost](#7-tech-stack-and-cost)
8. [Repository layout (planned)](#8-repository-layout-planned)
9. [Setup and running for free](#9-setup-and-running-for-free)
10. [Configuration (planned)](#10-configuration-planned)
11. [Open decisions](#11-open-decisions)
12. [Known constraints and risks](#12-known-constraints-and-risks)
13. [Build order](#13-build-order)

---

## 1. What this project is

The gateway is a serving layer that sits between users and a large language model running on a single GPU.

**The problem.** If a server handles one request at a time, the GPU sits mostly idle. Generating one token for one user reads all the model weights from GPU memory but does very little arithmetic with them. Meanwhile every other user waits in line, so latency grows with queue length.

**What the gateway does:**

| Function | Description |
|---|---|
| Continuous batching | Groups active requests into one GPU forward pass per step. Requests join and leave the batch at token boundaries instead of waiting for a whole batch to finish. |
| Token streaming | Sends each token to its client as soon as it is generated, using Server-Sent Events (SSE). |
| Admission control | Limits how much work is on the GPU at once, based on free KV-cache memory and queue length. Excess requests wait or get rejected with HTTP 429, so bursts can't trigger out-of-memory (OOM) crashes. |
| Benchmarking | Load-tests the gateway against a naive one-at-a-time server and against vLLM, using the same workload and the same client. |

**Scope.** This is an engine of its own: the scheduler, KV-cache manager and model forward pass are written in this project in PyTorch. The model weights come from Hugging Face. vLLM is used only as a comparison point.

**Out of scope:** multi-GPU or multi-node serving, quantization, LoRA serving, authentication, and production deployment.

---

## 2. Goals and how they are measured

These are targets, not results. Each one is measured by the benchmark harness described in [§6](#6-benchmark-methodology).

| Target | Metric definition | Compared against |
|---|---|---|
| About 6× higher throughput at 32 concurrent users | Output tokens per second, summed over all requests during the measurement window | Naive baseline |
| About 80% lower p99 latency at 32 concurrent users | 99th percentile of end-to-end request latency (request sent → last token received) | Naive baseline |
| Time to first token (TTFT) under 200 ms | Time from request sent to first streamed token, reported at p50 and p99 | Naive baseline (seconds under load) |
| Zero OOM crashes under 2× burst | Count of CUDA OOM errors and server crashes while load runs at 2× the steady-state level ([§6.3](#63-burst-test)) | Naive baseline with admission control off |
| Competitive with vLLM | Same metrics, reported side by side. No target ratio; vLLM is the reference. | vLLM |

See [§12](#12-known-constraints-and-risks) for when the TTFT target is and isn't achievable on a T4.

---

## 3. Architecture

```
 clients (load generator / curl)
        │  HTTP POST /v1/completions  (stream=true → SSE)
        ▼
┌─────────────────────────── API process ───────────────────────────┐
│  FastAPI + uvicorn (asyncio event loop)                            │
│    ├─ request validation (pydantic)                                │
│    ├─ admission check #1: queue length  ──► 429 if full            │
│    └─ per-request asyncio.Queue  ◄── tokens pushed by engine       │
│                         │ submit                                    │
│                         ▼                                           │
│  Engine thread (owns the GPU)                                       │
│    loop:                                                            │
│      1. Scheduler picks the batch for this step                     │
│           - admission check #2: free KV blocks                      │
│           - preempt if decode would run out of blocks               │
│      2. Model runner: one forward pass (prefill and/or decode)      │
│      3. Sampler: next token per sequence                            │
│      4. Push tokens to request queues; free finished sequences      │
│                                                                     │
│  KV-cache manager: fixed pool of GPU memory, allocated per sequence │
│  Metrics: Prometheus counters/histograms at GET /metrics            │
└─────────────────────────────────────────────────────────────────────┘
```

**Why a separate engine thread.** GPU calls block. If they ran on the asyncio event loop, the server couldn't accept connections or flush streamed tokens during a forward pass. The engine runs in one dedicated thread and hands tokens back to the event loop with `loop.call_soon_threadsafe`.

**Why one process.** Splitting the API and engine into separate processes, as vLLM does, takes CPU work such as tokenization and HTTP handling off the engine's critical path. It also adds inter-process communication. This project starts with one process and measures CPU overhead before deciding to split ([§11.8](#118-process-model)).

---

## 4. Components

### 4.1 API server

- **Framework:** FastAPI on uvicorn.
- **Endpoints:**
  - `POST /v1/completions`: an OpenAI-compatible subset. The fields are `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `stream` and `stop`. With `stream: true`, the response is SSE `data:` chunks ending in `data: [DONE]`.
  - `GET /health`: liveness check.
  - `GET /metrics`: Prometheus text format.
- **Why OpenAI-compatible:** the gateway, the naive baseline and vLLM then all accept the same requests. One load generator and one dataset can test all three, which makes the comparison fair.
- **Client disconnects:** when a client drops its connection, the request is aborted and its KV blocks are freed.

### 4.2 Admission controller

Admission happens at two points:

1. **At the door (API).** If the waiting queue holds `max_queue_len` requests, the new request gets `429 Too Many Requests` with a `Retry-After` header. A request that waits longer than `queue_timeout_s` gets `503`.
2. **At the GPU (scheduler).** A waiting request moves to running only if the KV-cache manager has enough free blocks for it ([§11.4](#114-kv-memory-reservation-policy)).

Together these cap GPU memory use at a fixed budget no matter how many requests arrive. That cap is the mechanism behind the zero-OOM target.

### 4.3 Scheduler

The scheduler runs once per engine step and uses iteration-level ("continuous") batching, the technique introduced by the Orca paper and used by vLLM. Each step it:

1. Removes finished sequences and frees their KV blocks.
2. If the running sequences need more blocks than are free, preempts the most recently admitted sequence. That sequence's KV is discarded and it goes back to the front of the waiting queue to be recomputed later.
3. Admits waiting requests in FIFO order while all of these hold: `running < max_num_seqs`, tokens this step `≤ max_num_batched_tokens`, and enough free blocks.
4. Builds the step's batch from new prompts (prefill) and running sequences (one decode token each).

How prefill and decode share a step is an open decision ([§11.3](#113-prefill--decode-scheduling-policy)).

### 4.4 KV-cache manager

Each generated token depends on cached attention keys and values (the "KV cache") for every earlier token. The manager:

- Preallocates one fixed pool of GPU memory at startup, sized by `kv_cache_memory_gb` or by a number of blocks.
- Divides the pool into blocks of `block_size` tokens and keeps a free list.
- Gives each sequence a block table that maps its token positions to physical blocks.
- Reports how many blocks are in use and free, for admission control and metrics.

How the attention kernel reads from this memory is an open decision ([§11.2](#112-kv-cache-layout)).

Memory per token, fp16, `2 (K,V) × layers × kv_heads × head_dim × 2 bytes`:

| Model | Layers | KV heads | Head dim | KV bytes/token | Weights (fp16) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 2 | 64 | 12 KB | ~1.0 GB |
| Qwen2.5-1.5B-Instruct | 28 | 2 | 128 | 28 KB | ~3.1 GB |
| TinyLlama-1.1B-Chat | 22 | 4 | 64 | 22 KB | ~2.2 GB |

### 4.5 Model runner

- A custom PyTorch forward pass for the Qwen2/Llama architecture: RMSNorm, rotary position embeddings (RoPE), grouped-query attention and a SwiGLU MLP. Qwen2 also adds biases to its Q, K and V projections.
- Weights load from the Hugging Face `safetensors` files.
- **Why custom instead of `transformers` `model.generate`:** `generate` assumes a fixed batch that starts and ends together. Continuous batching needs sequences of different lengths, at different stages, to share one forward pass and read from a KV cache this project controls.
- **Correctness check:** a test compares the custom forward's logits with the Hugging Face reference model on the same input, within a set fp16 tolerance.
- **Attention:** PyTorch `scaled_dot_product_attention` (SDPA). FlashAttention-2 does not support the T4 (Turing architecture), so it is not used.
- **Precision:** fp16. The T4 has no native bf16 support.

### 4.6 Sampler

Per-sequence `temperature` and `top_p` are applied to the whole batch in one vectorized operation, and `temperature = 0` means greedy decoding. A sequence stops on `max_tokens`, an end-of-sequence token, or a `stop` string.

### 4.7 Streaming and detokenization

Each request has an `asyncio.Queue`. The engine pushes new token IDs into it. The API side decodes them incrementally into text and writes SSE chunks. Decoding is incremental because one token isn't always a complete character: a multi-byte character can be split across tokens. Text is emitted only once it decodes cleanly.

### 4.8 Metrics

Exported through `prometheus_client` at `/metrics` and also logged per request for the benchmark:

- TTFT, inter-token latency (ITL), end-to-end latency (histograms)
- Throughput: output tokens/s and requests/s
- Queue depth, running sequences, KV blocks used/free
- Rejections (429/503), preemptions, OOM count
- GPU memory allocated, from `torch.cuda.memory_allocated`

---

## 5. Baselines

| Baseline | What it is | Why it is included |
|---|---|---|
| **Naive** | The same FastAPI interface, serving one request at a time with Hugging Face `model.generate` and `TextIteratorStreamer` behind a lock. | The "before" picture. All throughput and latency targets are measured against it. |
| **Static batching** (optional) | Collects up to N requests or waits T ms, then runs one padded `generate` call. No request joins or leaves mid-batch. | Shows why *continuous* batching matters, not just batching. Short requests wait for the longest one in their batch. |
| **vLLM** | `vllm serve <model>` with the same model, dtype, and max context. | The industry-standard reference. |

All three serve the same model in fp16 on the same GPU and are tested with the same client and workload.

---

## 6. Benchmark methodology

### 6.1 Load generator

The load generator is an async Python client (`asyncio` + `httpx`). It opens N concurrent streaming requests and records per-request TTFT, ITL, end-to-end latency and token counts from the SSE stream. The options are listed in [§11.6](#116-load-generator).

### 6.2 Steady-state test

- Concurrency levels: 1, 4, 8, 16, 32, with 32 as the headline number.
- Each level runs for a fixed number of requests after a warm-up, so CUDA kernel compilation and memory allocation aren't counted.
- Outputs per level: throughput, TTFT p50/p99, end-to-end p50/p99, ITL p50/p99.

### 6.3 Burst test

- Load runs at 32 concurrent users and steps up to 64 (2×) for a fixed window, then drops back.
- Pass criteria: no OOM, no server crash, every request either completes or gets an explicit 429/503, and throughput recovers after the burst.
- Repeated with admission control turned off to show the failure mode it prevents. The KV budget may need to be capped to make the failure visible ([§12](#12-known-constraints-and-risks)).

### 6.4 Workload

The dataset choice is open ([§11.5](#115-benchmark-workload)). The same prompts, the same seed and the same `max_tokens` are used for every system.

### 6.5 Reporting

Results are saved as JSON per run (system, model, GPU, config, git commit, per-request records). A report script uses `matplotlib` to plot throughput vs concurrency, latency percentiles vs concurrency, and a burst timeline.

---

## 7. Tech stack and cost

| Layer | Choice | License | Cost |
|---|---|---|---|
| Language | Python 3.10+ | PSF | $0 |
| Tensor / GPU | PyTorch (CUDA) | BSD-3 | $0 |
| Weights, tokenizer | `transformers`, `safetensors`, `tokenizers` (Hugging Face Hub) | Apache-2.0 | $0 |
| API | FastAPI, uvicorn, pydantic | MIT / BSD | $0 |
| HTTP client | httpx | BSD-3 | $0 |
| Metrics | prometheus_client | Apache-2.0 | $0 |
| Plots | matplotlib | PSF-based | $0 |
| Tests | pytest | MIT | $0 |
| Reference engine | vLLM | Apache-2.0 | $0 |
| Model | Qwen2.5-0.5B/1.5B-Instruct (Apache-2.0) or TinyLlama-1.1B (Apache-2.0) | see [§11.1](#111-model) | $0 |
| GPU | Google Colab free tier (NVIDIA T4, 16 GB) | — | $0 |
| CI | GitHub Actions, CPU-only tests (free for public repos) | — | $0 |

**About Colab Pro.** You have Colab Pro, but it is a paid subscription, so the project is designed to run fully on the free T4. Pro's faster GPUs (L4, A100) are an optional extra and spend monthly compute units. If you use them, report the GPU next to each result, because numbers from different GPUs aren't comparable. Free alternative: **Kaggle Notebooks** give about 30 GPU-hours a week (T4 ×2 or P100) with 12-hour sessions.

---

## 8. Repository layout (planned)

```
LLM Inference Gateway/
├── gateway/
│   ├── api/
│   │   ├── server.py        # FastAPI app, routes, SSE streaming
│   │   └── protocol.py      # pydantic request/response schemas
│   ├── engine/
│   │   ├── engine.py        # engine thread and step loop
│   │   ├── scheduler.py     # continuous batching, preemption
│   │   ├── admission.py     # queue limits, timeouts
│   │   ├── kv_cache.py      # block pool, block tables
│   │   └── sequence.py      # per-request state
│   ├── model/
│   │   ├── qwen2.py         # custom forward pass
│   │   ├── loader.py        # safetensors → custom model
│   │   └── sampler.py
│   ├── metrics.py
│   └── config.py
├── baselines/
│   ├── naive_server.py
│   └── static_batch_server.py   # optional
├── bench/
│   ├── load_test.py
│   ├── workloads.py
│   └── report.py
├── notebooks/
│   └── colab_benchmark.ipynb    # end-to-end run on Colab
├── tests/
├── docs/
│   └── PROJECT.md
└── pyproject.toml
```

---

## 9. Setup and running for free

> These are the intended steps. The commands will work once the code in §8 exists.

### 9.1 Local development (no GPU)

For CPU-only unit tests: scheduler logic, the KV block allocator, admission control, and the protocol.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests use a tiny model on CPU, so no GPU is needed. On an Apple Silicon Mac, PyTorch's `mps` backend can run the custom model for smoke tests. vLLM cannot run there because it needs an NVIDIA GPU.

### 9.2 GPU runs on Google Colab (free T4)

1. Open `notebooks/colab_benchmark.ipynb` in Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the setup cell. It clones the repo and runs `pip install -e .` (plus `pip install vllm` for the vLLM comparison).
4. Start the server in the background from a cell:
   ```bash
   nohup python -m gateway.api.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 > server.log 2>&1 &
   ```
5. Run the benchmark against `http://localhost:8000`:
   ```bash
   python -m bench.load_test --url http://localhost:8000 --concurrency 1,4,8,16,32 --out results/gateway.json
   ```
6. Repeat steps 4–5 for `baselines/naive_server.py` and for `vllm serve`, stopping each server before starting the next. Only one should hold the GPU at a time.
7. Run `python -m bench.report results/` and download `results/` and the plots. Colab deletes local files when the session ends.

The server and load generator run on the same Colab VM and talk over `localhost`. That keeps network latency out of the measurements, and no tunnel or public endpoint is needed.

### 9.3 Quick manual test

```bash
curl -N http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"default","prompt":"Explain KV caching in one sentence.","max_tokens":64,"stream":true}'
```

---

## 10. Configuration (planned)

| Setting | Meaning | Affects |
|---|---|---|
| `model` | Hugging Face model ID | Memory, speed |
| `max_model_len` | Max prompt + output tokens per request | KV per request |
| `block_size` | Tokens per KV block | Fragmentation vs. bookkeeping |
| `kv_cache_memory_gb` / `num_kv_blocks` | Size of the KV pool | Max concurrent tokens; OOM safety |
| `max_num_seqs` | Max sequences running at once | Batch size, ITL |
| `max_num_batched_tokens` | Max tokens processed in one step | Step time, TTFT vs. ITL |
| `max_queue_len` | Waiting requests before 429 | Burst behavior |
| `queue_timeout_s` | Max wait before 503 | Tail latency under overload |

---

## 11. Open decisions

Every option below is free. Each one is your call.

### 11.1 Model

| Option | Pros | Cons |
|---|---|---|
| Qwen2.5-0.5B-Instruct | Fastest; smallest KV (12 KB/token); easiest to hit the TTFT target | Lower output quality; results look less impressive |
| Qwen2.5-1.5B-Instruct | Realistic size; still fits a T4 easily | Prefill is about 3× slower than 0.5B; TTFT target harder |
| TinyLlama-1.1B-Chat | Plain Llama architecture (no QKV bias) | Weaker model; 2k context |

Qwen2.5-0.5B and 1.5B share one architecture, so the custom forward pass supports both. Llama-3.2-1B is also free but gated: it requires accepting Meta's license on Hugging Face.

### 11.2 KV-cache layout

| Option | How it works | Pros | Cons |
|---|---|---|---|
| A. Slot-based contiguous | Preallocated `[max_num_seqs, max_model_len]` cache; each sequence gets one slot | Simplest; SDPA works directly | Reserves `max_model_len` for every request, so short requests waste memory and fewer fit |
| B. Paged + gather | Block pool; before attention, gather a sequence's blocks into a contiguous tensor, then SDPA | Memory efficient; no custom kernels | Gather copies memory every step, which costs time |
| C. Paged + Triton kernel | Custom attention kernel reads blocks in place, like vLLM's PagedAttention | Memory efficient and fast | Hardest to write and debug; Triton on T4 is supported but less tuned |

A natural path is A first, to get a working baseline, then B, then C only if profiling shows the gather is the bottleneck.

### 11.3 Prefill / decode scheduling policy

| Option | Pros | Cons |
|---|---|---|
| Prefill-first | Lowest TTFT for new requests | Running requests pause during long prefills (ITL spikes) |
| Decode-first | Smooth streaming for running requests | New requests wait longer (higher TTFT) |
| Chunked prefill | Splits long prompts across steps alongside decodes; balances TTFT and ITL | More complex scheduler and attention masking |

### 11.4 KV memory reservation policy

| Option | Pros | Cons |
|---|---|---|
| Reserve `prompt + max_tokens` at admission | Never needs preemption; simple; OOM impossible by construction | Over-reserves (most requests stop early), so fewer run at once and throughput is lower |
| Reserve on demand + preemption | Higher memory use, so higher throughput | Must implement preemption and recompute; preempted requests see latency spikes |

### 11.5 Benchmark workload

| Option | Pros | Cons |
|---|---|---|
| Fixed synthetic lengths (e.g., 512 in / 128 out) | Reproducible; easy to reason about | Unrealistic; hides the gains from requests of different lengths |
| ShareGPT conversations (free on Hugging Face; used by vLLM's own benchmarks) | Realistic length distribution; directly comparable to published numbers | Needs filtering and truncation; results vary more |
| Both | Clean numbers plus realistic numbers | More runs on a time-limited GPU |

### 11.6 Load generator

| Option | Pros | Cons |
|---|---|---|
| Custom asyncio + httpx script | Exact TTFT/ITL measurement; full control over burst shape | Code to write and maintain |
| vLLM's built-in serving benchmark (`vllm bench serve`; older versions: `benchmarks/benchmark_serving.py`) | Standard tool; reports TTFT/ITL/throughput; works against any OpenAI-compatible server | Less control over custom burst patterns; interface changes between vLLM versions |
| Locust | Mature; web UI | Built for request/response, so streaming TTFT is awkward to measure |

### 11.7 Streaming transport

| Option | Pros | Cons |
|---|---|---|
| SSE | OpenAI-compatible; works with curl and benchmark tools; one-way fits token streaming | Text-only; one request per connection |
| WebSocket | Bidirectional; can cancel or send multiple messages on one connection | Not OpenAI-compatible, so the shared benchmark tools can't use it |

### 11.8 Process model

| Option | Pros | Cons |
|---|---|---|
| Single process, engine thread | Simplest; no inter-process communication | Python's GIL can make HTTP, tokenization and the engine loop compete for CPU |
| API and engine in separate processes (ZeroMQ or `multiprocessing` queues) | Engine loop isolated from HTTP overhead | More code; serialization cost; harder to debug |

### 11.9 GPU platform

| Option | Pros | Cons |
|---|---|---|
| Colab free (T4) | Zero cost; simple | Session limits; GPU not guaranteed; 2 vCPUs |
| Kaggle (T4 ×2 / P100) | Zero cost; about 30 GPU-h/week; 12h sessions | Weekly quota; notebook-only workflow |
| Colab Pro (L4 / A100) | Faster; newer GPUs (FlashAttention-capable) | Paid (already owned); compute units run out; results not comparable to T4 runs |

---

## 12. Known constraints and risks

- **TTFT under 200 ms depends on the setup.** If 32 requests arrive at the same instant, the last one waits for all 32 prompts to be prefilled. Rough estimate: 32 prompts of 512 tokens on Qwen2.5-1.5B on a T4 take more than a second in total, however they're batched. The target is realistic with Poisson (spread-out) arrivals, shorter prompts, or the 0.5B model. Report the arrival pattern alongside every TTFT number.
- **A small model may not OOM even without admission control.** A T4 has 16 GB, and Qwen2.5-0.5B leaves about 14 GB for KV, which is over a million tokens. To show the zero-OOM mechanism working, the burst test may need a capped KV budget (`num_kv_blocks`) or long contexts. State this in the results.
- **Colab has only 2 vCPUs.** The load generator, HTTP server, tokenizer and engine loop all share them. Monitor CPU use during benchmarks. If the CPU is saturated, the numbers measure CPU limits rather than the scheduler.
- **vLLM on the T4.** Recent vLLM releases have changed their support for older GPUs. Check that the installed version runs on compute capability 7.5 (the T4). If it doesn't, pin an older version or run the vLLM comparison on Kaggle or an L4, and note which.
- **Colab sessions are ephemeral.** Save results to Google Drive or download them after each run, and pin package versions so runs can be reproduced.
- **The 6× and 80% targets are relative to the naive baseline.** The size of the gain depends on how the baseline is built. Document it exactly ([§5](#5-baselines)) so the comparison can be defended.

---

## 13. Build order

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | Naive baseline server + load generator | Baseline numbers recorded at 1–32 concurrency |
| 2 | Custom model forward + loader | Logits match the Hugging Face reference within tolerance |
| 3 | Engine loop + continuous batching (slot-based KV) + SSE streaming | Gateway serves concurrent streaming requests correctly |
| 4 | Admission control + metrics | Burst test passes; `/metrics` populated |
| 5 | Paged KV cache, preemption, chunked prefill (per decisions in §11) | Throughput and latency improve with no correctness regressions |
| 6 | vLLM comparison + report | Final plots and tables for all three systems |
