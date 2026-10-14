# What this project is and why it matters

## The short version

I built the piece of software that sits between users and a language model and decides who gets the GPU, when, and in what groups. It takes a stream of chat-style requests and packs them into small batches. It streams each user's words back as they're produced, and it refuses work it can't safely fit instead of crashing. The whole thing runs on a free Google Colab T4, and I measure it against a one-at-a-time server and against vLLM, the open-source engine most companies use in production.

## What it does, in plain terms

A language model writes one token (roughly one word piece) at a time. Each token needs a full pass through the model. If a server handles one user at a time, everyone else waits in line, and the GPU sits mostly idle while it waits on memory.

The gateway changes three things:

1. **It batches.** Instead of running the model once per user per token, it runs the model once for a group of users per token. On a GPU that's nearly free: a step for 32 users costs only a little more than a step for one.
2. **It streams.** Each user sees their answer appear word by word, starting a fraction of a second after they hit enter, instead of waiting for the whole answer.
3. **It protects itself.** Before letting a group onto the GPU, it checks that the group's memory will fit. If too many people show up at once, extra requests get a polite "try again in a second" (HTTP 429) instead of taking the server down for everyone.

## The real problem it solves

Serving a model costs money, and the cost is GPU time. Most of that time is spent waiting on memory, not doing math. A single user's request reads all 3 GB of model weights from memory to produce one token. Batching lets 32 users share that one read. That's the difference between needing ten GPUs and needing one.

The industry answer is a serving engine like vLLM, TGI, or TensorRT-LLM. They're large, complex projects. This project rebuilds the core ideas from first principles, at a scale I can reason about end to end:

- How requests get grouped (micro-batching and length bucketing)
- How memory gets budgeted (KV-cache accounting and admission control)
- How tokens flow back to many clients at once (asyncio, threads, Server-Sent Events)
- How you prove any of it worked (closed-loop load tests, burst tests, profiling)

## Why it isn't trivial

- **The GPU and the web server live in different worlds.** The model runs in a blocking loop on a GPU. The web server is an asyncio event loop that must never block. Getting tokens from one to the other safely, for dozens of users at once, takes careful hand-off between threads.
- **Batching variable-length text means padding.** Prompts and answers have different lengths. To run them together, shorter ones get padded, and a finished answer keeps occupying its row until the whole batch is done. That waste is invisible unless you measure it. I measured it with PyTorch Profiler and then fixed it with length bucketing.
- **Memory is the real limit, not compute.** Each request's attention cache (the KV cache) grows with every token. Admit too much and the GPU runs out of memory mid-generation, which kills every request in flight. The gateway reserves each batch's worst-case padded footprint up front, so a burst can't push it over.
- **Benchmarks lie easily.** Throughput, time-to-first-token and tail latency trade off against each other. A fair comparison needs the same model, the same prompts, the same output lengths, the same client, and the same GPU for every system. The harness enforces all of that.

## What makes it different from a tutorial build

- **It's measured, not just built.** Every claim comes from a reproducible run: a closed-loop sweep at 1/8/16/32 users, a burst test that doubles the load mid-run, and a profiler pass. The baseline is the *same server* with batching turned off, so the comparison isolates exactly one variable.
- **It's compared against the real thing.** vLLM runs on the same T4 with the same workload. I report the gap honestly and can explain where it comes from.
- **It handles failure on purpose.** Out-of-memory errors are caught and contained to one batch. Disconnected clients free their slot. Requests that wait too long get a 503 rather than hanging forever. There's a preset with admission control turned off, specifically to show what breaks without it.
- **It runs for free.** Open model (Qwen2.5-1.5B, Apache 2.0), open libraries, free Colab GPU, free GitHub Actions CI. Anyone can clone it and reproduce the numbers.

## Who would care

- Anyone who has to serve a model to more than one person and wants to understand what vLLM is actually doing for them
- Interviewers for ML infrastructure, backend, or systems roles: the project covers concurrency, GPU memory, scheduling, API design and performance measurement in one place
- Anyone choosing between "just call `model.generate` in a Flask app" and a real serving engine, who wants numbers on what that choice costs
