# LLM Inference Gateway

A scheduling layer that sits in front of a large language model and serves many users at once on a single GPU. It uses continuous batching, token streaming and memory-aware admission control.

> **Status:** early development. The design is documented in [docs/PROJECT.md](docs/PROJECT.md). The code is being built in phases.

## Why

Serving an LLM one request at a time leaves the GPU mostly idle and makes users wait in long queues. This project builds the serving engine that keeps the GPU busy and keeps users from waiting in line.

## What it does

- **Continuous batching:** groups active requests into one GPU forward pass per step. Requests join and leave the batch at token boundaries.
- **Token streaming:** sends tokens to clients as they are generated, over Server-Sent Events (OpenAI-compatible `/v1/completions`).
- **Admission control:** limits how much work enters the GPU at once, based on free KV-cache memory and queue length, so traffic bursts can't cause out-of-memory crashes.
- **Benchmarking:** load-tests the gateway against a naive one-request-at-a-time server and against [vLLM](https://github.com/vllm-project/vllm), with the same model, workload and client.

## How it's built

- The engine is written in this project: the scheduler, the KV-cache manager and the model forward pass (PyTorch). Model weights load from Hugging Face.
- The API is FastAPI + uvicorn. A dedicated engine thread owns the GPU.
- Metrics are exported in Prometheus format at `/metrics`.
- Everything runs at zero cost: open-source libraries, open-weight models (Qwen2.5 / TinyLlama), and the free Google Colab T4 GPU.

See [docs/PROJECT.md](docs/PROJECT.md) for the architecture, component details, benchmark methodology and open design decisions.

## Getting started

Setup instructions will be added once the first runnable version lands. The planned workflow (CPU unit tests locally, GPU benchmarks on Colab) is in [docs/PROJECT.md §9](docs/PROJECT.md#9-setup-and-running-for-free).

## Roadmap

1. Naive baseline server + load generator
2. Custom model forward pass, verified against Hugging Face
3. Engine loop with continuous batching and SSE streaming
4. Admission control and metrics
5. Paged KV cache, preemption, chunked prefill
6. vLLM comparison and benchmark report
