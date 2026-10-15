# Explaining and Defending the Project

For demos, class presentations and interviews. All numbers come from a Google Colab T4 run with Qwen2.5-1.5B (see the README's Results section). Each is a single run.

---

## 1. The 30-second version

I built a serving layer for a language model on a single free T4 GPU. Serving one request at a time leaves the GPU mostly idle and makes everyone queue. My gateway batches requests together, lets new requests join a batch that's already running, and streams tokens back as they're generated. It also reserves memory for each batch up front, so traffic bursts get a polite retry instead of crashing the server. At 32 concurrent users it gets **11.9×** the throughput of a one-at-a-time baseline, cuts time to first token from **104 seconds** to **158 milliseconds**, and reaches **39%** of vLLM's throughput on the same GPU.

## 2. The 2-minute version

Language models write one token at a time, and each token needs a full pass through the model. On a GPU that pass is limited by memory bandwidth, not math. Generating one token for one user reads all 3 GB of weights to do a tiny amount of work. If you serve users one at a time, you pay that 3 GB read per user per token, and everyone else waits.

My gateway sits in front of Qwen2.5-1.5B and fixes that in three ways.

**First, batching.** Requests are prefilled in small groups and merged into a running batch at the next step boundary, and finished requests are evicted immediately. One weight read serves the whole batch, and nobody waits for a stranger's long answer to finish.

**Second, admission control.** Each batch reserves its worst-case KV-cache footprint against a budget computed from free GPU memory. Prefill size is capped, and if the queue is full the server returns 429 with Retry-After. Memory can't be oversubscribed, so OOMs can't happen.

**Third, padding.** Batching variable-length text means padding. I profiled it and found padding was about **a third** of decode time (33% in my profiler run), made worse because Hugging Face's attention expands the KV cache six times when a padding mask is present. So I added length lanes: the running set can split into short and long groups. A cost model the engine calibrates at startup decides when a split saves more than the extra forward pass costs.

I measured everything with a closed-loop load tester at 1 to 32 users, a burst test from 32 to 64 users, and PyTorch Profiler. The baseline is the same server with batch size 1, so batching is the only thing that changes. vLLM is the reference, and I can explain where the remaining gap comes from: paged attention, CUDA graphs, and no padding at all.

---

## 3. Whiteboard narration

Draw left to right while talking.

1. "On the left are clients, sending OpenAI-style completion requests over HTTP."
2. "They hit a FastAPI server running on an asyncio event loop. The handler tokenizes the prompt, creates a Sequence object with its own asyncio queue, and submits it. Submit is where backpressure lives: too big is a 400, queue full is a 429 with Retry-After."
3. "Accepted requests go into a waiting list, which is the only shared state, protected by a lock."
4. Draw a box: "The engine thread. It's the only thing that touches the GPU. It runs a loop, and each step has four parts."
   - "Expire anything that waited too long. That's a 503."
   - "Admit. Take the oldest waiting request, plus others headed for the same lane. Check three limits: free rows, a cap on padded prefill tokens, and the token budget. Prefill them together, then merge them into the running lane."
   - "Decode. One forward pass per lane, one token per row. Sample, push each token to its request's queue, and evict anything that finished."
   - "Every eight steps, rebalance. Split a lane if the padding saved is worth more than the cost of an extra forward pass, or merge two lanes if combining them is nearly free."
5. Draw an arrow back to the left: "Tokens cross threads with call_soon_threadsafe. The HTTP handler detokenizes incrementally, checks stop strings, and writes SSE chunks."
6. Underneath: "Memory. Every lane reserves rows × max(prompt + max_tokens) KV slots. The KV tensor is rows × width, and width can never exceed that, so real memory stays below the reservation. The budget is sized from free memory after the model loads, minus headroom for the largest prefill."
7. To the side: "Measurement. A closed-loop load tester, a burst test, and a profiler that runs the real scheduler in-process."

---

## 4. Design decisions and the alternatives I rejected

### Batching: join at step boundaries, not run-to-completion micro-batches
- **Rejected:** collect requests, run them as a fixed batch until all finish.
- **Why:** a batch is only done when its longest request is done. Short requests hold their row as dead weight, and new arrivals queue behind the whole batch. I built that version first. On my local test it barely beat the naive baseline (about 53 vs 39 tok/s at 32 users) with huge tail latency. Letting requests join and leave at every step fixed both.

### KV cache: HF `DynamicCache`, left-padded, per lane
- **Rejected:** a paged KV cache with a custom attention kernel (vLLM's PagedAttention).
- **Why:** paged attention needs custom CUDA or Triton kernels and a block allocator. That's a project on its own, and hard to verify on a free T4. Left padding works with stock PyTorch SDPA and keeps every line reviewable. The cost is padding, which I measure and manage with lanes. Paged attention is the first item for the next version.

### Length bucketing: cost-model-driven lanes, not fixed buckets
- **Rejected 1: fixed length buckets, each with its own batch.** I built this. It cut padded tokens by about 60% on my Mac but *lowered* throughput, because every bucket costs a full forward pass per step.
- **Rejected 2: no bucketing.** That leaves a large share of decode time on padding.
- **Why the cost model:** the right answer depends on the hardware. A forward pass costs `fixed + per_slot × slots`. Splitting a lane adds one `fixed` and saves `saved × per_slot`. I measure both at startup and split only when the saving beats the cost by 1.5×, merging back below half the cost so it doesn't flap.

### Memory: reserve the worst case up front
- **Rejected:** reserve as you go, and preempt (evict and recompute) when memory runs out.
- **Why:** up-front reservation is safe by construction. A lane can never exceed `rows × max need`, so there's no preemption code and no recompute storms. The cost is over-reservation, since requests often stop before `max_tokens`. With a 1.5B model on a 16 GB T4 the budget is around 300k tokens, far above what 32–64 users need, so over-reserving doesn't limit admission here.

### Admission: three separate limits
- A queue cap (backpressure), a token budget (KV memory), and a prefill cap (activation memory).
- **Why three:** they protect different things. The budget alone doesn't stop one huge prefill from spiking activation memory. The prefill cap alone doesn't stop the KV cache from growing. The queue cap stops waits from growing unbounded and turns overload into fast, explicit 429s.

### Concurrency: one process, asyncio plus one engine thread
- **Rejected:** separate API and engine processes linked by ZeroMQ, as vLLM does.
- **Why:** no serialization or IPC, and the code is simpler to reason about. The GIL is the tradeoff. Tokenization, detokenization and the engine loop's Python share a core. On Colab's 2 vCPUs I'd watch CPU usage, and splitting processes is the fix if it saturates.

### Baseline: the same server with batch size 1
- **Rejected:** a separate Flask + `model.generate` server.
- **Why:** a different server changes the tokenizer path, sampling, streaming and HTTP stack. Using the same code with `max_batch_size=1` isolates batching. It's the fairest baseline, and it's easy to defend.

### Streaming: SSE, OpenAI-compatible
- **Rejected:** WebSockets.
- **Why:** SSE is one-directional, which is all token streaming needs. It works with curl, and it's what OpenAI and vLLM speak, so one load tester measures both systems identically.

### Attention: SDPA, fp16
- **Rejected:** FlashAttention-2, bf16.
- **Why:** the T4 is Turing (compute capability 7.5). FA2 needs Ampere (8.0+), and so does native bf16.

### Waiting for the first token before opening the stream
- **Why:** once a streaming response starts, the 200 status is already sent. Waiting for the first engine message lets queue timeouts and OOMs come back as real 503s, at no latency cost, because the first chunk can't be sent before the first token exists anyway.

---

## 5. Hard Q&A bank

**Q1. Why is batching nearly free on a GPU?**
Decode is memory-bandwidth-bound. Each step reads every weight once, about 3 GB for 1.5B parameters in fp16, and does about two FLOPs per weight per row. A T4 does about 65 TFLOPS but moves about 300 GB/s, so at batch 1 the compute units are mostly idle. Adding rows reuses the same weight read, and time per step grows slowly until KV-cache traffic or compute starts to dominate.

**Q2. What's the difference between your batching and vLLM's continuous batching?**
Scheduling is the same idea: requests join and leave at step boundaries. The difference is memory layout. vLLM stores KV in fixed-size pages and uses a kernel that reads them in place, so there's no padding and no copying when rows join or leave. I use a contiguous left-padded tensor, so joining pads and copies, evicting copies, and short rows carry padding. That, plus CUDA graphs and C++/Triton kernels, is most of the gap.

**Q3. Where exactly does your remaining gap to vLLM come from?**
- **Padding:** attention over padded KV positions, amplified by HF's GQA expansion under a mask.
- **Cache growth:** `DynamicCache` uses `torch.cat` every step, which copies the whole cache.
- **Python overhead:** HF's forward runs Python per layer per step, where vLLM replays CUDA graphs.
- **Join and evict copies.**

The profiler's op table shows these directly: the attention and copy ops, and time in `item`/sync.

**Q4. How do you know batched output is correct?**
Tests compare token-for-token: a batch of different-length prompts against each prompt alone, requests that join a running batch midway against running alone, and split lanes against a single batch. They pass with greedy decoding on a random model in fp32. If padding, positions, merge, evict or split were wrong, tokens would differ.

**Q5. Why left padding and not right padding?**
With left padding, every row's newest token is in the last column. Decode appends one column and all rows write to the same place. With right padding, each row's next position would be a different column. You'd need per-row scatter into the cache, and the mask would have holes in the middle.

**Q6. Doesn't left padding break positional embeddings?**
It would if positions came from column index. I pass `position_ids = cumsum(mask) − 1`, so each row's real tokens get 0, 1, 2… regardless of padding, and decode uses each row's own `next_pos`. The equivalence tests confirm it.

**Q7. Prove admission control prevents OOM.**
A lane's KV memory is `rows × width × bytes_per_token`. Width is the longest live row, and each row's length is at most its `prompt + max_tokens`, so width ≤ the lane's max need. The reservation is `rows × max need`, so memory ≤ reservation. The sum of reservations ≤ budget, and the budget is free memory after weights, times 0.85, minus prefill headroom. Prefill activations are bounded by the prefill-token cap. The assumptions are the activation estimate and allocator fragmentation, which the 0.85 margin and `expandable_segments` cover. The burst test checks it empirically: in a 32 → 128 user burst with prompts up to 3,072 tokens, admission control kept peak memory at 7.0 GB of 15 with zero OOMs, while the server without it peaked at 14.1 GB, about 1 GB from crashing.

**Q8. What happens when the budget is exhausted?**
The oldest waiting request stays at the front. It isn't skipped, so it can't starve. Smaller requests behind it can still be admitted if they fit. As rows finish, `_resize` releases budget immediately. If the queue reaches `max_queue`, new requests get 429. If a request waits longer than `queue_timeout_s`, it gets 503.

**Q9. Isn't returning 429 just hiding latency by dropping users?**
It's load shedding, and the report shows it: every burst table has status counts next to latency. The alternative is accepting everything and letting the queue grow until requests time out, which is worse for everyone, including the users already inside. With 429 and `Retry-After`, clients get a fast, honest answer and can back off. In the steady-state test at 32 users, the queue cap of 48 is never reached, so those latency numbers include every request.

**Q10. Your TTFT number: is it measured fairly?**
It's measured client-side, from sending the request to receiving the first chunk with content, over localhost, including queueing. That's the same for every system. Under a closed loop at 32 users, the naive baseline's TTFT is dominated by waiting behind other requests. The gateway's is roughly one join wait plus one prefill plus one decode round.

**Q11. Why closed-loop load and not a fixed request rate?**
"32 concurrent users" means 32 clients, each waiting for its answer before asking again. That's a closed loop. An open loop at a fixed rate measures behavior beyond saturation, where queues grow without bound. Both are valid. I report closed-loop as the headline and use the burst test for overload behavior.

**Q12. Why `ignore_eos` in the benchmarks? Isn't that unrealistic?**
It makes output length a property of the workload, not the system. Without it, two systems could generate different lengths for the same prompt, which makes throughput comparisons meaningless. vLLM's own serving benchmark does the same. Mixed lengths still come from the workload's different `max_tokens` values.

**Q13. What's the bottleneck at 32 users?**
Decode step time, split between the fixed per-forward cost (weight read plus Python overhead) and the per-slot KV cost (attention plus copies, including padding). The profiler gives the split, and the calibrated `StepCost` gives the two coefficients directly. On Colab, the 2 vCPUs can also become a bottleneck: tokenization, SSE and the engine loop share them.

**Q14. Why not use Hugging Face's own continuous batching or TGI?**
Because the scheduling layer is the point of the project. Using an existing engine would leave me able to configure it, not explain it. vLLM is in the project as the reference, so the comparison is honest.

**Q15. What breaks under bad input?**
- **Empty prompt:** 422 from validation.
- **`prompt + max_tokens` too long:** 400.
- **Bad `temperature` or `top_p`:** 422.
- **Unknown token ids** (out of vocabulary) would raise inside the embedding lookup. The engine catches it, fails that micro-batch with 500, and keeps serving everyone else. Validating ids against the vocab size at the API is a cheap addition.
- **A client that never reads its stream:** tokens accumulate in its queue, a few bytes each. There's no per-request cap, so that's an open limitation.

**Q16. What happens if a client disconnects mid-generation?**
Starlette closes the SSE generator, and the `finally` block calls `cancel`. The engine sees the flag on its next step, finishes the row as cancelled, evicts it, and releases its budget. A client that disconnects while still queued is caught by the one-second disconnect poll in `_first_item`.

**Q17. What if an OOM happens anyway?**
`_crash` catches it per micro-batch or per lane. It fails only those requests with 503, counts the OOM, empties the CUDA cache, and the loop continues. One bad batch doesn't take down the server, and the count shows up in `/stats` and the burst report.

**Q18. How does it scale to bigger models or more GPUs?**
- **Bigger model on one GPU:** the budget and cost model recalibrate automatically. A bigger model has a larger per-slot KV cost, so lanes split sooner.
- **More GPUs:** this design is single-GPU. Tensor parallelism would need sharded weights and collective ops inside the forward pass, out of scope.
- **Horizontal scaling:** several gateway replicas behind a load balancer, each with its own GPU. The 429 plus Retry-After behavior composes well with that.

**Q19. What's the scaling limit of the scheduler itself?**
Per step it's O(waiting) to build the pool and O(rows²) for `best_cut` every 8 steps. Both are tiny at these sizes. The real limits are the batch size (`max_batch_size=32`), where KV traffic starts to dominate step time, and the GIL, when request rates get high enough that Python HTTP work competes with the engine loop.

**Q20. Why reserve budget per lane instead of per request?**
Because real memory is per lane. A request's KV occupies `lane_width` columns, not its own length, once it's in a padded batch. Per-request reservation would undercount padding and could OOM. This is also why lanes help admission: less padding means a smaller footprint for the same requests.

**Q21. Is your cost model accurate?**
It's a two-parameter linear fit measured at two widths. It ignores secondary effects: per-lane sampling and sync overhead, cache-growth copies, and the merge and split copies themselves. That's why the split threshold has a 1.5× margin and merging has hysteresis. On my Mac with the 0.5B model, even forced splitting lowered throughput while cutting padding by about half, and the calibrated model correctly declined to split. On the T4, lanes made no difference with short prompts (279.9 vs 282.1 tok/s) and gave 1.17× throughput and 18% lower p99 latency with long, mixed prompts (128–2048 tokens).

**Q22. Why only 32 rows max?**
It covers the target load (32 users). Beyond that, padded KV traffic grows linearly with rows while the fixed weight-read cost is already amortized, so returns diminish, and memory spent on KV competes with prefill headroom. It's a flag, and the sweep can test other values.

**Q23. What would you do differently?**
- **Write my own KV cache** (preallocated, paged) instead of reusing HF's `DynamicCache`. That removes the per-step `cat`, the join and evict copies, and padding.
- **Use a custom attention path** that keeps grouped-query attention without expanding KV heads under a mask.
- **Split the HTTP server and engine into two processes** once CPU becomes the limit.

**Q24. How confident are you in the numbers?**
They come from one GPU type with a fixed seed and workload, warm-up excluded, and the same client for every system. I'd want repeated runs with confidence intervals before quoting small differences, but the ratios to the naive baseline are large enough that run-to-run noise doesn't change the conclusion. The raw per-request records are saved in the results JSON, so anyone can recompute.

**Q25. Can you show it working?**
Yes. Start the server, send a streaming curl, open `/stats` and `/metrics` while the load tester runs, and open the profiler trace in Perfetto to show prefill, decode, merge and evict ranges on the timeline.

---

## 6. Limitations and what version 2 adds

**Where it stands now:**
- **Padded attention:** single GPU, left-padded batches, so padded positions still cost bandwidth. Lanes reduce it when the hardware makes that worth it.
- **Cache growth copies:** HF's `DynamicCache` copies the cache on every append and on every join or evict.
- **Python per layer:** the forward pass runs Python per layer per step; there are no CUDA graphs.
- **Worst-case reservation:** reserving for `max_tokens` over-reserves when requests stop early.
- **No prefix caching:** two requests with the same system prompt each prefill it from scratch.
- **Raw completions only:** no chat template, no authentication.

**Version 2:**
1. **Paged KV cache with a block allocator** and an attention kernel that reads pages in place (Triton). That means no padding, no join or evict copies, and admission per block instead of per worst case.
2. **Chunked prefill.** Split long prompts across steps so they don't stall decode for everyone else.
3. **Preemption with recompute,** so reservations can be optimistic and only the newest request gets bumped when memory runs short.
4. **Prefix caching,** to share KV for common prompt prefixes.
5. **CUDA graphs for decode,** removing Python overhead per step.
6. **A separate tokenizer/HTTP process,** taking CPU work off the engine's critical path.
