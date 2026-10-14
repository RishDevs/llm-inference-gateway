# Code Walkthrough

This document goes through the project file by file and block by block. Every significant piece gets two explanations:

- **Plain:** the idea in everyday terms, with an analogy.
- **Technical:** the exact mechanics, the data shapes, and what would break if it were written differently.

Read sections 1–3 first. They give you the whole system in your head. After that, the file sections can be read in any order. Section 13 lists the blocks worth knowing cold.

---

## Contents

1. [The whole system as one story](#1-the-whole-system-as-one-story)
2. [Core ideas you need first](#2-core-ideas-you-need-first)
3. [End-to-end execution trace](#3-end-to-end-execution-trace)
4. [gateway/config.py](#4-gatewayconfigpy)
5. [gateway/model.py](#5-gatewaymodelpy)
6. [gateway/engine.py](#6-gatewayenginepy)
7. [gateway/admission.py](#7-gatewayadmissionpy)
8. [gateway/scheduler.py](#8-gatewayschedulerpy)
9. [gateway/metrics.py](#9-gatewaymetricspy)
10. [gateway/protocol.py](#10-gatewayprotocolpy)
11. [gateway/server.py](#11-gatewayserverpy)
12. [bench/, tests/, scripts, Docker, CI](#12-bench-tests-scripts-docker-ci)
13. [The load-bearing blocks](#13-the-load-bearing-blocks)

---

## 1. The whole system as one story

**Plain.** Picture a restaurant with one very fast chef (the GPU). The chef can stir 32 pots at once almost as fast as one pot. The expensive part is walking to the pantry (memory), and one trip serves every pot on the stove.

- The **front desk** (FastAPI) takes orders, gives each customer a pager (an `asyncio.Queue`), and puts the ticket on a rail (the waiting list).
- The **kitchen manager** (the scheduler thread) keeps pulling tickets off the rail, starting new dishes in small groups (prefill micro-batches) and sliding them onto the stove next to dishes already cooking (merge). Every round, each pot gets one more stir (one decode step), and the manager calls each customer's pager with the latest bite (a token). Finished dishes come off the stove right away (evict) so their burner frees up.
- Before taking a new group, the manager checks the **fridge space** (the token budget). If the dishes might not fit, the group waits. If the rail is completely full, the front desk tells new customers "come back in a second" (HTTP 429).
- Pots of very different sizes on the same stove waste space, because every pot is sized for the biggest one (padding). When that waste is big enough to matter, the manager moves the small pots to a second stove (a lane). Running a second stove costs a fixed amount of effort per round, so the manager only does it when the numbers say it's worth it (the cost model).

**Technical.** One process, two kinds of execution:

- The asyncio event loop handles HTTP, tokenization, incremental detokenization and SSE writes.
- One engine thread owns every CUDA call. Each `Scheduler.step()`:
  - expires stale waiting requests;
  - admits one prefill micro-batch into one lane (budget- and cap-checked);
  - runs one decode forward pass per lane;
  - evicts finished rows;
  - every 8 steps, rebalances lanes with a calibrated cost model `t ≈ fixed + per_slot × rows × width`.
- Tokens cross from the engine thread to the event loop through `loop.call_soon_threadsafe(queue.put_nowait, token)`.

---

## 2. Core ideas you need first

### 2.1 Tokens and autoregressive generation

**Plain.** The model writes like someone who can only add one word piece at a time, and has to reread the whole page before each one.

**Technical.** A causal LM maps a token sequence to a probability distribution over the next token. Generation is a loop: sample a token, append it, run the model again. Qwen2.5's vocabulary has 151,936 tokens, so the output at each position is a vector of 151,936 scores (logits).

### 2.2 Prefill vs decode

**Plain.** Prefill is reading the question. Decode is writing the answer one word at a time. Reading is done all at once, but writing has to happen one word after another.

**Technical.**
- **Prefill** runs the whole prompt `[rows, prompt_len]` through the model in one pass. It's compute-bound: lots of matrix multiplies on many tokens.
- **Decode** runs `[rows, 1]`, one new token per sequence. It's memory-bandwidth-bound: every step reads all ~3 GB of fp16 weights from GPU memory to do a small amount of math.

That's why batching decode is nearly free: 32 rows share one weight read.

### 2.3 The KV cache

**Plain.** Instead of rereading the whole page each time, the model keeps sticky notes summarizing every word it has seen. Each new word only needs a new sticky note plus a glance at the old ones.

**Technical.** In attention, each token produces a key (K) and a value (V) per layer. Caching them means decode only computes K/V for the new token.

- **Shape per layer:** `[rows, kv_heads, seq_len, head_dim]`. For Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128.
- **Bytes per token:** `2 (K and V) × 28 × 2 × 128 × 2 bytes = 28,672`, about 28 KB.
- **For one request:** 2,000 tokens is about 57 MB, so 32 such requests is about 1.8 GB.

The cache grows by one token per row per step. That's why memory, not compute, limits how many users fit.

### 2.4 Padding and masks

**Plain.** To stack papers of different lengths into one neat pile, you add blank sheets to the short ones. A note on each sheet says "blank, ignore me" (the mask).

**Technical.** GPU tensors are rectangular, so sequences of different lengths are padded to the longest. This project pads on the **left**, so every row's newest token sits in the same final column. That's the column decode writes to.

- The attention mask `[rows, width]` has 1 for real tokens and 0 for padding. Attention never looks at masked positions.
- Padding still costs memory, and in HF's SDPA path it costs bandwidth too (section 6.11). Every padded position is pure waste.

### 2.5 Position ids and RoPE

**Plain.** Each word carries a seat number so the model knows word order. Blank sheets shouldn't shift anyone's seat number.

**Technical.** Qwen2 uses rotary position embeddings, rotating Q and K by an angle that depends on the position. With left padding, the real tokens of a short row start at column 3, not column 0. Passing `position_ids = cumsum(mask) − 1` gives them positions 0, 1, 2…, exactly as if they'd been run alone. That's why batched output matches solo output exactly; there's a test for it.

### 2.6 Why the GPU is idle when serving one user

**Plain.** The chef spends most of the time walking to the pantry. One trip can carry ingredients for everyone, so making one trip per customer is wasteful.

**Technical.**
- **Arithmetic intensity at batch 1:** about 2 FLOPs per weight byte read. A T4 can do about 65 TFLOPS of fp16 tensor math but only moves about 300 GB/s, so at batch 1 it's limited by bandwidth by a factor of over 100.
- **What batching buys:** it raises FLOPs per byte, so throughput grows almost linearly with batch size until compute or KV-cache traffic starts to dominate.

---

## 3. End-to-end execution trace

### 3.1 Startup

1. `python -m gateway.server --preset gateway` → `server.main()` → `parse_args()` builds `Settings`. The order is defaults → env vars → preset → CLI flags, then `validate()`.
2. `uvicorn.run(create_app(settings))`. The FastAPI `lifespan` hook runs `build_engine` in a worker thread (`asyncio.to_thread`) so the event loop isn't blocked during the model load.
3. `build_engine` runs these in order:
   - `model.load()` picks cuda/mps/cpu and fp16/fp32, and loads the tokenizer and model with `attn_implementation="sdpa"`.
   - It builds a `ModelRunner`.
   - `size_budget()` measures free GPU memory and converts it to KV slots.
   - `runner.calibrate()` (only when bucketing is on) times decode at two widths to get `StepCost`.
   - It builds the `Scheduler`.
4. `scheduler.start()` launches the `engine` daemon thread running `_loop()`.

### 3.2 One streaming request

1. The client POSTs `{"prompt": [...], "max_tokens": 128, "stream": true}`.
2. FastAPI validates it into `CompletionRequest`. The handler `completions()` tokenizes the prompt if it's a string.
3. The handler builds `Sequence(prompt_ids, max_tokens, …, loop=<event loop>, outbox=asyncio.Queue())`.
4. `scheduler.submit(seq)` checks `need ≤ max_model_len`, checks `need ≤ budget`, and checks queue length under the lock. It appends to `waiting` and sets the wake event. The failure modes: `RequestTooLarge` becomes HTTP 400, `QueueFull` becomes 429 with `Retry-After`.
5. The handler awaits `_first_item()`, polling for client disconnect every second.
6. **On the engine thread,** `step()` runs:
   - `_admit` → `_pick` decides this request is ready, routes it to a lane, gathers others for the same lane, and reserves budget.
   - `runner.prefill(picked)` returns a `Batch` plus the first token per row.
   - `_settle` → `runner.settle` → `seq.push(token)` → `loop.call_soon_threadsafe(outbox.put_nowait, token)`.
   - `runner.merge` slides the new rows into the lane, and `_shrink` evicts any row that already finished.
7. Back on the event loop, `_first_item` gets the token. The handler returns `StreamingResponse(_sse(...))`. `_sse` feeds each token to `TextStream`, yields `data: {...}` chunks, and stops on `Done`, `Failed` or a stop string.
8. Each following `step()` runs `_decode(lane)`:
   - `runner.decode(batch)` does one forward pass;
   - `sample` picks the next tokens;
   - `_advance` appends them to each `seq.output_ids`;
   - `settle` pushes them and marks `finish("length" | "stop")`;
   - `_shrink` evicts finished rows and `_resize` returns unused budget.
9. When the request finishes, `Done("length")` arrives. `_sse` sends the final chunk with `finish_reason`, then the usage chunk if requested, then `data: [DONE]`.
10. If the client disconnects mid-stream, Starlette closes the generator and the `finally` block calls `scheduler.cancel(seq)`. The engine evicts the row next step.

### 3.3 Data shapes along the way

| Stage | Data |
|---|---|
| HTTP body | JSON, prompt as `list[int]` or `str` |
| `Sequence` | `prompt_ids: list[int]`, `output_ids: list[int]` |
| Prefill input | `ids [r, w]`, `mask [r, w]`, `position_ids [r, w]` (int64) |
| Prefill output | `logits [r, 1, 151936]` (only the last position) and a cache of 28 layers × K/V `[r, 2, w, 128]` |
| Lane | `mask [R, W]`, `next_pos [R]`, `last [R]`, `temps [R]`, `top_ps [R]` |
| Decode input | `last.view(R, 1)`, `mask [R, W+1]`, `next_pos.view(R, 1)` |
| Engine → loop | Python `int` token ids, then `Done(reason)` or `Failed(status, msg)` |
| SSE | `data: {"choices":[{"text": "...", "finish_reason": null}]}` |

---

## 4. gateway/config.py

### 4.1 `PRESETS`

**Plain.** Named settings recipes, so the four experiments are each one word away.

**Technical.** A dict of overrides applied over `Settings` defaults:
- `naive` sets batch size 1, no admission and no waiting, so requests run one at a time through the same code path.
- `unbucketed` turns off lanes.
- `unguarded` turns off admission and allows 256 rows, for the burst test.

Keeping baselines as config instead of separate programs means only the variable under test changes.

### 4.2 `Settings`

**Plain.** The control panel: every knob in one place.

**Technical.** A frozen dataclass. Frozen means it can't be changed by accident at runtime, and `dataclasses.replace` makes modified copies. `name` returns the served model name, used in responses and `/v1/models`. `validate()` rejects `max_batch_size < 1` and, with admission on, `max_prefill_tokens < max_model_len`, because a single maximum-length prompt would otherwise never be admissible.

### 4.3 `from_env`, `_coerce`, `apply_preset`

**Plain.** Read settings from environment variables, which is how Docker passes configuration.

**Technical.**
- `from_env` reads `GATEWAY_PRESET` first, then any `GATEWAY_<FIELD>` variable.
- Because of `from __future__ import annotations`, `fields()[i].type` is a string like `"int"` or `"bool"`. `_coerce` switches on that string. Booleans accept `1/true/yes/on`.
- `apply_preset` raises on an unknown name, so typos fail loudly.

---

## 5. gateway/model.py

### 5.1 `pick_device`, `pick_dtype`

**Plain.** Use the best hardware available, and use half-size numbers when the hardware is good at them.

**Technical.** Order: CUDA, then MPS, then CPU. fp16 on GPU (the T4 has no bf16 support), fp32 on CPU (CPU fp16 matmuls are slow and some ops aren't implemented).

### 5.2 `load`

**Plain.** Download the model and its dictionary, put the model on the GPU, and switch it into "answer questions" mode.

**Technical.** `AutoModelForCausalLM.from_pretrained(name, dtype=..., attn_implementation="sdpa")` loads the weights as fp16 and routes attention through `torch.nn.functional.scaled_dot_product_attention`. `.eval()` disables dropout. The pad id falls back to EOS if the tokenizer has no pad token.

### 5.3 `eos_ids_for`

**Plain.** Find every "I'm done talking" signal the model uses.

**Technical.** Qwen instruct models list several EOS ids in `generation_config` (`<|im_end|>` and `<|endoftext|>`). They're merged with the tokenizer's EOS into a frozenset for O(1) membership checks per token.

### 5.4 `kv_bytes_per_token`, `activation_bytes_per_token`

**Plain.** Arithmetic for "how much fridge space does one word take?"

**Technical.**
- `2 × layers × kv_heads × head_dim × element_size`, which is 28,672 bytes for Qwen2.5-1.5B fp16.
- `head_dim` falls back to `hidden / heads` for configs that don't state it.
- Activation bytes per prefill token is an estimate, `2 × (6·hidden + 3·intermediate) × elt`. It covers the residual stream, attention projections, and the three MLP tensors (gate, up, act), with a 2× margin. It sizes the headroom reserved for the largest allowed prefill.

---

## 6. gateway/engine.py

This file is the GPU side: what a request is, what a batch is, and the operations on them.

### 6.1 `Done`, `Failed`

**Plain.** The two kinds of final message a customer's pager can get: "your order is complete" or "sorry, something went wrong".

**Technical.** Frozen dataclasses put on the per-request queue after the token ints. The HTTP layer switches on their type. `Failed` carries an HTTP status, so a queue timeout becomes 503 and an engine bug becomes 500.

### 6.2 `Sequence`

**Plain.** The ticket for one order: what was asked, how long the answer may be, how to reach the customer, and everything written so far.

**Technical.**
- **Fields:** prompt ids, limits, sampling parameters, the event loop and `asyncio.Queue` for delivery, timestamps (`arrived`, `first_token_at`, `finished_at`), `output_ids`, `finish_reason`, a `cancelled` flag set from the event-loop thread, and `reported` so metrics count each request exactly once.
- `need = prompt_len + max_tokens`: the most KV slots this request can ever occupy. Admission and lanes are built on this number.
- `push()`: the only place engine-thread data enters the event loop. It must use `call_soon_threadsafe` because `asyncio.Queue` isn't thread-safe. Calling `put_nowait` directly from another thread can corrupt the queue or fail to wake the awaiting coroutine. The `RuntimeError` guard covers an event loop that already shut down.
- `finish()` and `fail()` are idempotent: the first call wins. Duplicate terminal messages can't happen.

### 6.3 `Batch`

**Plain.** One stove: the dishes on it, plus the shared sticky-note board for all of them.

**Technical.** A lane's live state:
- `seqs`: the rows in order;
- `cache`: an HF `DynamicCache` with 28 layers of K/V `[R, 2, W, 128]`;
- `mask [R, W]`;
- `next_pos [R]`: each row's next RoPE position;
- `last [R]`: the token to feed next step;
- `temps`, `top_ps [R]`.

Properties:
- `width` is the cache length.
- `horizon` is the largest `need` among the rows, used for budgeting.
- `cached` is how many real tokens each row has in the cache: `prompt_len + len(output_ids) − 1`. The newest sampled token hasn't been fed through the model yet. Computing this in Python avoids a GPU→CPU sync that `mask.sum().item()` would force on every step.

### 6.4 `StepCost`, `padding`, `best_cut`

**Plain.**
- `StepCost` is the price list: "running a stove costs X per round, and every empty pot slot costs Y."
- `padding` counts wasted space in a group.
- `best_cut` finds the best place to split a group into short and long so the least space is wasted.

**Technical.**
- `padding(lengths) = n × max − Σ`: how many padded KV positions a left-aligned batch carries.
- `best_cut` sorts rows by cached length and tries every split point. For each, it computes `whole − padding(short) − padding(long)` and returns the row indices of both halves plus the slots saved. That's O(n²) on n ≤ 32 rows, which is negligible. Contiguous cuts in sorted order are optimal for this cost, because any non-contiguous partition can be improved by swapping elements across the boundary.
- `worth_splitting`: `saved × per_slot > fixed × factor`. A new lane adds one forward pass per step (`fixed`) and removes `saved` padded slots per step.
- `worth_merging`: `added × per_slot < 0.5 × fixed × factor`. The 0.5 is hysteresis, so lanes don't split and merge back and forth.

### 6.5 `sync`

**Technical.** CUDA and MPS kernels run asynchronously. Timing code needs `synchronize()` or it only measures how fast Python queued the work. Used only by `calibrate` and the profiler.

### 6.6 `StepStats`

**Technical.** Returned by prefill and decode: rows, width, `slots = rows × width` (positions computed or attended), and `useful` (real ones). `slots − useful` is the padding fed into metrics.

### 6.7 `_layers`, `map_kv`, `zip_kv`

**Plain.** Helpers to apply the same edit to every layer's sticky-note board.

**Technical.** transformers ≥ 4.56 stores the cache as `cache.layers[i].keys / .values`. `map_kv(cache, fn)` replaces each tensor with `fn(tensor)`; `zip_kv(dst, src, fn)` combines two caches layer by layer. `_layers` raises a clear error on older cache formats rather than corrupting state.

### 6.8 `stack_left`, `stack_mask`

**Plain.** To put two piles of different heights side by side, start with an empty frame tall enough for both and slide each pile in so their tops line up.

**Technical.**
- `stack_left` allocates one zero tensor of shape `[Ra + Rb, H, max(Wa, Wb), D]` and copies `a` and `b` right-aligned, so both batches' newest tokens land in the last column.
- That's one allocation and one copy per input. Padding the narrower batch and then calling `torch.cat` would copy the wide one twice.
- The zero padding stays invisible because `stack_mask` puts zeros in the same positions of the mask.

### 6.9 `ModelRunner.prefill`

**Plain.** Read a group of new questions at once and write the first word of each answer.

**Technical.**
1. Build `ids [r, w]` filled with pad, and `mask [r, w]`. Copy each prompt into the right end of its row.
2. `pos = (mask.cumsum(-1) − 1).clamp_(min=0)` gives real tokens positions 0..n−1. Pad positions get 0, which is harmless because they're masked.
3. Move everything to the device with one transfer per tensor, then run `model(..., use_cache=True, logits_to_keep=1)`. **`logits_to_keep=1` is critical.** Without it, prefill materializes `[r, w, 151936]` logits: 32 rows × 1024 tokens × 151,936 × 2 bytes ≈ 10 GB, which on its own would crash a T4.
4. Build the `Batch` with `next_pos = pos[:, -1] + 1` and per-row sampling tensors.
5. `StepStats(r, w, r·w, Σ prompt_len)` records how much of the prefill was padding.
6. Sample and `_advance`.

`@torch.inference_mode()` disables autograd tracking and version counters. That saves memory and a little time compared with `no_grad`.

### 6.10 `ModelRunner.decode`

**Plain.** Every dish on this stove gets one more stir: one new word per answer.

**Technical.**
- Append a column of ones to the mask, so the token being fed is real.
- Call the model with `input_ids [R, 1]`, `position_ids [R, 1]` (each row's own position), the lane's cache, and `use_cache=True`. HF's `DynamicLayer.update` concatenates the new K/V onto each layer.
- Increment `next_pos`.
- `useful = Σ cached + R` counts the real positions attended this step.
- Sample, then `_advance`.

All rows in a lane are live at decode time, because finished rows were evicted after the previous step.

### 6.11 Why padding is expensive here

**Plain.** The kitchen's rules make every blank sheet get photocopied six times before anyone ignores it.

**Technical.** Qwen2.5-1.5B uses grouped-query attention: 12 query heads share 2 KV heads. When an attention mask is present, HF's SDPA integration can't use PyTorch's native GQA path, so it calls `repeat_kv` and materializes K and V at 12 heads, a 6× copy every step. Padded positions get copied and attended like real ones, so a lane with 40% padding spends a large share of decode time on nothing. The profiler measures this, and the lanes exist to reduce it.

### 6.12 `ModelRunner.merge`

**Plain.** Slide a tray of freshly started dishes onto a stove that's already cooking.

**Technical.**
- If there's no running lane, the joining batch becomes the lane.
- Otherwise `zip_kv(running.cache, joining.cache, stack_left)` combines every layer. `stack_mask` combines masks, and the per-row tensors are concatenated.
- Rows keep their own `next_pos`, so positions stay correct even though columns shifted.
- `joining.cache = None` drops the reference so its memory can be freed.

This is what makes the batching continuous: new requests start decoding alongside old ones instead of waiting for the old ones to finish.

### 6.13 `ModelRunner.evict` and `take`

**Plain.** Take finished dishes off the stove. If the tallest pot just left, shrink the frame so the remaining pots aren't sitting in extra empty space.

**Technical.**
- `evict` returns the batch untouched if nothing finished. That's the common case, and it costs no copy and no sync. If everything finished, it returns `None`. Otherwise it calls `take` with the live row indices.
- `take(batch, rows)` computes `lead = width − max(cached[rows])`, the number of leading columns that are padding for every kept row. It slices `[:, :, lead:, :]` and then `index_select`s the rows. Slicing first means `index_select` reads only the kept window and writes one compact tensor.
- `take` shallow-copies the cache object and its layer objects before replacing tensors. That way `take` can produce two independent lanes from one batch (used by `_split`) without both pointing at the same layer objects.
- Without the trim, a batch that once held a 1,500-token row would keep 1,500 columns forever. Every later step would attend to and copy that dead width.

### 6.14 `ModelRunner._time_decode` and `calibrate`

**Plain.** Before opening, the chef times one round on a nearly empty stove and one round on a full stove, and works out the price list.

**Technical.**
- `_time_decode(rows, width, steps)` prefills dummy rows of the given width and times a few decode steps with device sync, taking the median.
- `calibrate` does a warm-up (so CUDA kernel selection and allocator growth aren't counted), then measures at widths 32 and 768 with 16 rows. It fits `per_slot = Δt / (rows × Δwidth)` and `fixed = t_short − per_slot × rows × 32`.
- It runs on the real GPU and model, so the split rule adapts to the hardware. On a small model with high per-call overhead, splitting rarely pays; on a bigger model with an expensive KV path, it pays sooner.

### 6.15 `ModelRunner._advance` and `settle`

**Plain.** Write the new word on each ticket, then call each customer's pager and check whether their order is done.

**Technical.**
- `_advance` sets `batch.last = picked` (the next step's input) and calls `picked.tolist()`. That's the one unavoidable GPU→CPU sync per step: the tokens have to reach Python to be streamed. It appends to `output_ids` and stamps `first_token_at`.
- `settle` pushes each token, then finishes the sequence as `cancelled`, `stop` (EOS seen, unless `ignore_eos`) or `length` (hit `max_tokens`).
- It's split into two methods so the engine thread finishes all tensor work before doing per-request Python work.

### 6.16 `sample`

**Plain.** Choose the next word: either the single most likely one, or a random pick among the likely ones, with each customer's own settings.

**Technical.** On logits `[R, V]` cast to fp32 (fp16 softmax over 151,936 entries loses precision):
1. `greedy = argmax`. If no row has `temperature > 0`, return it straight away. The benchmarks use temperature 0, so this fast path is what they exercise.
2. Otherwise: scale by per-row temperature, softmax, sort descending, and zero every token whose cumulative probability *before* it already exceeds `top_p`. The top token always survives.
3. `multinomial` on the unnormalized remainder (it normalizes internally), `gather` back to vocabulary ids, and `where(hot, drawn, greedy)` to mix greedy and sampled rows in one batch.

---

## 7. gateway/admission.py

### 7.1 `QueueFull`, `RequestTooLarge`

**Technical.** Exceptions raised from `submit` on the event-loop thread, mapped to 429 and 400. `QueueFull` carries `retry_after` for the header.

### 7.2 `TokenBudget`

**Plain.** The fridge-space ledger. Before a group of dishes starts, its worst-case space is written down. When dishes finish, space is crossed off.

**Technical.** A counter with `capacity` (`None` means unlimited) and `used`.
- `fits(n)` checks whether `used + n ≤ capacity`.
- `take(n)` checks and reserves atomically under a lock.
- `give_back(n)` releases and clamps at 0.

The scheduler always passes deltas. A lane's reservation goes from `held` to `footprint`, so the ledger always equals the sum of current lane footprints.

### 7.3 `size_budget`

**Plain.** Look at how much fridge space is free, set some aside for prep work, and see how many sticky notes fit in the rest.

**Technical.**
- With admission off it returns `None` (unlimited). An explicit `token_budget > 0` wins over the calculation.
- On non-CUDA devices it returns `max_model_len × max_batch_size`, because there's no reliable free-memory call there.
- On CUDA it runs after the model has loaded, so the weights are already subtracted: `torch.cuda.mem_get_info()` gives free bytes. It computes `usable = free × 0.85 − max_prefill_tokens × activation_bytes_per_token` and returns `usable // kv_bytes_per_token`.
- The 0.85 leaves room for allocator fragmentation and CUDA workspace.

**Why it's safe.** A lane's reservation is `rows × max(need)`. Its K/V tensors are `rows × width`, and width is at most the longest live row, which is at most that row's `need`. Real memory can't exceed the reservation, so the sum over lanes can't exceed the budget. Prefill activations are capped separately by `max_prefill_tokens`. Together these make OOM impossible by construction, as long as the estimates hold, which is what the burst test checks.

---

## 8. gateway/scheduler.py

The policy brain. It runs entirely on the engine thread, except `submit` and `cancel`, which run on the event loop and only touch the waiting list under a lock.

### 8.1 `is_oom`

**Technical.** Recognizes `torch.cuda.OutOfMemoryError`, plus generic `RuntimeError`s containing "out of memory" (older PyTorch and other backends), so they're handled as OOMs rather than bugs.

### 8.2 `Scheduler.__init__`

**Technical.** State:
- `waiting` (FIFO list);
- `lanes` (lane id → `Batch`);
- `reserved` (lane id → reserved slots);
- `budget`, `costs` (defaults to "padding is expensive", so tests split eagerly);
- a lane-id counter;
- a step counter for rebalancing;
- the lock, wake event, halt event and thread handle.

### 8.3 `submit`

**Plain.** The front desk's checks: is the order too big to ever cook, and is the rail already full?

**Technical.**
- Validates an empty prompt, `need > max_model_len`, and `need > budget capacity`. A single request that could never fit is rejected now instead of waiting forever.
- Under the lock it checks `len(waiting) ≥ max_queue` (counted in metrics, raises `QueueFull`), then appends and sets the wake event.
- This is the backpressure point: a burst gets fast, explicit 429s instead of an ever-growing queue that would time out anyway.

### 8.4 `cancel`

**Technical.** Sets `seq.cancelled`. If the request is still waiting, it's removed and finished right away under the lock. If it's running, the engine sees the flag in `drop_cancelled` or `settle` and evicts it on its next step. The cross-thread write is a single attribute assignment, which is atomic in CPython.

### 8.5 `step` and `_loop`

**Plain.** One round in the kitchen: throw out tickets that waited too long, start one new group, stir every stove once, and sometimes reorganize the stoves.

**Technical.**
- `step()` runs `_expire`, `_admit`, `_decode` for each lane, increments the step counter, runs `rebalance` every `rebalance_every` steps, then `_publish`. It returns whether any work happened.
- `_loop` calls `step()` until halted and calls `_idle()` when there's no work. Any unexpected exception fails every request with 500, so clients don't hang, and the loop keeps running.
- On shutdown, everything outstanding gets 503.

Admission happens before decode. A request that arrives during step k gets its first token at the end of step k+1 at worst. That's why TTFT stays at roughly one prefill plus one decode round even under load.

### 8.6 `_idle`

**Technical.** With nothing waiting, it waits on the event for 0.5 s so it can notice shutdown. With something waiting but not yet ready, it waits only until the oldest request hits `max_wait_ms`. Then it clears the event. The lost-wakeup race is harmless: if `submit` sets the event between the check and the wait, `wait` returns immediately, and the next loop re-checks the list anyway.

### 8.7 `_expire`

**Technical.** Removes requests that have waited longer than `queue_timeout_s` and fails them with 503 "timed out waiting for capacity". It's a backstop if load stays above capacity for longer than the queue can absorb.

### 8.8 `join_padding` and `route`

**Plain.** For a new order, which stove would waste the least space if it joined?

**Technical.**
- `join_padding`: if the prompt is shorter than the lane width, the new row carries `width − prompt_len` padding. If it's longer, every existing row gains `(prompt_len − width)` columns, so the cost is `(prompt_len − width) × rows`.
- `route` returns lane 0 when bucketing is off. Otherwise it returns the lane with the least join padding (ties go to the older lane id).

### 8.9 `rebalance`, `_split`, `_merge_lanes`

**Plain.** Every few rounds the manager asks two questions. Would moving the small pots to their own stove save more than the cost of running another stove? Are two stoves now similar enough to combine?

**Technical.**
- **Split:** if lanes < `max_lanes`, take the lane with the most padding and compute `best_cut` on its `cached` lengths. If `worth_splitting(saved)`, call `_split`, which `take`s the long rows into the old lane id and the short rows into a new one. The reservation moves with them, and `_resize` returns the difference: each part's `rows × horizon` is no bigger than the original's.
- **Merge:** otherwise, among lanes sorted by width, take the adjacent pair whose union adds the least padding. If `worth_merging(added)`, call `_merge_lanes`. That combines the reservations into `(Ra + Rb) × max(horizon)`, takes the extra from the budget first (skipping the merge if it doesn't fit), and calls `runner.merge`.
- Splitting and merging only move K/V. The model never recomputes anything, so outputs are unchanged. A test checks that lanes produce exactly the same tokens as a single batch.

### 8.10 `_pool`, `_ready`, `_pick`

**Plain.** Choose which waiting orders start together. Always start with the order that's waited longest, add others headed for the same stove, and stop before the group breaks a rule.

**Technical.**
- `_ready(seed, free_rows)`: start now if the waiting list could fill every free row, or if the oldest request has waited `max_wait_ms` (idle) or `join_wait_ms` (busy). Waiting a few milliseconds when idle lets near-simultaneous arrivals share one prefill.
- `_pool` returns the waiting requests routed to the same lane as the seed (or all of them with bucketing off or only one lane).
- `_pick` walks the pool in arrival order and adds a request only if all of these hold:
  - rows stay under `max_batch_size`;
  - the padded prefill `rows × max_prompt` stays under `max_prefill_tokens` (the first request is always allowed);
  - the lane's new footprint `(lane_rows + picked + 1) × max(horizon, need)` minus what the lane already holds fits the budget.
- If the seed itself doesn't fit, nothing is picked. The seed isn't skipped, so big requests can't starve. Later candidates that don't fit are skipped with `continue`, so a smaller one behind them can still go.
- Finally it reserves the footprint delta and removes the picked requests from `waiting`, all under the same lock, so `submit` can't interleave halfway through.

### 8.11 `_admit`

**Technical.** Pick, record the micro-batch size, prefill, record prefill padding, settle the first tokens, merge into the lane, then shrink, because rows with `max_tokens=1` or an immediate EOS finish during prefill. If prefill throws, `_crash` fails just the picked requests and `_resize` returns their reservation. The rest of the lane is untouched.

### 8.12 `_decode`

**Technical.** Finish cancelled rows, shrink, run one decode step, record decode metrics, settle, shrink again. If decode throws (an OOM, say), only that lane's rows are failed and its reservation released. Other lanes continue.

### 8.13 `_settle`, `_shrink`, `_resize`

**Technical.**
- `_settle` delivers tokens, counts them, and reports any finished requests.
- `_shrink` reports finished rows (metrics are idempotent through `seq.reported`), evicts them, and deletes the lane if it's empty.
- `_resize` recomputes the lane's footprint `rows × horizon` and gives back any excess. Budget released by finished requests is immediately available to the next `_pick`.

### 8.14 `_crash`, `_fail_everything`, `_publish`

**Technical.**
- `_crash` classifies OOM (503, counted) vs other errors (500, logged with the traceback), fails the given requests, and calls `torch.cuda.empty_cache()` so freed blocks go back to the driver before the next attempt.
- `_fail_everything` drains waiting and running requests and releases every reservation. It's used for engine bugs and shutdown.
- `_publish` updates the gauges once per step.

---

## 9. gateway/metrics.py

**Plain.** The kitchen's scoreboard: how many orders, how fast the first bite arrived, how much space was wasted, how often it nearly ran out of room.

**Technical.** Prometheus `Counter`, `Histogram` and `Gauge` objects on a private `CollectorRegistry`. Being private means several apps in one test process don't collide on the global registry. Every update also writes into a plain `totals` dict under a lock, so `/stats` can return exact JSON without parsing Prometheus text.

- `on_finish` records TTFT (`first_token_at − arrived`) and end-to-end latency once per request, guarded by `seq.reported`.
- `on_slots(phase, slots, useful)` records padding.
- `snapshot` adds `torch.cuda.max_memory_allocated()` when CUDA is present.

The burst report reads `ooms` and `gpu_peak_bytes` from here.

---

## 10. gateway/protocol.py

### 10.1 `CompletionRequest`, `StreamOptions`

**Technical.** A pydantic model with OpenAI field names and bounds checking: `max_tokens ≥ 1`, `0 ≤ temperature ≤ 5`, `0 < top_p ≤ 1`, and non-empty prompts. `prompt` accepts a string or a list of token ids; vLLM accepts both too, which is why the benchmark can send exact-length prompts to both systems. `ignore_eos` is a vLLM extension that both servers honor.

### 10.2 `TextStream`

**Plain.** Turns the model's word pieces back into readable text. It waits when a piece is half of a character (some emoji and non-English characters are split across tokens), and watches for stop phrases.

**Technical.**
- `feed(token)` appends the token and decodes the whole id list. If the text ends in `�` (the replacement character from an incomplete UTF-8 sequence), it emits nothing yet.
- Re-decoding the full list each time is O(n²) over a response, but n ≤ 2,048 and fast tokenizers decode in microseconds. The cost is negligible next to a decode step, and it avoids the classic offset bugs of incremental decoding.
- `_flush` searches for each stop string starting a little before the last emitted position. On a hit, it emits up to the stop and marks `stopped`. Otherwise it emits everything except the last `len(stop) − 1` characters, which could be the start of a stop string.
- `close()` flushes the remainder at the end.

### 10.3 `chunk`, `usage_chunk`, `completion`, `error_body`

**Technical.** Build OpenAI-shaped JSON. SSE framing is `data: <json>\n\n`. The usage chunk has empty `choices`, matching OpenAI's `include_usage` behavior, and it's where the load tester reads exact token counts.

---

## 11. gateway/server.py

### 11.1 `build_engine`

**Technical.** Load the model, build the runner, size the budget, calibrate if bucketing is on, build the scheduler, and log the budget and the calibrated cost. Kept separate from `create_app` so tests can inject a scheduler built on a tiny model.

### 11.2 `create_app` and `lifespan`

**Plain.** Set up the front desk and, when the restaurant opens, start the kitchen.

**Technical.** `lifespan` builds the engine in a worker thread (if one wasn't injected), starts the engine thread, yields, and stops it on shutdown. Routes:
- `/health`
- `/v1/models` (OpenAI clients probe this)
- `/metrics` (Prometheus text)
- `/stats` (JSON, plus the settings that matter for interpreting results)

### 11.3 `completions`

**Plain.** The order window: write the ticket, check it can be accepted, wait for the first bite, then either stream bites or hand over the whole plate.

**Technical.**
1. Tokenize if needed and build a `Sequence` bound to the running loop and a fresh queue.
2. `submit`: `QueueFull` → 429 with `Retry-After`; `RequestTooLarge` → 400.
3. `await _first_item`. **Why wait before streaming:** once a `StreamingResponse` starts, the 200 status is already sent. Waiting for the first item means a queue timeout (503) or an OOM during prefill (503) comes back as a proper HTTP error the client can act on. The cost is zero: the first chunk can't be sent before the first token exists anyway.
4. Non-streaming: `_collect` gathers the text and returns one JSON body with usage.
5. Streaming: `StreamingResponse(_sse(...))` with `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so proxies don't buffer the stream.

### 11.4 `_first_item`

**Technical.** `asyncio.wait_for(queue.get(), 1.0)` in a loop. On each timeout it checks `request.is_disconnected()` and cancels if the client is gone. Without this, a client that gives up while queued would keep its place, and later its GPU work.

### 11.5 `_collect` and `_sse`

**Technical.**
- Both walk the queue items: ints go to `TextStream`, `Done` ends with the mapped finish reason, `Failed` ends with an error.
- A stop-string hit calls `cancel(seq)` so the engine stops generating.
- `_sse`'s `finally` cancels the sequence if the stream ended for any reason other than normal completion. Starlette closes the generator on client disconnect, so this is what frees the GPU slot.
- `FINISH_MAP` turns internal reasons into OpenAI's (`cancelled` → `abort`).

### 11.6 `parse_args` and `main`

**Technical.** argparse mirrors the `Settings` fields. `BooleanOptionalAction` gives `--bucketing/--no-bucketing`. Order: env → preset → explicit flags → `validate()`. `main` configures logging (httpx request logs muted) and runs uvicorn.

---

## 12. bench/, tests/, scripts, Docker, CI

### 12.1 `bench/workload.py`

**Plain.** A fixed, repeatable set of test orders, some short and some long, so every system faces exactly the same customers.

**Technical.**
- A built-in English corpus is tokenized and repeated to at least 4,096 tokens.
- Each job picks a prompt bucket (64/128/256/512) and an output bucket (32/64/128/256), then `jitter` draws uniformly from `(bucket/2, bucket]`, so lengths spread inside each bucket.
- Prompts are slices of corpus token ids, so lengths are exact.
- The seed makes the list identical across runs and systems.

### 12.2 `bench/load_test.py`

**Plain.** A crowd of simulated customers. Each one orders, waits for the full answer, then orders again, and the tool times everything.

**Technical.**
- `fire` streams one request with httpx. It timestamps `sent`, the first chunk with `choices` (TTFT), and the end. It reads `usage.completion_tokens` from the final chunk (falling back to the chunk count) and records status and errors.
- `closed_loop` uses N workers draining a shared job queue. It discards the first N records as warm-up and backs off 0.25 s after a 429.
- `burst` keeps `peak` workers alive, and each checks whether its index is below the current target (`base`, then `peak`, then `base`), recording the phase per request.
- `summarize` computes throughput over the measured window, p50/p99 TTFT, end-to-end latency and inter-token latency, and status counts.
- `main` writes one JSON per run including `/stats`.

Closed-loop testing was chosen over open-loop (fixed arrival rate) because it models N users waiting on answers. That's how "32 concurrent users" is defined.

### 12.3 `bench/profile_decode.py`

**Plain.** Puts the kitchen under a stopwatch and a camera, once with one stove and once with stoves allowed to split, and adds up exactly where the time and the wasted space went.

**Technical.**
- `TimedRunner` subclasses `ModelRunner` and wraps prefill, decode, merge and evict with device-synchronized timers, recording `(rows, slots, useful, dt)` per decode step.
- `drive` runs the real `Scheduler` in-process with N simulated users. There's no HTTP, so the numbers isolate the engine.
- The FIFO run happens inside `torch.profiler.profile` (CPU + CUDA activities). It writes the top-30 op table and a Chrome trace you can open in `chrome://tracing` or Perfetto.
- Then the lanes run happens, with the calibrated cost. Padding time = KV padding slots × calibrated per-slot cost, and prefill padding time = prefill time × padded share.
- It reports padded-token reduction and the in-process throughput ratio.

### 12.4 `bench/report.py`

**Technical.** Loads every results JSON by `label` and builds the headline table at 32 users (the ratios behind each resume claim), the full sweep table, the burst table with server-side OOMs and peak memory, and three matplotlib plots.

### 12.5 `tests/`

**Plain.** Automatic checks that run in under a second on a laptop, with no download.

**Technical.**
- `conftest.py` builds a 2-layer, 64-wide random Qwen2 with SDPA attention, plus a byte-level tokenizer stub.
- The key tests:
  - `test_prefill_batch_matches_one_at_a_time`: batched left-padded output equals solo output, token for token.
  - `test_join_and_evict_mid_generation_matches`: requests joining a running batch midway produce the same tokens as running alone.
  - `test_split_lanes_generate_same_tokens`: splitting lanes doesn't change any output.
  - Budget, prefill-cap, queue, timeout, cancellation and rebalance tests.
  - Server tests: SSE framing, 400, 429 with `Retry-After`, `/stats`, `/metrics`, stop strings, presets and env parsing.
- The equivalence tests are the proof that padding, positions, merge, evict and split are correct. If any of those were wrong, tokens would differ.

### 12.6 `scripts/run_benchmarks.sh`

**Technical.**
- Refuses to start if something already listens on the port, so stale servers can't poison results.
- For each preset: start the server, poll `/health`, run the sweep, stop the server.
- Then the burst runs (gateway and unguarded, long prompts), the profiler, and the report.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation from tensors that keep changing size. Every knob is an env var.

### 12.7 Notebook, Dockerfile, compose, CI

**Technical.**
- **Notebook:** clones, installs, runs the script, then installs vLLM (`--dtype half` because the T4 has no bf16), benchmarks it on port 8001, builds the report, and downloads a zip.
- **Dockerfile:** the `pytorch/pytorch` CUDA runtime image, the package installed with the bench extras, the model cache as a volume, and `ENTRYPOINT python -m gateway.server`.
- **Compose:** reserves one NVIDIA GPU and passes the preset by env.
- **CI:** GitHub Actions on Ubuntu installs CPU-only torch and runs pytest.

---

## 13. The load-bearing blocks

If you only learn eight things cold, learn these:

1. **Left padding plus `position_ids = cumsum(mask) − 1`** (`engine.py`, `prefill`): why batched output equals solo output. Without the position fix, short rows would get shifted positions.
2. **`logits_to_keep=1`** (`prefill`): skips about 10 GB of logits on a big prefill. Without it the burst test crashes on prefill alone.
3. **`merge` with `stack_left`/`stack_mask`**: continuous batching. New requests join running ones at step boundaries.
4. **`evict` / `take` with the lead trim**: frees rows and dead width right away. Without it the cache keeps its maximum width forever.
5. **Reservation = `rows × max(need)` per lane** (`_pick`, `_resize`, `size_budget`): the proof that a lane can't outgrow its memory. That's why OOM can't happen with admission on.
6. **`_pick`'s seed rule and caps**: oldest first (no starvation), prefill cap (bounds activation memory), footprint delta against the budget.
7. **`call_soon_threadsafe` in `Sequence.push`**: the only correct way to hand data from the engine thread to asyncio.
8. **Calibrated `StepCost` + `best_cut` + hysteresis** (`rebalance`): length bucketing that only happens when the measured cost model says it pays.

Close runners-up: waiting for the first item before streaming (real error codes), and the `finally: cancel()` in `_sse` (disconnects free GPU work).
