# vLLM lab: prefix caching, KV pressure and CPU KV offload under multi-turn tool traffic (2026-10-30)

E067 kvcache lab (hands-on, upstream vLLM 0.20.1rc1.dev31, one RTX 3090). Code and raw runs:
`experiments/E067_vllm_lab/kvcache/`. The question: what does the scheduler do once the sum of the agents' contexts
exceeds the KV pool, why do prefix hits collapse, when does CPU offload help, and which knobs matter for
concurrency, gates and offload sizing.

## TL;DR

__TLDR__

## Setup

- **Server:** GPU 1 (RTX 3090), vLLM 0.20.1rc1.dev31 (upstream). Settings:
  - prefix caching on, chunked prefill 2,048 tokens/step and max-num-seqs 256 (the defaults), async scheduling, FCFS;
  - `--generation-config vllm`;
  - `--kv-cache-memory-bytes 2.4e9`, which gives a fixed KV pool of **195,312 tokens** (12,207 blocks of 16).
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (12,288 B of KV per token). The two suggested models have no weights in the
  local HF cache (`Qwen2.5-Math-1.5B-…-vocab152064` has only refs, `R1-Distill-Qwen-7b` only its tokenizer), and
  downloading either is over 1 GB. KV and scheduler dynamics depend on token counts, not on model quality.
- **Load (`loadgen.py`):** closed-loop slots run attempts back to back.
  - Attempt structure: 9-15 turns, starting from 3,000 tokens (a shared 400-token system prefix plus a unique task).
  - Each turn appends the sampled reply (forced length U(200, 800), `ignore_eos`) and a tool output of U(200, 2,000)
    random ids, then waits U(1, 4) s of think time.
  - Prompts are token ids (the `--token-exact` pattern), so the ideal hit rate is 90%: everything except the newest
    tool output was seen before. Final contexts are about 22k tokens, the mean in-flight prompt about 11k.
  - Runs last 300 s and are measured over t = 60-300 s. Every run starts a fresh server, so the cache is cold and the
    lock is released between runs.
- **Working set** (sum of all attempts' contexts, ≈ C × 11k) relative to the 195k-token pool: C=8 0.45×, C=16 0.9×,
  C=32 1.8×, C=64 3.6×. The Kaggle analogue is 64-80 attempts × ~30k tokens against a pool of 1.5-3M tokens, that is
  about 1-1.5× late in the run.
- **Instrumentation (observers only):**
  - `kvlab_sched.LabScheduler`, a subclass of the default `AsyncScheduler` loaded with `--scheduler-cls`. Per step it
    records scheduled prefill, decode and recompute tokens (recompute: prefill below the request's earlier high-water
    mark), steps where the waiting queue was blocked, and the evictable cached blocks.
  - `--kv-cache-metrics` (block idle-before-evict and reuse gap, 5% sampled).
  - Client-side TTFT and `cached_tokens` per turn (`--enable-prompt-tokens-details`).
  - `/metrics` at 1 Hz, and the server process tree's RSS.

## What the scheduler does when KV is full (vLLM 0.20 V1, code-level)

File references are to `vllm/v1/core/sched/scheduler.py`, `kv_cache_manager.py` and `block_pool.py` in 0.20.

1. **Running requests go first.** Each step walks the running queue and allocates blocks for each request's new
   tokens; in decode, that is a new block every 16 tokens.
   - If no block is free, it **preempts the newest running request** (FCFS: `self.running.pop()`), frees its blocks,
     sets `num_computed_tokens = 0` and puts it at the front of the waiting queue.
   - V1 has only recompute-mode preemption: no swap, and no V0-style watermark.
2. **Then the waiting queue, but only in steps without a preemption.**
   - The head request is looked up in the GPU prefix cache (`get_computed_blocks`), then in the connector (CPU) for
     the rest.
   - `scheduler_reserve_full_isl` (default True) then requires that the **whole prompt** fits in free blocks
     (`can_fit_full_sequence`). If it does not, `break`: nothing behind the head is admitted that step. This is
     strict FCFS **head-of-line blocking**.
   - Only the first chunk (2,048 tokens) is computed in that step; the rest follows in later steps.
3. **"Free" includes the cache.**
   - A finished turn's blocks go to the tail of the free queue with their hashes intact: evictable cache, in LRU order.
   - Allocation pops from the head of the queue and evicts the hash.
   - `kv_cache_usage_perc` counts only blocks referenced by running or loading requests. It says nothing about how much
     of the free part is still a useful cache.
4. **Between turns an attempt owns nothing.**
   - Its context exists only as evictable blocks. The next turn hits only if they survive the gap: think time plus the
     time it waits in the queue.
   - Survival time ≈ evictable blocks ahead of them ÷ block-allocation rate. As running contexts approach the pool,
     both move the wrong way: there are fewer evictable blocks, and more allocation, because every miss allocates and
     prefills its whole context.
5. **The cliff is a positive feedback loop.**
   - A miss makes the turn prefill 10-25k tokens instead of about 1.5k, so the engine becomes prefill-bound (steps/s
     fall 246 → 44).
   - Turns take longer and more attempts queue. A queued request's own prefix sits unprotected in the free queue and
     is evicted by the admissions ahead of it. **Waiting in the queue destroys your own cache**, which causes more
     misses.
   - At about 93% KV usage, the head is admitted only when a running request finishes. So running ≈ pool ÷ context ≈
     14-16, and everyone else waits.
6. **Preemption is not the problem here.**
   - With `reserve_full_isl` the prompt is checked to fit before admission, so only decode growth (about 500
     tokens/turn) can trigger preemption.
   - Measured recompute after preemption is ≈ 0% of prefill. The lost throughput is re-prefill of evicted prefixes.
7. **CPU offload (`--kv-offloading-size GiB`, which selects the native OffloadingConnector):**
   - **Store:** every full block is copied GPU→CPU as soon as it is computed (write-through, async: `_get_reqs_to_store`)
     into an LRU pool of `cpu_bytes_to_use` (`eviction_policy` lru or arc; `store_threshold` filters out blocks seen
     only once).
   - **Load:** at admission, the remaining prefix after the GPU hit is looked up on the CPU. On a hit, GPU blocks for
     the **whole** prefix are allocated at once and an async CPU→GPU load is issued.
     - Until the load finishes, the request is parked as `WAITING_FOR_REMOTE_KVS` in `skipped_waiting`: it holds GPU
       blocks but is counted under `num_requests_waiting`.
     - A finished request whose stores are still in flight keeps its GPU blocks until they complete.
   - **Offload turns a recompute into a PCIe copy. It does not enlarge the GPU pool:** a request still needs its whole
     context in GPU blocks to run.
   - In 0.20 it refuses to start with the hybrid KV manager on ("Connector OffloadingConnector does not support HMA"),
     so pass `--disable-hybrid-kv-cache-manager`. The SimpleCPUOffloadConnector (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`) does
     support HMA.
8. **The remaining knobs** (there is no watermark in V1):
   - `--scheduler-reserve-full-isl`: admission needs the whole prompt free;
   - `--max-num-seqs`: caps running;
   - `--max-num-batched-tokens` / `--long-prefill-token-threshold`: chunking;
   - `--scheduling-policy priority`: preempts the lowest priority instead of the newest.

## Exp 1: no offload, concurrency past the KV pool

Window t = 60-300 s. Pool: 195k tokens.

| C | working set / pool | gen tok/s | turns/min | attempts done (mean s) | GPU hit, per request | GPU hit, Prometheus | prefill computed tok/s | recompute | preempt/min | waiting queue blocked | running | waiting | KV usage | evictable cache (k tok) | block idle before evict | TTFT p50 / p90 / max (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.45× | 869 | 103 | 36 (56) | **0.90** | 0.90 | 2.1k | 0% | 0 | 0% | 3.7 | 0.0 | 0.23 | 147 | 36 s | 0.07 / 0.11 / 0.24 |
| 16 | 0.9× | **1104** | **131** | **49** (85) | 0.68 | 0.18 | 8.5k | 0% | 0.5 | 6% | 10.3 | 0.3 | 0.65 | 67 | 8.4 s | 0.11 / 0.93 / 3.6 |
| 32 | 1.8× | 640 | 79 | 30 (205) | **0.04** | 0.02 | 16.4k | 0% | 1.3 | **97%** | 14.6 | 13.8 | 0.93 | 12 | 11 s | 11 / 14 / 16 |
| 64 | 3.6× | 747 | 92 | 14 (264) | 0.03 | 0.03 | 17.0k | 0% | 3.8 | 97% | 16.2 | 43.8 | 0.94 | 9 | 11 s | 31 / 40 / 44 |

- **The cliff is a step, not a slope.** From C=16 to C=32, hits fall 0.68 → 0.04 and turns/min 131 → 79 (−40%).
  - Doubling the attempts lowers throughput.
  - Each turn now re-prefills its whole context: computed prefill tokens run at 16-17k tok/s against ~1.5k of
    genuinely new tokens per turn.
- **C=16, the working set at 0.9× the pool, is metastable.** The 30-s series alternates healthy windows (hit 0.81-0.84,
  1,250-1,340 gen tok/s, KV 0.5-0.6) with degraded ones (hit 0.29-0.55, 800-950 tok/s, KV 0.7-0.83), and recovers on
  its own.
- **C=32 locks into the bad state within 30 s and never recovers:**

  | t (s) | 0-30 | 30-60 | 60-90 | 90-120 | 150-180 | 270-300 |
  |---|---|---|---|---|---|---|
  | running / waiting | 9.9 / 0 | 22.4 / 3.9 | 16.8 / 11.0 | 14.2 / 13.9 | 13.5 / 15.5 | 15.8 / 12.7 |
  | GPU hit | 0.69 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 |
  | gen tok/s | 1,570 | 1,052 | 767 | 639 | 594 | 667 |

  This is the teammate's picture: running falls (22 → 14) while requests wait, KV sits at 0.93, and hits are near
  zero.
- **Preemption is negligible (≤ 4/min) and recompute is 0%:** the damage is cache eviction plus head-of-line admission.
- **Tails:**
  - At C=32, 72% of turns wait more than 10 s for their first token.
  - At C=64, 54% wait more than 30 s, and a typical turn takes 40 s for 500 tokens that run at about 270 tok/s per
    stream when unloaded.
  - Waits scale with queue length × prefill time per turn. With L4-scale prefill and 30-60k contexts, this is how a turn
    reaches the 900 s timeout.
- **V1 has no watermark knob.** The admission knobs (reserve-full-isl, max-num-seqs) are in the Exp 3 table.

## Exp 2: CPU KV offload (OffloadingConnector)

__EXP2__

## Exp 3: admission control

__EXP3__

## Practical rules for agentic tool-use traffic

__RULES__

## Measurement pitfalls found

__PITFALLS__

## serve_replay_bench.py: first real-server smoke

__BENCH__

## Files

- `experiments/E067_vllm_lab/kvcache/loadgen.py`: multi-turn load generator with optional KV or cache-health gates.
- `experiments/E067_vllm_lab/kvcache/kvlab_sched.py`: instrumented scheduler (observer only).
- `experiments/E067_vllm_lab/kvcache/run_suite.py`: one server per run under the GPU-1 lock.
- `experiments/E067_vllm_lab/kvcache/analyze.py`: the tables and series (`analyze.py RUN... [--series 30]`).
- `experiments/E067_vllm_lab/kvcache/bench_smoke.py`: the replay-bench smoke test.
- `experiments/E067_vllm_lab/kvcache/runs/`: raw per-run turns, metrics, scheduler logs and server logs.
