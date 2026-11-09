# vLLM lab: prefix caching, KV pressure and CPU KV offload under multi-turn tool traffic (2026-10-30)

E067 kvcache lab (hands-on, upstream vLLM 0.20.1rc1.dev31, one RTX 3090). Code and raw runs:
`experiments/E067_vllm_lab/kvcache/`. The question: what does the scheduler do once the sum of the agents' contexts
exceeds the KV pool, why do prefix hits collapse, when does CPU offload help, and which knobs matter for
concurrency, gates and offload sizing.

## TL;DR

- **There is a cliff, not a slope.** Once the sum of all live attempts' contexts (working set W) passes about the KV
  pool, per-request prefix hits drop from 0.68 to 0.04 and throughput falls 40%.
  - Doubling attempts from 16 to 32 *lowers* turns/min (131 → 79), and TTFT goes from 0.1 s to 11 s.
  - The collapsed state is self-sustaining: we never saw it recover while the load lasted.
- **The mechanism is admission plus eviction, not preemption** (preemption ≤ 4/min, recompute 0%).
  - Between turns an attempt's context lives only in the evictable part of the pool. When running requests hold
    ~93% of the pool, it is evicted before the next turn.
  - That turn re-prefills its whole context, turns slow down and queue, and queued requests lose their own prefixes.
  - Strict FCFS with the full prompt checked to fit before admission (`scheduler_reserve_full_isl`) blocks the head
    of the waiting queue in 97% of steps. Running pins at about pool ÷ context (~15); everyone else waits.
- **CPU offload works only if the CPU pool holds the whole working set.**
  - 12 GiB (3× W) restored hits to the ideal 0.90 (CPU 0.87 + GPU 0.03) and gave **2.3×** throughput at 2× overload.
  - 4 GiB (1× W) thrashed: 0-12% hits, −5% to +7% throughput, 47-51 GB of write-through traffic.
  - GPU hits of ~3% with offload are expected.
  - Offload does not raise the running count, so extra concurrency only adds queueing (C=64 vs C=32: same throughput,
    4.7× TTFT).
- **Reactive gates on server signals fail.**
  - A KV-usage gate (0.70) and a hit-rate gate (0.80) both over-admitted while contexts were still small, then stayed
    collapsed: the signals pin at KV 0.94 and hit 0.04, and a gate cannot evict.
  - A fixed cap at the healthy point beat both: 1.5× the throughput and over 20× lower TTFT p90.
  - Gate on projected tokens (the solver knows them), with margin.
- **Metric traps:**
  - `vllm:prefix_cache_hits/queries` counts a lookup every step for the blocked head request, so queries reach 30-35×
    the prompt tokens and the hit rate is understated (0.18 shown vs 0.68 real). Use `prompt_tokens_by_source_total`.
  - `kv_cache_usage_perc` ignores the evictable cache.
  - vLLM exports no recompute counter.
- **Host RAM:** `--kv-offloading-size 12` pins **16 GiB** (power-of-two rounding). The OffloadingConnector needs
  `--disable-hybrid-kv-cache-manager` in 0.20.
- **Not run (local GPU use stopped to protect the cards):** the projected-working-set gate, the max-num-seqs and
  reserve-full-isl A/Bs, gate + small offload, and the replay-bench retry. The hit-gate run is partial (to t=165 s).

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
  C=32 1.8×, C=64 3.6×. The Kaggle analogue is 64-80 attempts × ~30k tokens (≈ 1.9-2.4M) against a pool of 1.5-3M
  tokens, that is about 0.8-1.6× late in the run.
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
- **V1 has no watermark knob** (mechanism point 8). The A/B runs of its stand-ins (reserve-full-isl off,
  max-num-seqs) were not run (local GPU use stopped to protect the cards).

## Exp 2: CPU KV offload (OffloadingConnector)

- **Setup:** `--kv-offloading-size {4,12} --disable-hybrid-kv-cache-manager`, which selects the native
  OffloadingConnector with an LRU CPU pool and the same GPU pool as Exp 1.
- **CPU pool size:** 4 GiB = 349k tokens, which is 1.8× the GPU pool and about 1.0× the C=32 working set. 12 GiB =
  1.05M tokens: 5.4× the GPU pool, 3× the C=32 working set, 1.5× C=64.

| C | CPU pool | gen tok/s | turns/min | attempts done | hit GPU + CPU = total | running / waiting (loading) | KV usage | TTFT p50 / p90 / max (s) | turns with TTFT > 10 s | GPU→CPU / CPU→GPU (GB in 240 s) | host memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | none | 640 | 79 | 30 | 0.04 + 0 = 0.04 | 14.6 / 13.8 | 0.93 | 11 / 14 / 16 | 72% | - | 3.1 GB |
| 32 | 4 GiB | 686 | 84 | 32 | 0.04 + 0.12 = 0.17 | 14.8 / 13.5 (0.1) | 0.93 | 11 / 14 / 16 | 64% | 46.9 / 6.2 | 7.1 GB |
| 32 | 12 GiB | **1473** | **180** | **63** | 0.03 + **0.87** = 0.90 | 15.5 / 9.0 (0.2) | 0.96 | **3.0 / 4.4 / 6.5** | 0% | 14.9 / 95.6 | **19.2 GB** |
| 64 | none | 747 | 92 | 14 | 0.03 + 0 = 0.03 | 16.2 / 43.8 | 0.94 | 31 / 40 / 44 | 100% | - | 3.2 GB |
| 64 | 4 GiB | 706 | 87 | 13 | 0.04 + 0.00 = 0.03 | 15.7 / 44.5 | 0.94 | 33 / 41 / 45 | 100% | 51.0 / 0.05 | 7.2 GB |
| 64 | 12 GiB | 1449 | 179 | 62 | 0.03 + 0.87 = 0.90 | 15.2 / 41.0 (0.2) | 0.96 | 14 / 17 / 20 | 96% | 15.1 / 97.8 | 19.2 GB |

- **A CPU pool that holds the whole working set restores the ideal hit rate (0.90), now served from CPU.**
  - Throughput is 2.3× no offload at C=32 and 1.3× the best no-offload point (C=16, 1,104 tok/s).
  - GPU-side hits stay at 3%. That is the teammate's "CPU restores 90-98%, GPU hits ~3%" picture, and it is the
    *expected* state, not a fault: every running request's context occupies the GPU pool (KV 0.96), so nothing
    survives in GPU memory between turns, and every returning turn reloads from CPU.
  - The number to watch is GPU + CPU, against the ideal.
- **An undersized CPU pool is useless, or worse.**
  - At 4 GiB, C=32 gets 12% CPU hits (+7% tok/s), and C=64 gets 0% hits (−5% tok/s): 51 GB stored, 0.05 GB read.
  - The mechanism is the same cliff one level down. Store is write-through, so CPU writes equal the computed-prefill
    rate (~16k tok/s while thrashing), and a 349k-token LRU turns over in ~20 s, less than one attempt's reuse gap
    under load. Hits at 4 GiB come only in bursts (0.80 at 30-60 s, before the CPU pool first filled; 0.5 at 270-300 s).
  - Rule: CPU tokens ≥ the sum of all live attempts' contexts, with margin. Near that boundary it is bistable, like
    the GPU cache.
- **Offload does not raise the running count.**
  - Running stays about 15 because a request needs its whole context resident in GPU blocks; head-of-line blocking
    still holds (97% of steps).
  - So C=64 with 12 GiB gives the same throughput as C=32 (1,449 vs 1,473; 62 vs 63 attempts done) with 4.7× the TTFT
    (p50 14 s). Past the GPU's running capacity, concurrency only adds queueing (Little's law).
  - With L4-scale turn times and 30-60k contexts, a queue of ~40 behind a ~15-wide server reaches minutes, which is
    where a 900 s client timeout can trigger.
- **Preemptions rise with offload** (8-9/min vs 1-4). A likely cause, not verified: loads admit requests faster than
  prefills, so decode growth meets a full pool more often. They should be cheap, since the preempted request's blocks
  are already on CPU.
- **Host memory: the pinned buffer is rounded up to a power of two.**
  - `--kv-offloading-size 12` allocated **16.0 GiB** of pinned shared host memory (EngineCore `RssShmem` 16,799,592
    kB; system `Shmem` 16.9 GB). PyTorch's pinned host allocator rounds each `torch.zeros(..., pin_memory=True)` up to
    the next power of two.
  - 4 GiB, already a power of two, cost 4 GiB.
  - Size offload in powers of two (4/8/16/32 GiB), or budget the next power of two.
  - In tokens it is still the requested size: 12 GiB / 12,288 B = 1.05M tokens.
- **PCIe traffic is small next to the work it saves.** In the 12 GiB runs, 0.4 GB/s CPU→GPU (32k tok/s at 12 KB/token)
  replaces 32k tok/s of prefill, out of 37k prompt tok/s; only 3.6k tok/s is computed.

## Exp 3: admission control

**Design:** client-side gates on *new* attempts only; turns of running attempts are never held. There are 64 slots,
and `--gate-min 4` keeps at least 4 attempts active. The fixed caps (the cheapest gate) are the Exp 1 rows.

Local GPU use was stopped part-way through this experiment to protect the cards, so the hit-gate run is partial. For
a like-for-like comparison, every row here uses the window t = 60-160 s.

| config | active attempts (t ≥ 60 s) | gen tok/s | turns/min | GPU hit | running / waiting | KV | TTFT p50 / p90 / max (s) | turns with TTFT > 10 s |
|---|---|---|---|---|---|---|---|---|
| fixed C=16 (no gate) | 16 | **1132** | **136** | **0.70** | 10.2 / 0.2 | 0.64 | **0.11 / 0.73 / 2.6** | 0% |
| fixed C=32 | 32 | 663 | 83 | 0.03 | 14.7 / 13.7 | 0.94 | 10.7 / 14.1 / 15.4 | 67% |
| fixed C=64 | 64 | 884 | 114 | 0.04 | 19.6 / 39.5 | 0.96 | 23 / 31 / 35 | 100% |
| C=64, gate `kv_cache_usage_perc` < 0.70 | 45 | 746 | 94 | 0.05 | 16.2 / 24.6 | 0.94 | 17 / 25 / 27 | 99% |
| C=64, gate recent hit rate ≥ 0.80 (partial, to t=165 s) | 38 | 721 | 85 | 0.04 | 15.5 / 18.5 | 0.94 | 13 / 16 / 17 | 78% |

- **Both reactive gates let the system fall off the cliff and could not bring it back.**
  - In the first 30-60 s the signals still looked healthy: KV usage 0.31-0.55, because young attempts have small
    contexts, and the hit rate 0.4-0.5 over a 10-s window.
  - So the KV gate admitted 45 attempts and the hit gate 39 before either closed.
  - In the collapsed state both signals are pinned (KV 0.94, hit 0.04), whatever the active count. The gate stays
    closed, but it cannot evict attempts already inside, so the system stays collapsed. Active attempts drain only as
    they finish (the KV-gate run fell 45 → 28 by t = 300 s, and gen tok/s fell to ~450).
  - The collapse is bistable, so a gate must keep the system out of it. It cannot react once inside.
- **A fixed cap at the healthy point wins on every axis:** 1.5× the throughput of the reactive gates, hits 0.70 vs
  0.04, and TTFT p90 under 1 s vs 16-25 s.
- **Why the team's KV gate works on Kaggle anyway:** `ARC_KV_ADMIT_MAX` 0.6 holds KV ≤ ~0.72, and 0.75 is too late
  (AGENTS.md). There, contexts grow over hours, so the lag of a KV signal is small relative to the growth. Here
  attempts grow 3k → 22k in about a minute. The lesson transfers as a margin: set the KV gate low enough that the
  growth of the already-admitted attempts still fits.
- **Not run (local GPU use stopped to protect the cards):**
  - the projected-working-set gate: admit only while Σ over active attempts of max(current context, expected final
    context) ≤ ~1.5× pool. It is implemented as `loadgen.py --gate-ctx/--gate-ctx-est`, suite `exp3c`;
  - the hit gate with 4 GiB offload;
  - `--max-num-seqs 12` and `--no-scheduler-reserve-full-isl` at C=32.

  From the code (mechanism points 2 and 5), the expectations are:
  - max-num-seqs caps running, but the queued attempts' prefixes still live in the evictable cache, which is what
    thrashes. It should not help unless the cap is below the healthy running count.
  - reserve-full-isl off admits on the first chunk, which means more decode-time preemption (recompute) on top of the
    same eviction.

## Practical rules for agentic tool-use traffic

Notation: W = working set = Σ over active attempts of their current context (prompt + generated), in tokens. P = the
GPU KV pool in tokens.

1. **Size concurrency by W/P, not by KV usage.**
   - Keep W ≤ ~0.8 P with no offload. At 0.45 P the cache is perfect; at 0.9 P it is metastable (oscillating hits
     0.3-0.8); at 1.8 P it is collapsed, with hits ~0, throughput −40% and seconds-to-minutes of queueing.
   - Past ~1× there is no gentle degradation. The system flips into a state it does not leave while the load lasts.
   - W must include contexts late in the run: attempts grow, so a concurrency that is safe at hour 1 can be 2× over at
     hour 8. That is the Kaggle "binds after ~1 h".
2. **Gate on the solver's own token counts, with margin, before admission.**
   - Use projected W: each active attempt counts max(current context, expected final context), plus the new
     attempt's expected size.
   - Server aggregates (`kv_cache_usage_perc`, hit rate) lag growth and pin at the same values in the healthy-full and
     collapsed states, so they can only work with a large margin (the team's 0.6 KV gate).
   - Gate new attempts, never the turns of running ones: a held turn's prefix sits in the evictable pool and is
     evicted.
3. **Once collapsed, only lowering W helps.** Stop admitting until finishes bring W down. More slots only add
   queueing (C=64 vs C=32: TTFT p50 31 s vs 11 s at similar throughput). Raising max-num-seqs is not expected to help
   (not measured).
4. **CPU offload is worth it only if the CPU pool holds W.**
   - CPU tokens ≥ ~1.2 × W at the end of the run. In bytes: tokens × KV bytes/token, then rounded up to a power of two
     for host RAM (the PyTorch pinned allocator does this rounding).
   - Below that it is LRU thrash with write-through traffic: 4 GiB, 1.0× W, gave 0-12% hits and −5% to +7%
     throughput.
   - At 3× W it restored hits to the ideal (0.90) and gave 2.3× throughput at 1.8 P.
5. **With offload, GPU prefix hits of ~3% are normal.**
   - Judge the cache by (GPU + CPU hits) ÷ ideal reuse, from `prompt_tokens_by_source_total`.
   - CPU 90-98% with GPU ~3% means the offload is doing its job, and the GPU pool is the running-set bottleneck.
6. **Offload does not raise how many attempts run at once.**
   - Running ≈ P ÷ mean context (here about 15) with or without offload; head-of-line blocking stays at 97%.
   - Above that, more attempts add only queueing: C=64 vs C=32 on 12 GiB gave the same throughput with 4.7× TTFT.
   - Pick concurrency ≈ running capacity × (1 + think time ÷ turn time), and size client timeouts for queue length ÷
     running × turn time. A 900 s timeout fires when about 40 queued turns sit behind about 15 running ones, each
     taking minutes.
7. **Prefill, not preemption, is what you lose.** With `reserve_full_isl` on (the default), preemptions stayed at
   ≤ 4/min and recompute at 0%. The cost is re-prefilling evicted prefixes: computed prefill went 2k → 17k tok/s while
   generation fell. Do not switch reserve-full-isl off (expected: more preemption; not measured).
8. **Measure it right:**
   - hit rate from `prompt_tokens_by_source_total` (not `prefix_cache_queries`);
   - waiting split by `num_requests_waiting_by_reason` (`deferred` = loading from CPU);
   - evictable cache and recompute from a scheduler observer (`kvlab_sched.py` works on 0.20);
   - client TTFT tails.

   Leading indicators of the cliff: the waiting queue blocked in a growing fraction of steps, and block
   idle-before-evict falling toward the turn gap (36 s → 8 s from C=8 to C=16, against a 5-7 s reuse gap).

## Measurement pitfalls found

1. **`vllm:prefix_cache_hits / vllm:prefix_cache_queries` is not a hit rate under pressure.**
   - `get_computed_blocks` records a query every time the scheduler peeks at the head-of-line request, which is every
     step while it is blocked.
   - Queries per prompt token: 1.0× unloaded, 4.9× at C=16, 30-35× at C=32/64. At C=16 the counter ratio says 0.18
     while the real per-request hit rate is 0.68.
   - The connector's `external_prefix_cache_*` is recorded once per admission, after allocation, so the GPU and CPU
     numbers are not comparable.
   - Use `vllm:prompt_tokens_by_source_total{source=local_cache_hit|external_kv_transfer|local_compute}` over
     `vllm:prompt_tokens_total` (counted once, at the first prefill), or the client's
     `usage.prompt_tokens_details.cached_tokens` (`--enable-prompt-tokens-details`). The teammate's "GPU hits ~3%"
     is real in the offload regime (we also measure 3% per request), but the counter exaggerates the collapse near the
     edge.
2. **`kv_cache_usage_perc` counts only referenced blocks.** Evictable cache counts as free, so usage 0.65 (C=16)
   already costs 22 points of hits. Usage says "how full with running requests", not "how healthy the cache is".
3. **vLLM exports no recompute counter.**
   - `iteration_tokens_total` counts the full prompt at the first output, cache hits included.
   - `prompt_tokens_by_source` counts only the first prefill.
   - The preempted-request prefix stats (`preempted_queries/hits`) are collected but never exported.
   - Measure scheduled prefill tokens instead, e.g. with a `--scheduler-cls` subclass (`kvlab_sched.py`). It must
     subclass `AsyncScheduler` when async scheduling is on (the default); a plain `Scheduler` subclass is not the
     scheduler you measure in production.
4. **`num_requests_waiting` includes requests whose KV is loading from CPU** (`WAITING_FOR_REMOTE_KVS`, also shown as
   `num_requests_waiting_by_reason{reason="deferred"}`). They already hold GPU blocks.
5. **OffloadingConnector in 0.20 needs `--disable-hybrid-kv-cache-manager`.** The pinned CPU buffer rounds up to a
   power of two (12 GiB → 16 GiB).
6. **Shared-GPU locking:** `flock LOCK vllm serve` releases the lock when `flock` exits, even if vLLM's EngineCore child
   still holds the GPU. Kill your own process group and wait until no member is left before releasing. Also, flock is
   not FIFO: another agent can take the lock in a 3 s gap, so do not count lock wait toward a startup timeout.

## serve_replay_bench.py: first real-server smoke

The coordinator asked for a first run against a real server. The bench was run unmodified with `--mode synthetic`
against the lab server (upstream vLLM 0.20 on GPU 1, Qwen2.5-0.5B, 195k-token pool), using
`experiments/E067_vllm_lab/kvcache/bench_smoke.py`:

```
python evals/serve_replay_bench.py --base-url http://127.0.0.1:8067 \
  --traces experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz \
  --mode synthetic --synthetic-vocab 100:150000 --concurrency 8 --duration 240 --warmup-s 60 --think-time 2 \
  --seed 0 --server-log <server.log> --out <out>
```

**What works:**
- It loads 480 attempts / 8,703 turns from the tarball, with planned prompt lengths exact (`prompt_len_exact_frac`
  1.0).
- Requests are accepted, `max_tokens = min_tokens` is honoured (`completion_served` = requested), and `/metrics` is
  parsed.
- Windows, `requests.jsonl`, `summary.json` and `report.md` are written. Client and server gen tok/s agree
  (1,193.4 vs 1,193.3).
- After the server died, it stopped cleanly at `--max-errors 50`.

**Issues to fix in the bench** (reported here, not changed):
1. **The server-side `prefix_hit` and `uncached_tok_s` use `vllm:prefix_cache_{hits,queries}`**
   (`serve_replay_bench.py` ~L866-876 and L918). Under KV pressure those queries are re-counted every step for the
   head-of-line request (pitfall 1): queries reached 30-35× the prompt tokens in our C=32/64 runs. `uncached_tok_s =
   q − h − ext` would then be overstated about 30×, and `prefix_hit` understated.
   - In this smoke there was no waiting, so the numbers were still right (queries = prompt tokens).
   - Fix: hit = Δ(`prompt_tokens_by_source_total{local_cache_hit}` + `{external_kv_transfer}`) /
     Δ`prompt_tokens_total`; uncached = Δ`prompt_tokens_by_source_total{local_compute}`. These exist in vLLM 0.20; the
     fork needs a check.
   - The client-side `client_cached_frac` (from usage) is fine.
2. **Synthetic mode loses the cross-task shared prefix.** Turn-1 ids are seeded per task (`first_digest`), so the two
   attempts of a task share turn 1, as in the real run. But the system prompt + tool block shared by *all* tasks
   becomes random per task. Turn 1 of the first attempt of each task gets 0 cached tokens (`cached_served` null)
   against 512 recorded. It is small (about 512 tokens per attempt). A fixed common prefix of the recorded shared
   length would close it.
3. **`cached_served` is null instead of 0 when vLLM omits `prompt_tokens_details`** (0 cached). Harmless, but averages
   over the field should treat null as 0.
4. **The upstream server prints no `GPU memory:` lines**, so headroom is `None`. As documented; not a bug.

**A crash in my setup, not the bench:**
- Try 1 served the 32k-native Qwen-0.5B at `--max-model-len 131072` with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- At t ≈ 90 s the engine died with `torch.AcceleratorError: CUDA error: device-side assert triggered` (in
  `gpu_model_runner.get_output`), presumably the first sequence past 32,768 positions. The traces' max context is p50
  49.5k, p90 125.8k.
- The bench then logged 500s and refused connections and stopped at 50 failed attempts: correct behaviour.
- The retry would use YaRN ×4 instead (`--hf-overrides` rope_scaling, already in `bench_smoke.py`). It was not run
  (local GPU use stopped to protect the cards).
- Next steps: re-run the smoke on a pod with a model whose native context covers the traces, and run an A/A pair, as
  the bench doc says.

## Files

- `experiments/E067_vllm_lab/kvcache/loadgen.py`: multi-turn load generator with optional KV or cache-health gates.
- `experiments/E067_vllm_lab/kvcache/kvlab_sched.py`: instrumented scheduler (observer only).
- `experiments/E067_vllm_lab/kvcache/run_suite.py`: one server per run under the GPU-1 lock.
- `experiments/E067_vllm_lab/kvcache/analyze.py`: the tables and series (`analyze.py RUN... [--series 30]`).
- `experiments/E067_vllm_lab/kvcache/bench_smoke.py`: the replay-bench smoke test.
- `experiments/E067_vllm_lab/kvcache/runs/`: per-run turns, scheduler logs and server logs.
- The 1 Hz `/metrics` samples (`*.metrics.jsonl`, over 1 MB each) and a full copy of `runs/` are in the private HF
  dataset `livobereshka/arc-agi2-vllm-serving-lab` under `kvcache/runs/`. The restore command is in
  `experiments/E067_vllm_lab/kvcache/README.md`.
