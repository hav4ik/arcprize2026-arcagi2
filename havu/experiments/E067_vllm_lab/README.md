# E067: our own vLLM serving knowledge

Started 2026-10-30. This is the hub for what we know about serving DeepSeek-V4.1 with vLLM on Kaggle's 4xL4, and for
the tools we built to measure it. It covers memory, failure modes, where the speed comes from, a replay benchmark and
a reliability kit. Each study has its own note or README; this page summarizes them and links to them.
Unless a source says otherwise, numbers are measured; **est.** marks estimates. Labs run on the dev box's 2x RTX 3090;
pod numbers come from rented 4xL4 pods (L4 at 72 W, while Kaggle may run at 50 W).

## Goal

Choose, tune and qualify the Kaggle serving stack from code and cheap measurements, not from daily submissions. The
stack is Manh's vLLM fork (`nguyen599/vllm`) with DSv4.1 E128 2-bit at TP4. Three questions drive the work:
1. Why did B32-M (LB 46.5, replica ~77.9) fail?
2. Which knobs keep a 12 h run alive?
3. Which features make the fork fast, and at what risk?

The answers should be understood well enough that Manh and the rest of the team can reason about a change before
spending a pod day or a Kaggle slot on it.

## Index

| study / tool | status | where |
|---|---|---|
| GPU memory lab: startup profile, KV sizing, allocator cap, stranded memory, OOMs (vLLM 0.20, Qwen2 7B, one 3090) | done | [note](../../notes/2026-10-30_vllm_lab_memory.md), [scripts + tables](memory/), raw logs on HF `memory/logs/` |
| B32-M OOM modes (summary page) | done | [note](../../notes/2026-10-30_b32m_oom_modes.md) |
| Failure paths: how each failure travels worker → EngineCore → API → solver → supervisor (fork v11, code reading) | done | [note](../../notes/2026-10-30_vllm_failure_paths.md) |
| Acceleration map: where the fork's speed comes from, plus the ablation plan (code reading + pod logs) | done; ablation not run | [note](../../notes/2026-10-30_vllm_fork_acceleration_map.md) |
| Serve replay bench: replays recorded harness attempts token-exactly at fixed concurrency (~20 min per config) | built, validated offline; not yet run on a real server | [`evals/serve_replay_bench.py`](../../evals/serve_replay_bench.py), [tests](../../evals/tests/test_serve_replay_bench.py), [note](../../notes/2026-10-30_serve_replay_bench.md), [token cache](replay/) (HF) |
| Reliability kit + qualification protocol: notebook runner, sampler, watcher, report, GPU hold, fault injection | built, unit-tested | [`kaggle/reliability/README.md`](../../kaggle/reliability/README.md) |
| Safety edits proposed for Manh's notebook (tries, step timeout, util, wheel pin) | done | [`kaggle/dsv41_tools/MANH_SAFETY_EDITS.md`](../../kaggle/dsv41_tools/MANH_SAFETY_EDITS.md) |
| All-reduce lab: NCCL settings, host all-reduce prototypes, TP1/TP2 decode over PCIe without P2P (vLLM 0.20, 2x 3090) | done | [note](../../notes/2026-10-30_vllm_lab_comm.md), [scripts + results](comm/), logs on HF `comm/results/` |
| KV pressure lab: multi-turn agentic load, admission gate, preemption and head-of-line counters | done (stopped early to protect the local GPUs; projected-context gate, max-num-seqs and gate+offload runs not done) | [note](../../notes/2026-10-30_vllm_lab_kvcache.md), [scripts + small logs](kvcache/), raw runs on HF `kvcache/runs/` |
| Illegal-memory-access risk: are V12 and B32-M exposed to the CUDA IMA Manh hit on Kaggle at 120 concurrent requests? (code reading + issue search) | done; pod repro not run | [note](../../notes/2026-10-30_illegal_memory_access_risk.md) |
| Replica vs benchmark pass@1 (paired on the same tasks) | done for the first hours: V12 −2.5 [−8.1, +3.2], B32-M −2.4 [−9.0, +3.4]; rerun at the end | [note](../../notes/2026-10-30_replica_vs_benchmark_pass1.md), [script](../E062_manh_vllm_fork/per_layer/scripts/replica_vs_bench.py) |
| 12 h rerun-mode replica of Manh's V12 notebook (pod 2) | **running** | [E062 phase_h/manh_v12](../E062_manh_vllm_fork/per_layer/phase_h/manh_v12/) |
| 12 h rerun-mode replica of B32-M V3 with Kaggle-like VRAM (pod 3) | **running**; CUDA OOM at 2.11 h, restart 1 at 2.25 h | [E062 pod3_runs/b32m_v3](../E062_manh_vllm_fork/per_layer/pod3_runs/b32m_v3/), [OOM modes](../../notes/2026-10-30_b32m_oom_modes.md), [restart sim](../E062_manh_vllm_fork/per_layer/scripts/oom_restart_sim.py) |

## Key findings so far

### KV pressure under agentic multi-turn load (local lab, Qwen2.5-0.5B, vLLM 0.20)
- **A cliff, not a slope:** when the live attempts' total context goes from 0.9x to 1.8x the KV pool, real prefix hits fall
  0.68 → 0.04, turns/min 131 → 79, and TTFT 0.1 s → 11 s. It doesn't recover while the load lasts.
- **The cause is admission plus eviction, not preemption.** Between turns an attempt's context exists only as evictable cache,
  which gets reused before the next turn. Strict FCFS blocks the head of the queue in 97% of scheduler steps.
- **CPU offload helps only if the CPU tier holds the whole working set:** 12 GiB brought total hits back to 0.90 (2.3x
  throughput); 4 GiB thrashed. GPU hits around 3% are normal with offload.
- **Gates on server signals (KV use, hit rate) over-admit, then stay collapsed.** A fixed concurrency cap beat both (1.5x
  throughput, 20x lower TTFT p90). Gate on the solver's own projected context instead (not yet tested).
- **Measurement trap:** vLLM's `prefix_cache_hits/queries` counters overstate queries 5-35x under pressure. Use
  `prompt_tokens_by_source_total`. `serve_replay_bench.py`'s server-side hit rate uses the inflated counter and needs
  this fix.

### Memory: how the KV pool is sized and why the margin runs out
Sources: [memory lab](../../notes/2026-10-30_vllm_lab_memory.md),
[B32-M OOM modes (summary page)](../../notes/2026-10-30_b32m_oom_modes.md), failure paths section 1, acceleration map
E1-E2.
- **How the pool is sized.** `KV = util x total - weights - peak_act - non_torch - cudagraph_est`.
  - `peak_act` comes from one eager dummy step at `max_num_batched_tokens` with `attn_metadata=None`, so **attention
    kernels and their workspaces are not profiled**.
  - Each +0.01 util adds 0.01 x total to the KV pool and nothing else: 0.236 GiB on a 3090, 0.225 GiB per L4.
- **Upstream never caps the PyTorch allocator.** Util only sizes the pool; the allocator can grow into the `1 - util`
  remainder.
- **The fork caps it after warmup.** `VLLM_GPU_DRIVER_RESERVE_MB` sets the cap to `reserved + free - reserve` (reserve
  256 MiB in wheel v11, 512 in B32-M). Two consequences:
  - **The cap does not depend on util.** A higher util only moves memory from the runtime margin into KV.
  - Every later working buffer has to fit under that fixed line.
- **Stranded memory OOMs before zero.** Lab case `cap_u95_b8192`: an OOM on a 296 MiB request with 0.39 GiB free by
  allocated bytes.
  - 0.53 GiB sat in 67 partly used split segments, which the allocator's retry cannot release.
  - With native segments, fragmentation was ~70% of the largest transient (0.74 GiB at a 1.03 GiB transient).
  - B32-M's signature is the same: 0.11 GiB left to the cap, 0.26 GiB stranded. Hence the **danger band starts at
    ~0.26 GiB** of cap - peak.
- **Expandable segments.**
  - Switched on after warmup (as the fork does, `46a2cb33de`): no help, because the startup segments' holes stay.
  - On from launch: they survived util 0.95, at the cost of 2,775 allocator retries and 11-19% slower long prefills.
    At util 0.96 they OOMed anyway.
  - On the fork they cannot be set at launch: upstream refuses to start with a KV connector while they are set
    (acceleration map E1).
- **Other levers.**
  - Smaller chunks alone do not add margin: with a profile-sized pool, KV grows into the freed space.
  - A cold torch.compile cache inflated the profiled `peak_act` by 0.78 GiB (mbt 2048), and a warm one does not. Pod and
    Kaggle cache states change the pool.
- **Why peak memory grows with context only on DSv4.1.**
  - On a dense FlashAttention model the per-step transient depends only on the step's token count (~0.126 GiB per 1k
    tokens for the 7B). It is identical at 2k and 61k context, and reserved memory jumps once in the first minute and
    then stays flat.
  - DSv4.1 has context-sized per-step tensors that the profile misses: the sparse-attention indexer's prefill logits
    (`M x N x 4` bytes per chunk, chunked to 512 MiB), its K gathers, and decode logits of `rows x max_model_len`
    outside the CUDA-graph pools. Data-dependent MoE padding adds to them.
  - Agent transcripts lengthen over the run, so the worst steps arrive hours in: a running maximum over a heavy tail,
    not a leak (allocated memory stays flat).
  - B32-M's wheel `5e1bccd07c` lacks the profile fixes `7ab756940a` and `f506afa210`, which are in wheels v10 and v11.
- **Pods have more GPU headroom than Kaggle.**
  - CUDA graphs took 0.65 GiB on Kaggle vs 0.34 GiB on pods.
  - With B32-M, Kaggle's cap - reserved was 0.67-0.73 GiB, against 1.16-1.22 on pods.
- **Replica numbers.**
  - B32-M V3 replica (Kaggle-like VRAM): cap - peak went 0.95 → 0.55 GiB in the first 10 min. The first allocator
    retry came at 1.26 h, and a retry storm at 1.7 h (9 → 23 retries in 10 min) with 0.44 GiB left. The projection
    reaches the danger band at ~4.2 h; on Kaggle the run stopped answering at ~4.85 h.
  - Manh V12 replica (wheel with the fixes, util 0.91): cap - peak 2.00 → 1.58 GiB in 3.5 h, 0 retries.
  - Pod 3 C1 run: ~1.3 GiB to the cap at 3.4 h, reserved memory creeping ~0.1 GiB per 30 min.

### KV sizing on 4xL4
- **Per-token KV.** The fork's KV footprint per token is **1.64x B32's**: fp8 + bf16-RoPE compressed rows and an FP8
  indexer, against B32's FP4 rows and MXFP4 indexer. Expert pad 64,64 (−1.26 GiB/GPU) buys most of it back.
- **Pool sizes on pods:**
  - B32: 1.56M tokens;
  - Manh production, no DCP: 1.45-1.62M depending on util and model;
  - DCP4 builds: 3.06M.
- **Free KV levers not yet tried:**
  - `VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES=0`: the default estimated 0.51 GiB of graphs per rank against 0.18 GiB
    actual, so est. +0.3 GiB per rank;
  - `VLLM_DSV41_INDEXER_TP_SHARD=2`: +13% pool per Manh's commit, but it loses the decode row shard.
- **An inert flag:** `--prefix-cache-retention-interval 128` does nothing on the fork. SWA is not prefix-cached under
  bounded replay.

### Any step error kills the engine
Source: [failure paths](../../notes/2026-10-30_vllm_failure_paths.md).
- **No recovery.** vLLM 0.32-dev has no per-request recovery from any step error, CUDA OOM included. Fault tolerance is
  off by default and exists for DP scale-down. One OOM on one rank costs every in-flight attempt (up to 64) plus a
  restart (116-174 s outage on pods; 3-25 min reported).
- **What the client sees:**
  - open streams get an in-stream SSE error, "The server had an error while processing your request." (HTTP status
    already 200);
  - new requests get HTTP 500;
  - then connections are refused. The API server exits ~13 s after a worker kill.
  - The body never says `EngineDeadError`, so Manh's solver `11bfaaaa` spends a try on each in-flight attempt. Ours
    (`a9983f4c`) re-queues for free.
- **Ranks 1-3 fail late.** EngineCore reads the worker replies in rank order, so a failure on rank 1-3 is noticed
  only at the execute deadline (300 s default, 900 s ours) or at the 600 s host all-reduce trap. That gives 5-15 min
  with no generation.
- **Latent risk without a KV connector:** rank 1-3 exceptions are then only logged (a hang, or est. garbage tokens).

### Admission and hang gaps
- **Watermark.** `--watermark 0.1` holds back ~145k tokens of a 1.45M pool.
  - At reported KV usage >= ~0.90 no waiting request is admitted, FCFS with head-of-line blocking; a 100k follow-up
    turn needs usage <= ~0.83.
  - Pod 3: a turn waited > 900 s with 0 preemptions, hit the client read timeout, and its attempt restarted from turn
    1.
  - SSE keep-alive is off, so a queued request sends no bytes.
- **Hang timers:**
  - the execute RPC deadline: 300 s default, 900 s ours;
  - host all-reduce / all-gather GPU traps: 600 s, hard-coded;
  - PyNccl collectives and the host reducer threads: none.
- **Hangs nothing detects:**
  - **A hang outside any RPC** has no vLLM timer. Manh's `--supervise` restarts only on process exit, so such a hang
    costs the rest of the run. Our cell 2s supervisor restarts after 1500 s without generated tokens.
  - **SimpleCPUOffload's copy thread** has no try/except. If it dies, loads never complete and KV leaks silently
    (inferred).

### Where the fork's ~1.5x per-stream decode speed comes from
Source: [acceleration map](../../notes/2026-10-30_vllm_fork_acceleration_map.md). Pods only; fits are rough (est.).
- **Size of the gain.** At matched load, Manh's production (TP4, no DCP, wheel v11) decodes **1.5-1.6x faster per
  stream than R1** and ~1.7x faster than the B32 replica.
  - The fitted decode step at 56 streams is 68-72 ms, against 98-104 ms.
  - **The fixed per-step cost halves (~65 → 33 ms), while the per-stream slope stays 0.6-0.7 ms.** So the gain is in
    once-per-step costs (communication, weight reads, sampling, gathers), not in per-sequence attention.
- **Ranked causes:**
  1. Host all-reduce MODE 556 + L2 weight prefetch + host all-gather replace every NCCL call in a decode step. est.
     −8 to −15 ms/step; medium confidence. E062b measured +9.8% per stream with the row shard. This is the first
     ablation.
  2. Prefill ~2-3x faster (upstream SWA bounded replay, AR overlap +11.8%, L4 MoE tiles), so mixed steps stall
     decodes for less time. est. +5-10% per stream.
  3. The fork's SM89 kernel base (FP8 indexer, Triton sparse MLA). est. −5 to −15 ms/step; low-medium confidence; no
     A/B exists.
  4. Decode indexer row shard: −5.1 ms/step at B = 64 (commit).
  5. Scheduling under KV pressure: watermark + SimpleCPUOffload give 0 preemptions, against B32's 100-275 per 5 min.
     est. +10-15% per stream, in the KV-bound hours only.
  6. Expert pad 64,64: +3-4.5% decode (high confidence, measured on B32's stack).
- **Budget check:** the parts sum to est. −21 to −44 ms/step, against −30 to −36 ms fitted.
- **DCP4 builds** sit at B32's per-step cost and won only through the 2x pool. On Kaggle they ran at 0.70x of pod
  speed.

### Communication over PCIe without P2P
Source: [comm lab](../../notes/2026-10-30_vllm_lab_comm.md) (2x 3090, vLLM 0.20, a Qwen2 7B at TP2).
- **The local box looks like Kaggle at the transport level.** The 3090s have no P2P, so NCCL goes through pinned host
  memory (SHM), and vLLM's custom all-reduce switches itself off.
- **NCCL's defaults are the best NCCL setting at every size.** They pick LL below ~128 KiB and Simple above.
  - In CUDA graphs a 2-rank all-reduce costs 13.4 µs at 8 KiB, 63 µs at 256 KiB and 172 µs at 1 MiB. Bus bandwidth
    levels off at 6.7-6.9 GB/s.
  - Forcing `NCCL_PROTO=LL` is 2.4-4.7x slower from 256 KiB, and `LL128` 1.7-4x slower at every size.
  - Without SHM, NCCL falls back to sockets: 6-20x slower per call, 4.2x slower decode at 32 streams.
  - Eager calls have a ~30 µs CPU floor, so collectives must be graph-captured.
- **A graph-capturable host all-reduce ties NCCL at TP2.** The test kernel is `har_ext.cu`, Manh's design without
  packing, SIMD or prefetch.
  - Per call: 13.7 vs 13.4 µs at 8 KiB, 41 vs 46 µs at 128 KiB; it loses 1.2-2x from 256 KiB.
  - End to end: 1,881 vs 1,906 tok/s at 32 streams, with identical greedy tokens.
  - Its advantage can only appear at TP4. There the ring moves 1.5x the bytes per GPU in 6 dependent hops, against 1x
    in 2 hops for the host all-reduce.
- **CPU reducer threads must stay warm.** Reducers that sleep between polls add ~90 µs per call back to back and 20-90
  µs with compute gaps, which costs 9-20% of decode end to end.
- **Comm explains part of the ~1.7x, not all.**
  - A Kaggle decode step has ~88 TP all-reduces. est. NCCL is ~15-35% of the step (~6 / 12 / 23 ms per step at 1 / 24
    / 64 streams), so even free communication gives only ~1.2-1.5x.
  - The rest has to come from the kernel base, prefill speed and the pool size.
  - A same-pod A/B of MODE 556 vs NCCL (no DCP) is still needed.

### Illegal memory access (IMA) risk
Source: [IMA note](../../notes/2026-10-30_illegal_memory_access_risk.md). Code reading and a public-issue search; we
have no logs of Manh's crash.
- **No shape bug.** No path we read goes out of bounds merely from 120 decode rows, a 128k context or images: loads
  are masked, offsets are int64, buffers are size-checked.
- **The real gap is three unbounded index sinks, in both wheels** (v11 and B32-M's 5e1b):
  - the worst is the top-k → slot gather `_compute_global_topk_indices_and_lens_kernel`
    (`deepseek_v41/common/ops/cache_utils.py:895-955`). It checks only `local_idx >= 0`. Public fix: #55636, unmerged;
  - Triton sparse MLA's `_load_kv_tile` accepts any slot >= 0;
  - the paged logits kernel does not check `page < num_blocks`.
- **Top hypothesis (H1, est. 30%):** NaN logits in a prefill indexer reach `top_k_per_row_prefill`, which has no NaN
  handling (public #49896, open). It then hands garbage indices to those sinks. The next hypothesis is a race on an
  index buffer, such as SimpleCPUOffload's unfenced load stream (est. 20%).
- **Exposure:**
  - Manh's V12 runs the same kernels, sinks and offload path, so it is exposed, at a lower rate than his crash run (est.;
    text only, 64 rows, no top_k).
  - Our ~14 h of clean no-DCP serving (~30M tokens, est.) cannot rule out a per-run risk of ~10-30% (est.).
  - B32-M has low exposure.
  - An IMA kills the engine exactly like an OOM.
- **Cheap guards** (listed in the note, none built): bound the slot gather and `_load_kv_tile` with one mask each, and
  map NaN logits to −inf in the logits kernel's store before the prefill top-k.

### Replay bench and reliability kit
- **Replay bench.** Tokens mode reproduces the recorded prompt lengths on **8,703 of 8,703 turns** of the b32m_x2t set
  (480 attempts). Prep takes 4.4 min on CPU, and the client uses ~0.4 cores at concurrency 64.
  - Forced output lengths at temperature 0 flatter MTP and may shift MoE load.
  - Compare arms with each other, not with the leaderboard.
- **Reliability kit.** It qualifies a candidate on a pod in four steps: a commit smoke, a GPU hold calibrated to
  Kaggle's cap - reserved, a 12 h rerun replica, and a separate fault-injection run. Alerts are tied to the numbers
  above (`cap_peak` WARN < 0.5 / ALERT < 0.26 GiB; any allocator retry is an ALERT).

## Practical rules for Kaggle 4xL4 candidates

1. **Pin the KV pool with `--kv-cache-memory-bytes`.**
   - Size it from Kaggle's own startup numbers, so pods, cold or warm compile caches and Kaggle all get the same pool
     and margin. Then util only checks that it fits.
   - First check that the fork honours the flag (open question 2).
2. **Spare-room rule.** On the worst GPU after warmup, keep cap - peak >= stranded + 0.3 GiB, ideally >= 0.75 GiB.
   - Take "stranded" from the `split segments` figure late in a replica (0.26 GiB on B32-M).
   - The margin must hold at the run's end, not just in hour 1: read the slope.
   - For B32-M's settings, util ~0.02 lower (~0.45 GiB/GPU, ~13% of its pool) would have been enough (est.).
3. **Only ship wheels whose startup profile covers long-context steps:** the indexer logits, gathers, decode logits at
   `max_num_seqs` rows, Engram (`7ab756940a`, `f506afa210`). Verify on a long-context replay that the profiled peak is
   at least the replay's peak.
4. **Treat the first allocator retry as an alarm, not noise.**
   - Retries mean the allocator is already flushing its cache to survive.
   - Log `VLLM_GPU_MEMORY_LOG_INTERVAL=60` and track per rank: peak, reserved, split bytes, retries and cap - peak
     (`kaggle/reliability/watcher.py`).
   - nvidia-smi free memory is not a pressure signal.
5. **Replicate with Kaggle's headroom.** Pods have ~0.5 GiB/GPU more. Calibrate `gpu_hold.py` per stack until every
   rank's cap - reserved is within ~0.02 GiB of Kaggle's.
6. **Make a death cheap (recovery settings):**
   - a supervisor that restarts on exit **and** on a stall (cell 2s: 1500 s without generated tokens, at most 2
     restarts, none within 60 min of stop-new);
   - a solver that re-queues for free on the 500 text and `APITimeoutError` (`a9983f4c`). With Manh's solver, use
     `ARC_JOB_TRIES=3` and `ARC_CONN_TRIES=12`;
   - `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`, **only together with** the tries fix (otherwise one hang can cost every
     in-flight attempt);
   - keep the KV connector on, and pin the wheels dataset version;
   - each of these is in [MANH_SAFETY_EDITS.md](../../kaggle/dsv41_tools/MANH_SAFETY_EDITS.md).
7. **Change one serving flag at a time,** measured on the replay bench. Run an A/A pair first: the stack is not
   run-to-run deterministic at the server level.

## Open questions and next experiments

1. **Ablation plan** (acceleration map, "Ablation plan"). Leave-one-out from production with the KV pool pinned, on a
   user-rented 4xL4 pod.
   - Tier 1 arms: P (x2), −AR, −PF, −ROW, −BSS, B32; ≈ 6.5-9 h.
   - Tier 2: micro workloads, ≈ 4 h.
   - Tier 3: KV levers, ≈ 1.5 h.
   - Then P vs −AR vs −PF at a 50 W power limit for Kaggle transfer, est. ~1 h.
   - The replay bench can serve as the shape-exact workload.
2. **Fork check of `--kv-cache-memory-bytes`.** Does the fork honour it with DSv4.1's several KV groups (compressed
   MLA, indexer, SWA), the SimpleCPUOffload connector and DCP? Does it give the same pool on a pod and on Kaggle? The
   ablation plan uses `--num-gpu-blocks-override` meanwhile.
3. **IMA repro and guards** (IMA note, sections 4-6).
   - Ask Manh for every rank's traceback, the scheduler dump and his launch config.
   - On a pod, run arm R0 (his run: 120 rows, images, top_k 64, 4 h) and arm R1 (V12 under `serve_replay_bench.py
     --order heavy-first`), with CUDA core dumps on.
   - Fuzz the three sinks and the prefill top-k with `compute-sanitizer` on one GPU.
   - Build the bound + NaN guards as an overlay.
4. **Watermark A/B:** 0.1 vs 0.02-0.03 (or 0), comparing queue wait and running count against preemptions. Also SSE
   keep-alive 60 s, or a client read timeout of 1800 s.
5. **The free KV levers:** `VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES=0` and `VLLM_DSV41_INDEXER_TP_SHARD=2`.
6. **First real-server runs of the replay bench:** an A/A pair, then A/B/A.
7. **Small fork patches to propose to Manh:**
   - a try/except in SimpleCPUOffload's `_copy_loop`;
   - polling all worker response queues, not in rank order;
   - a stall rule in `--supervise`;
   - the Engram pinned-memory fallback (+22 GiB host, est.).
8. **Kaggle unknowns:** cgroup `memory.oom.group`, the /dev/shm size, the 50 W vs 72 W power cap, and whether
   `nvidia-smi --query-compute-apps` behaves as on pods.
9. **Comm on the target:** a same-pod A/B of MODE 556 vs NCCL without DCP, with and without `PREFETCH_BLOCKS` (comm
   lab section 5).
10. Results of the running kvcache lab, the two 12 h replicas, and the replica-vs-benchmark pass@1 note.

## Data on Hugging Face

Large raw files go to the private dataset **`livobereshka/arc-agi2-vllm-serving-lab`** (team only; never make it
public). Code, small tables and notes stay in git.

| HF path | what | produced by |
|---|---|---|
| `memory/logs/` (8.5 MB, 50 files) | the memory lab's raw logs: server logs (DEBUG), per-step memory JSONL, per-request JSONL, nvidia-smi snapshots, batch summaries | `memory/run_case.sh`, `drive.py`, `hook/sitecustomize.py`; LAN IP redacted; compile caches and pid files left out |
| `comm/results/*.log` (0.3 MB, 25 files) | the all-reduce lab's console logs, with `NCCL_DEBUG=INFO`; git skips `*.log`, so they live only here (the `.json` results are in git) | `comm/scripts/`; hostname, LAN and docker IPs redacted |
| `replay/b32m_x2t.tokens.json.gz` (4.4 MB) + `tools.json`, `prepare.json` | the replay bench's prepared token cache for b32m_x2t | `evals/serve_replay_bench.py --prepare-only --local-render`; download in [replay/README.md](replay/README.md) |

| `kvcache/runs/` | the KV pressure lab's raw runs (per-request and per-window logs, server logs) | `kvcache/loadgen.py` and run scripts; see [kvcache/](kvcache/) |
