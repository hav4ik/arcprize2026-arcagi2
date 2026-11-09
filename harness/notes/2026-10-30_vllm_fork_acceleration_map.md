# Code map: where Manh's vLLM fork gets its speed (DeepSeek-V4.1 E128 on 4x L4)

Written 2026-10-30 from local reading only: the git clone, his notebook cell 2m, his launcher, and our pod logs. Nothing
ran on Kaggle, no pod was touched, and nothing was pushed. **Estimates are marked "est."**; every other number is
measured, with its source given.

**Scope:**
- His production server: `MANH_SERVER_ENV` plus the `dsv41_vllm.py` flags from cell 2m of
  `kaggle/dsv41_tools/out/manh_v12_363638159/dsv41-tools-submit-b32m-external-server.ipynb`.
- On our side: why it decodes ~1.5-1.7x faster per stream than B32's stack (the yhfgyyf SM89 fork,
  `0.28.1rc1.dev517+glm53.dsv41.vision11`, plus the 16-file overlay).

**Revisions** (clone `scratchpad/e062_port/manh_vllm`):
- wheel v10 = `f506afa210`;
- wheel v11 = `8cf0c0664b`, which is `71dca8bc24` plus our per-layer-experts PR;
- B32-M's wheel = `5e1bccd07c`.
- The clone is shallow at `b9ad670a67`, so fork features older than that are only in that commit's tree.
- The fork-only delta is `git diff 34091a07c7 71dca8bc24`: upstream main `34091a07c7` was merged in `570ee8a97b`. That
  is 208 files, about +27k lines, and much of it is Qwen/GDN code that DeepSeek-V4.1 does not use.

**Hardware recap:**
- Kaggle 4x L4 (SM89, 24 GB, 50 W in the scored rerun), PCIe Gen3 x16, no usable P2P.
- NCCL uses SHM there. A bf16 [2048, 5120] all-reduce takes 6.4 ms and [24, 5120] 0.14 ms (E027).
- Pods: 72 W and Gen4 x8. Pod 1's NCCL is ~2x faster than Kaggle's.
- E128: 40 layers, hidden 5120, and 128 routed experts (top-6) plus 1 shared expert.
  - TP4 shards: w13 N = 1152, w2 K = 576.
  - KV source layers are 2, 8, 14 and 20. Indexer layers are 2, 8, 14, 20, 24, 28, 32 and 36. The candidate source
    layer is 20 (2048 blocks x 8).
  - Engram layers are 1 and 14.

## Summary table

Gains are from L4 pods (72 W) unless marked. "Commit" means Manh's commit message, not reproduced by us. Section
letters point to the detail below.

| § | feature (knob) | mechanism | measured / est. gain | lossless? | main risk | knob |
|---|---|---|---|---|---|---|
| A1 | host all-reduce, MODE 556 | GPU ↔ pinned /dev/shm zero-copy; CPU threads sum fp32 in rank order; pipelined 16 KiB units; lossless packing (4), AVX-512 (8) | **+9.8% decode/stream, +6.2% e2e** (E062b, with B2, under DCP4); est. 10-20% of the decode step vs NCCL | **yes (556)**; Q8 bit 16 lossy (−7.5 pp, one draw) | 600 s GPU trap on a dead peer; up to 24 busy-spinning CPU threads; ~361 MiB shm | `VLLM_HOST_ALLREDUCE`, `_MODE` |
| A1 | ↳ L2 weight prefetch (16 blocks, 36 MiB) | extra blocks load the next GEMM's weights into L2 while the AR waits | not isolated (est. a large share of A1) | yes (loads only) | none known; PF_STOP (32) bounds it | `_PREFETCH_BLOCKS`, `_PREFETCH_MODE_FILE` |
| A1 | ↳ keep-warm (bit 512) | idle reducers keep AVX-512 units powered | 172 → 146 µs/call at 64 tok (Q8 modes), +6.5 ± 2.5 tok/s (commit) | yes | silent no-op without avx512bw | MODE 556 vs 44 |
| A2 | host TP all-gather | one-hop host gather ≤ 256 KiB, no CPU | 30.5 vs 55.7 µs at 64x128 (commit); est. 2-5 ms/step | yes (bytes) | 600 s trap | `VLLM_TP_HOST_GATHER` |
| A3 | AR overlap chunks = 2 | prefill layers split in 2 row chunks; ARs on a side stream under compute | **prefill +11.8%** (commit), decode flat; KV +2.0% after f506 | near-bitwise (tile choice) | silent fallback | `VLLM_DSV41_AR_OVERLAP_CHUNKS` |
| B1 | sparse indexer logits | consumers 24-36 score only layer 20's 16,384 candidate positions | none measured; est. < 1-3 ms/step, only past 16k-token contexts | same selection (ties) | none | `VLLM_TRITON_SPARSE_INDEXER_LOGITS` |
| B2 | decode indexer row shard | each rank scores 1/4 of the replicated indexer's decode rows, then gathers the top-k | **−5.1 ms/step at B = 64, +2.8% serving** (commit) | value-exact (ties) | silent fallback; conflicts with B3 | `VLLM_DSA_DECODE_INDEXER_ROW_SHARD` |
| B3 | indexer K TP shard (**off**) | shard indexer caches by position | **KV +8% / +13%** (commit) | exact | loses B2 | `VLLM_DSV41_INDEXER_TP_SHARD` |
| B4 | Triton sparse MLA + FP8 caches (auto) | SM89 Triton kernels; 584 B fp8_ds_mla rows; FP8 indexer | no kernel A/B; **KV bytes/token 1.64x B32's** | not vs B32 (finer quant); accuracy tie | capacity cost | none |
| B5 | prefill workspace factor 2 | smaller indexer prefill gather workspace | est. +0.5 GB KV/rank at 164k | yes | untested 160k-prompt edge | `--attention-config.sparse_mla_prefill_workspace_factor` |
| C1 | Humming pad 64,64 | no zero padding of 1152 / 576 expert shards | **−1.26 GiB/GPU, KV +60%, decode +3-4.5%** (R1) | no (K order); accuracy −0.1 pp (tie) | none | `VLLM_HUMMING_MOE_PAD_MULTIPLE` |
| C2 | Humming MoE tuning | L4 tiles for steps ≥ ~321 tokens | GEMMs 1.0-1.35x (commit) | no (tiles); error unchanged | L4-only configs | `VLLM_HUMMING_MOE_TUNING_CONFIG` |
| D1 | mHC token blocks (v11) | fused post+pre shares fn loads across token blocks, T ≤ 32 | 42 → 28 µs/call at 32 tok (commit); est. ≤ 1.1 ms/step at ≤ 32 running | bit-exact | none | wheel v10 vs v11 |
| D2 | Engram host tables + THP | 64 GiB n-gram tables in pinned host memory read by UVA; huge pages | enables the model; startup 47 s → 1,235 s worst case | lookup exact | startup under fragmentation | `--engram-config.use_thp` |
| D3 | embed_tokens UVA | embedding shard in pinned host memory | +0.31 GiB/rank GPU | yes | none | `--cpu-offload-params` |
| D4 | RoPE cap, page-cache drop | cos/sin at max_model_len; drop gptq/ page cache | **+448 MiB/rank, KV +10.8%** (commit); file cache 34 → 8.8 GiB | bit-identical | none | none |
| E1 | driver reserve 256 MiB + cap + expandable | allocator capped at startup free − 256 MiB; expandable segments after | stability (fixes a 70-min OOM, commit) | yes | reserved creep ~0.2 GiB/h → retries | `VLLM_GPU_DRIVER_RESERVE_MB` |
| E2 | prefill-profile fix (upstream 7ab756940a) + f506 + graph samples | profile covers Engram and indexer prefill peaks | fixes the re-prefill OOM class (not in 5e1b / B32); f506 +2.7% KV; `CUDAGRAPH_SAMPLES=0` est. +0.3 GiB/rank (**unset**) | yes | none | `VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES` |
| E3 | watermark 0.1 + gate fix | admit a prompt only if 10% of the pool stays free | 0 preemptions vs B32's 100-275 per 5 min (confounded) | yes | slower admission | `--watermark` |
| E4 | retention interval 128 | **inert**: SWA is not prefix-cached under bounded replay | none | n/a | none | `--prefix-cache-retention-interval` |
| E5 | SimpleCPUOffload, 10 GiB/rank, eager | async store and restore of compressed + indexer KV blocks | **19-98% of prompt tokens restored from host** in KV-bound windows, 0 preemptions | yes | **no timeout or failure reporting**; host RAM | `--kv-offload-gib-per-rank` |
| F1 | CUDA graphs FULL ≤ 64, 64 seqs | silently becomes FULL_DECODE_ONLY | same as B32 | n/a | mixed steps eager | `--compilation-config` |
| F2 | batch-sharded sampling | owner-rank sampling; all-to-all instead of all-gather | est. 1-3 ms/step; the top-K path is unused with our sampling | yes (greedy / top-k); top-p ties | none | `--enable-batch-sharded-sampling`, `VLLM_BATCH_SHARD_TOPK` |
| F3 | decoder SWA bounded replay (upstream) | layers 21-39 prefill only each request's last 128 tokens | est. ~1.8x prefill compute; with A1/A3/C2, prefill 1.4-1.5k → 2.3-4.6k tok/s | recompute; accuracy tie | none | `swa_bounded_replay` (cache config) |
| G | DCP4 (**not in production**) | compressed KV sharded over 4 ranks; fp8 query/partial exchange | 2x pool; step ≈ B32's; **0.70x of pod speed on Kaggle** | no (fp8 transport) | slow on PCIe; 512-token hit granularity | `--decode-context-parallel-size` |
| H | `--supervise N` | detached restart supervisor | outage 116-174 s on pods | n/a | no hang detection | `--supervise` |

## Where the ~1.5-1.7x per-stream decode speed over B32 comes from

### The measurement, first corrected for batch size

Per-stream speed rises as the running count falls, so compare at matched running counts.

**Method** (script `experiments/E062_manh_vllm_fork/per_layer/phase_h/scripts/fit_steps.py`):
- Per run: 5-min solver metric windows of 200 s or more. Each window counts as "running" the server's running count.
- Least-squares fit of `1 = (gen / n)·(a + b·n) + uncached·t_p`, i.e. decode step = a + b·n, plus a prefill cost per
  uncached token.
- This is a rough regression (est.):
  - mixed steps are folded in;
  - the DCP4 runs sit mostly at 70-97 running, so their a and b are collinear;
  - all are L4 pods at 72 W.

| run (stack; pod; workers; GPU pool) | per-stream tok/s at 52-60 running | at 24-40 running | step fit (ms) | step at 56 | prefill fit |
|---|---|---|---|---|---|
| B32 `candBf_x2t` (B32 stack; old pod; x56; 1.56M) | 8.00 (n 34) | 10.15 (KV 0.90, preempting) | 65 + 0.70·n | 104 | 1,350 tok/s |
| B32 + R1 `b32r1_x2t` (B32 stack + pad 64; pod 2; x56; 2.49M) | 8.67 (n 120) | 11.8 (n 2) | 45 + 0.95·n | 98 | 1,520 |
| fork b9ad `manh_x2t_w80` (DCP4, NCCL; pod 1; x80; 3.06M) | 9.7 (n 1); 7.54 at 70-97 | 12.5 | 32 + 1.09·n | 93 | 2,720 |
| B32-M 5e1b `b32m_x2t` (DCP4, NCCL; pod 2; x80; 3.06M) | 9.1 (n 1); 7.47 at 70-97 | 12.8 | 37 + 1.06·n | 96 | 2,780 |
| **Manh prod v11 `m12_rerun`** (no DCP, flags; pod 2; x64; 1.61M) | 13.7 (n 2); 11.4 at 60-70 | 17.8 (KV 0.88, 0 preempt) | 34 + 0.61·n | 68 | 2,300 |
| **Manh prod v11 `r1_rerun`** (same; pod 2; x64; 1.62M) | 13.2 (n 1); 12.1 at 60-70 | 18.3 | 33 + 0.70·n | 72 | 4,600 |

Also: on cold 32-45k-token prompts (phase H `ab_decode`), Manh's server prefilled at ~4,300 tok/s with ~14 decodes
riding along. B32's stack did 1,415 tok/s on cold 48 x 4K prompts (R1 table) and 1,760 with R1.

**Reading:**
- At matched load, **Manh's production decodes 1.5-1.6x faster per stream than R1** (same pod 2, different days) and
  **~1.7x faster than the B32 replica**.
- The fitted decode step at 56 streams is 68-72 vs 98-104 ms (1.4-1.5x).
- The rest of the per-stream gain is prefill and scheduling, below.
- **The fixed per-step cost halves (≈ 65 → 33 ms) while the per-stream slope stays ~0.6-0.7 ms.**
  - So the gain is mostly in costs paid once per step: communication, weight reads, sampling, gathers.
  - It is not in per-sequence attention.
- The DCP4 builds sit at B32's per-step cost: their extra collectives roughly cancel the base kernels' gain. They won
  only through the 2x pool.

### Ranked causes (B32 stack → Manh production), with confidence

**1. Communication: host all-reduce 556 + L2 weight prefetch + host all-gather replace every NCCL call in a decode
step** (A1, A2). est. −8 to −15 ms per step at 56-64 streams on pods; more on Kaggle, where NCCL is ~2x slower than pod
1's. **Confidence: medium.**
- B32's decode profile: NCCL all-reduce ≈ 27 ms per step at 48 streams (≈ 0.32 ms x 84, pod B) plus the Engram
  all-gather 6.3 ms (E042).
- E062b: +9.8% decode per stream for host AR + row shard, under DCP4, whose own collectives stayed on NCCL.
- Host gather is ~2x faster per call.
- Against: on the two-socket 4090 the host AR was *slower* than NCCL per call (E063 bench). So much of the gain likely
  comes from the prefetch (it turns waits into weight reads) and from the single-socket L4 host. Neither is isolated.
  **First ablation.**

**2. Prefill throughput ~2-3x** (F3 decoder SWA bounded replay [upstream]; A1 at prefill sizes; A3 +11.8%; C2
1.0-1.35x GEMMs; the mHC split-k pre-norm 3-4x). **Confidence: medium-high for the prefill speed, medium for its effect
on decode.**
- Mixed steps run eager and stall every decoding stream for the chunk's duration (F1).
- B32 spent ~12-15% of server time in prefill (fit); the fork 5-11%.
- est. +5-10% per stream on real traffic, and much faster recovery after restarts and KV-bound re-prefills.

**3. The fork's SM89 kernel base vs B32's** (FP8 indexer cache with the Triton FP8 tensor-core MQA logits kernel, vs
B32's MXFP4 kernel that is bound by register decoding; Triton sparse MLA; MXFP8 o_proj; the Engram UVA lookup and
value-sharded WKV; the dsv4_topk router). est. −5 to −15 ms per step. **Confidence: low-medium.**
- Inferred only from DCP4-fork ≈ B32 per step despite ~76 extra collectives. No same-workload kernel A/B exists.

**4. Indexer row shard** (B2): −5.1 ms per step at B = 64, ctx 20-102k; +2.8% serving at x120 (commit). **Confidence:
medium-high** (the author's numbers; ours folded it with A1).

**5. Scheduling under KV pressure** (E3 watermark 0.1 + gate fix, E5 SimpleCPUOffload eager restores): 0 preemptions
vs B32's ~100-275 per 5-min window at KV ≥ 0.88.
- Per stream at 24-40 running: B32 10.15 while preempting, R1 11.8 at KV 0.47.
- est. +10-15% per stream, **only in the KV-bound hours** (hour 2 onward). The pools are the same size (~1.6M).
- **Confidence: medium** (confounded with the solver's gate and the offload connector type).

**6. Expert pad 64,64** (C1): +3.0-4.5% decode, measured on B32's own stack. **Confidence: high.** It also buys the GPU
memory that lets the fork's 1.64x-larger per-token KV (B4) still reach B32's pool size.

**7. Batch-sharded sampling** (F2): est. 1-3 ms per step (all-to-all ~4 MB vs all-gather ~16.5 MB per rank at 64 rows).
The top-K path never fires with our sampling. **Confidence: low.**

**8. mHC token blocks** (D1, v11 only): ≤ 1.1 ms per step, and only at ≤ 32 running (KV-bound phases).
**Confidence: medium** (kernel numbers from the commit).

**9. Sparse indexer logits** (B1): est. < 1-3 ms per step at 20-50k contexts, more past 100k. **Confidence: low.**

**Not causes:**
- CUDA graph mode: both are FULL_DECODE_ONLY with the same 11 sizes.
- V2 runner: both stacks use it.
- `--max-num-batched-tokens`: 2048 in both.
- Retention interval: inert on the fork.
- `--enable-scale-out`: HTTP routes only.
- DCP: absent in production; it only slowed steps.

**Budget check (est.):** at 56 streams the parts sum to −21 to −44 ms per step, against the −30 to −36 ms fitted (104 →
68-72). Items 2 and 5 add per-stream speed on top of the step-time gain.

## A. Communication

### A1. `VLLM_HOST_ALLREDUCE=1`, `MODE=556` (+ PREFETCH_BLOCKS 16, PREFETCH_MAX_MB 36, SHAPE 12x12x256, LARGE_SHAPE 32x32x256, UNIT_KB 16)

**What it does.** It replaces NCCL's SHM ring for every bf16 TP all-reduce of 24 MiB or less, in decode and prefill.
- Setup:
  - Rank 0 creates a `/dev/shm` segment. Every rank `mmap`s it, then `cudaHostRegister`s it (mapped/portable) and gets
    a device pointer. The GPU then reads and writes host memory directly over PCIe (zero-copy, with no copy engines).
- One kernel per call has three kinds of blocks:
  - **Writer blocks** store the rank's input units (16 KiB each) into its host slot, fence at system scope, and post a
    flag.
  - **CPU reducer threads** (2 per rank per instance, `std::thread`) spin on the flags. For each unit they sum the 4
    inputs in **fp32 in rank order** (`((a+b)+c)+d`), round once to bf16, and post the unit to **one shared output
    slot**.
  - **Reader blocks** spin on each unit's done flag and copy the sum back to the GPU with `__ldcv`.
  - Units are pipelined: write, sum and read overlap.
- Traffic: per GPU, N bytes cross PCIe up and N down; packed units cut that to ~0.77N. A ring moves 1.5N each way in 6
  dependent steps. The comment reads "one write and one read of the message per GPU, against 1.5 of each for a ring".
- **Mode bits** (`csrc/host_all_reduce.h:20-57`):
  - 1 LEAD_FENCE, 2 POLL_AHEAD (timing only);
  - **4 PACK:** a lossless bf16 packing at 1.53 B per value (sign+mantissa byte, 4-bit exponent delta, a group-max byte
    per 32 values, escapes). It round-trips the bits exactly, NaN/Inf/±0 included;
  - **8 SIMD_HOST:** AVX-512BW reducer loops ("same operations in the same order, so the same bits"). It falls back
    silently without avx512bw;
  - **16 Q8:** int8 plus an fp16 scale per 16 values. **Lossy.** The `envs.py` comment's "power-of-two scale per 32"
    is stale;
  - **32 PF_STOP:** the prefetch blocks stop once the readers finish, so prefetch never lengthens a call;
  - 64 / 128: Q6 / Q4, experimental, rejected as instance bits;
  - 256 SMALL_RAW (with 4): calls of 64 KiB or less skip packing and use 4 KiB units;
  - **512 SIMD_WARM** (with 8): an idle reducer runs a ~0.2 µs AVX-512 burst every 200 µs so its core keeps the
    512-bit units powered (602df11557).
  - So 556 = 512+32+8+4 (PACK, SIMD, PF_STOP, WARM), 44 = 32+8+4, 60 = 44+16 and 572 = 556+16.
- **L2 prefetch** (`PREFETCH_BLOCKS`):
  - Extra blocks in the same kernel issue `ld.global.cg` over the **weights of the GEMM that follows the all-reduce**,
    so they land in the L4's 48 MB L2 while the call waits on PCIe.
  - The DSv4.1 wiring is in `models/deepseek_v41/nvidia/model.py` (`init_host_ar_prefetch` 1261-1283,
    `_ffn_prefetch_ranges` 310-323, `_attn_prefetch_ranges` 295-307, `prefetch_next` 743-779):
    - the attention-output AR prefetches the shared expert, the router and the leading routed experts' w13, up to the
      36 MiB cap;
    - the MoE-output AR prefetches the next layer's `fused_wqa_wkv`, `wq_b` and other attention linears.
  - The production log reads: "Host all-reduce prefetch: 40 DSV4.1 layers, MoE 36.0-36.0 MiB (8 ranges), next
    attention 19.3-34.9 MiB (9 ranges)".
  - The prefetch is baked in at graph capture. It can be switched at run time through `ctr[3]`
    (`VLLM_HOST_ALLREDUCE_PREFETCH_MODE_FILE`, polled every step).
- **Shapes:**
  - `SHAPE` = writer blocks x reader blocks x threads (decode instance, below 2 MiB);
  - `LARGE_SHAPE` builds a second instance with its own segment and reducers for messages of `LARGE_MIN_KB` (2 MiB,
    about 205 tokens) or more;
  - a third "inner" instance (8x8x128, no prefetch) serves the AR-overlap side stream (A3).
  - All three show in the production log.

**Where it lives:**
- `csrc/host_all_reduce.cpp` (`init_host_ar` 793-883, `reducer()` 650-758);
- `csrc/host_all_reduce_kernel.cu` (`host_all_reduce_kernel` 364-529, `prefetch_l2` 99-138);
- `vllm/distributed/device_communicators/host_all_reduce.py` (`HostAllReduce`, `from_envs`, `should_use` 206-217);
- `cuda_communicator.py:166-173` builds it for the "tp" group only. `all_reduce` 390-393 tries HOST before every other
  backend;
- `envs.py:1540-1643`;
- the keep-warm commit is 602df11557. The host AR itself predates the shallow root.

**Why it helps on PCIe without P2P:**
- Every all-reduce byte crosses the host link anyway. A single host pass moves fewer bytes than a ring, and it has one
  synchronisation chain instead of 6 dependent hops.
- Decode messages are latency-bound: B = 64 is about 655 KB, and there are about 80 all-reduces per step. The pod
  NCCL all-reduce at 64 tokens is 0.162 ms (pod 1) to 0.20 ms (pod B), roughly 13-27 ms per decode step.
- The prefetch turns the GPU's idle wait into weight reads. A decode step is weight-read-bound: B32's experts plus
  dense layers took 66 ms against a ~51 ms floor at 48 streams (E042 profile).

**Measured:**
- E062b, L4 pod 1, x80, DCP4 build 5e1b. Arm C (MODE 556 + row shard + MoE tuning) vs arm A (NCCL):
  - **+6.2% end-to-end gen tok/s** (590 vs 555);
  - **+9.8% pure-decode per stream** (9.85 vs 8.97 at 70 or more running).
  - That result is confounded with ROW_SHARD and run under DCP4. The DCP collectives stayed on NCCL.
- MODE 572 (Q8) = 556 on speed. On the 48 GB box, MODE 60 cost **-7.5 pp** of accuracy (one draw, p = 0.04).
- 602df11557's message gives, for Q8 modes only:
  - per call: 172 -> 146 µs at 64 tokens and 246 -> 224 µs at 120 tokens;
  - in serving: +6.5 ± 2.5 tok/s from bit 512.
- Comparison point (E063 bench, `results/bench.log`): on the two-socket 4090 box the b9ad host AR was *slower* than
  NCCL at every size. Its benefit depends on the host. Kaggle's single-socket host has never been measured.

**Throughput share, est.:**
- 10-20% of the decode step vs NCCL on L4 pods.
- Probably more on Kaggle, whose NCCL is ~2x slower than pod 1's.
- Plus prefill gains: the 32x32 instance covers the 20 MiB chunk all-reduces.
- The prefetch part has never been isolated.

**Lossless?**
- **556: yes.** It is fp32 in rank order with one rounding, identical on all ranks, and independent of unit split,
  shape, threads, packing or SIMD. `tests/distributed/test_host_all_reduce.py` checks it bit-exactly against an fp32
  oracle.
- It is **not bit-identical to NCCL**, whose ring rounds to bf16 per hop (est.), so switching changes outputs at bf16
  noise. Teacher-forced checks sit at the run-to-run noise floor (E062b `tf_manhflags.json`, even with Q8).
- **Q8 (16) is lossy. Never ship 60 or 572.**

**Memory:**
- /dev/shm per instance: `4096 + ⌈W·U·64⌉ + ⌈U·64⌉ + (W+1)·⌈24 MiB⌉` ≈ **120.5 MiB**. Three instances is **~361 MiB**
  of shared pinned host memory. (E062's "24 MiB per rank" understated this.)
- Plus 64 B on the GPU per instance.

**Failure modes:**
- It falls back to NCCL with a warning on: world size outside 2-8, multi-node, or a segment/mmap failure (the ranks
  agree over gloo).
- est.: a /dev/shm too small would more likely SIGBUS at `MAP_POPULATE` than fall back.
- A dead or stalled peer:
  - readers spin for **600 s**, then `printf` + `__trap()`, which is a sticky error that kills the engine;
  - reducers spin with no timeout;
  - so a TP death can look like a 10-minute hang. Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` (as we did) so the
    600 s trap fires first.
- **CPU:**
  - reducers busy-spin ~60 ms after each call (`pause` loop, then `usleep(50)`);
  - in decode, 8 threads stay hot;
  - in mixed steps, up to 24 threads spin, with no affinity set;
  - est.: this could throttle a 23-CPU pod cgroup and starve tool sandboxes. Kaggle's CPU count is not recorded in our
    logs.
  - Whether the AVX-512 path is actually active is **not logged**: SIMD and WARM silently do nothing on a CPU without
    avx512bw.
- Each instance must be used in the same order on every rank, never concurrently.

**Toggle:**
- `VLLM_HOST_ALLREDUCE=0/1` (off by default).
- `VLLM_HOST_ALLREDUCE_MODE`: 556 vs 44 isolates WARM; 556 vs 0 isolates PACK and SIMD.
- `VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=0` or the mode file isolates the prefetch.
- `VLLM_HOST_ALLREDUCE_THREADS=1` halves the spinning threads, with the same bits.

**Benchmark in isolation:**
- There is only a correctness test: `pytest tests/distributed/test_host_all_reduce.py -k 4`.
- Microbench (proposed, same skeleton as that test):
  - 4 processes;
  - `HostAllReduce(group, dev, 24<<20, threads=2, writers=12, readers=12, block=256, unit_bytes=16<<10, mode=556)`;
  - a captured graph of 40 calls on bf16 [T, 5120], T ∈ {8, 24, 64, 120, 2048}, timed by CUDA events (max over
    ranks);
  - vs a graph of NCCL `all_reduce`; modes 44 / 556 / 812 (+256), threads 1 vs 2. Our E063 `comm_bench.py` already has
    the NCCL and host columns.
- In-server: the PREFETCH_MODE_FILE A/B on one server, then the replay A/B in section "Ablation plan".

### A2. `VLLM_TP_HOST_GATHER=1` (commit 9065af32af, after the 5e1bccd07c wheel)

**What it does.** A one-hop, host-staged TP all-gather with no CPU work.
- The kernel (`csrc/host_all_gather.cu`, `host_all_gather_kernel` 73-144; 4 blocks x 512 threads):
  - each block writes its share to its host slot and to its own output, fences, and posts a flag;
  - then it spins on the peers' flags for the same block and reads their shares with `__ldcv`.
- Two parity slot sets, alternating by `seq & 1`, make back-to-back calls safe.
- `seq` lives in device memory, so the kernel is graph-replayable.
- Each call waits on the previous call's event (`cudaStreamWaitEvent`, cu:240-258), which keeps the Engram
  side-stream WKV chain in issue order on every rank.
  - est.: an eager call on another stream right after a graph replay is not event-ordered against it.

**Coverage.**
- The "tp" group, any dtype, 0 < bytes ≤ `VLLM_TP_HOST_GATHER_MAX_KB` (default **256 KiB per rank**), bytes % 16 = 0.
- In `cuda_communicator.all_gather` (491-502) it is tried after symm-mem and before pynccl.
- DSv4.1 call sites at decode size:
  - the Engram gathers (`common/engram.py:1330, 2068, 2079, 2111`; `specialized_decode/engram/prefetch.py:83`);
  - the row-sharded indexer's `_all_gather_rows` (B2);
  - logits only for tiny batches.
- Direct pynccl calls (e.g. `triton_sparse.py:215`, DCP) are untouched.

**Why it helps:** NCCL's all-gather over SHM pays a ring's latency for messages of a few KiB to 256 KiB. B32's
decode-step profile had the Engram all-gather at **6.3 ms per step** at 48 streams (E042, pod B).

**Measured:**
- Commit message, 4x L4 graph replay, host vs NCCL:

  | per-rank shape | host | NCCL |
  |---|---|---|
  | 64x8 | 12.5 µs | 21 µs |
  | 64x128 | 30.5 µs | 55.7 µs |
  | 64x512 | 116.6 µs | 147.6 µs |

- No serving A/B. Phase H's decode A/B switched it off together with B1/B2 for numerics only.
- est.: a 2-5 ms per step saving at 48-64 streams.

**Lossless:** yes, a byte copy. It is checked bitwise eager, under replay and across two streams. Phase H's
flags-on-vs-off decode A/B (with B1/B2) matched at the fp-noise level.

**Memory:** ~2.03 MiB of shared pinned memory.

**Failure:** a 600 s trap on a dead peer, as in A1.

**Toggle:** `VLLM_TP_HOST_GATHER=0/1`, `VLLM_TP_HOST_GATHER_MAX_KB`.

**Benchmark:** the A1 skeleton with `HostAllGather(group, dev, 256<<10)` on [64, 8 / 128 / 512].

### A2b. `VLLM_DCP_HOST_GATHER` / `VLLM_DCP_ASYNC_QUERY_GATHER` (DCP only; inert in production)

- `host_gather.py`: Triton spin kernels for the "dcp" group, off by default. MAX_MB defaults to 32 per rank, a
  **~256 MiB** segment at W = 4. `num_programs = min(SMs, 128)`, so all spinning programs stay resident.
- `VLLM_DCP_ASYNC_QUERY_GATHER` (default 1) runs the DCP fp8 query all-gather on a side stream.
  - E063 warns: host-gather programs spin on every SM, so a concurrent NCCL kernel on another stream can be starved,
    a possible cross-rank deadlock.
  - Production sets the variable but runs no DCP, so it does nothing.

### A3. `VLLM_DSV41_AR_OVERLAP_CHUNKS=2` (26c5906214, memory trimmed in f506afa210; wheel v10+)

**What it does.** In eager prefill and mixed steps, each decoder layer splits its rows into N chunks. Each chunk's
all-reduce then runs on a priority -1 side stream while the next chunk computes. Per layer:
1. **Attention:**
   - projections, indexer, top-k and KV writes run on all rows;
   - the sparse-MLA launches are *recorded* (`triton_sparse._mla_calls`, only with DCP = 1);
   - per chunk, the MLA is replayed for rows [s, e), o_proj runs (`reduce_results=False`), and `submit()` sends that
     chunk's AR into `out[s:e]`.
2. **FFN, per chunk:** wait for that chunk's attention AR, run `mhc_shifted_post_pre(out=slices)`, run the MoE with
   `skip_final_all_reduce`, then submit its AR.
3. **Deferred:** the FFN ARs stay pending into the next layer's `_attn_pre_rows`.

What hides under what:
- attention AR 0 under chunk 1's MLA and o_proj;
- attention AR 1 under chunk 0's mHC and MoE;
- FFN AR 0 under chunk 1's MoE;
- FFN AR 1 under the next layer's mHC.

Overlapped ARs use the third "inner" host-AR instance (8x8x128, few SMs). The last one is exposed and uses the normal
instance.

**Where it lives:**
- `vllm/models/deepseek_v41/nvidia/ops/ar_overlap.py` (`ChunkedAllReduce` 48, `get_chunked_all_reduce` 119,
  `plan_mla_rows` 168, `run_mla_rows` 203);
- `nvidia/model.py` (`DeepseekV4DecoderLayer.__init__` gating 432-441, `forward` 564, `_chunked_all_reduce` 782,
  `_attn_ffn_overlapped` 802, `_attn_rows` 878, `_attn_pre_rows` 917, `_run_layers` `defer_ar`/`flush_pending`
  1550-1558);
- `envs.py:1203`.

**When it triggers:**
- `num_chunks = min(N, T // 256)`, so 512 rows or more for 2 chunks, with boundaries on 16 rows;
- every chunk must pass the host AR's `should_use` (24 MiB or less);
- Triton sparse attention (the SM89 default), no SP, no SM100 mHC stream, no DeepGEMM, and the host AR on;
- off during capture and in piecewise graphs. So it never touches decode graphs. A 2048-token step becomes 2 x 1024
  rows (10 MiB each).

**Why it helps:** the prefill AR is about half of prefill time on these PCIe stacks: 0.6-1.3 s per 2048-token chunk
under NCCL (E042/E057). Chunking hides most of it under GEMMs.

**Measured** (commit, 4x L4, MODE 556, 64 x 12,288-token prompts): **prefill 3,758 -> 4,199 tok/s (+11.8%)** with
N = 2 (N = 3: +8.2%). Decode is flat (975 / 970 / 974).

**Lossless?** Mostly bitwise:
- the host AR is independent of the split;
- mHC chunks of 256 rows or more stay in the same split-k bucket;
- MLA launches are cut only where `_num_splits` is unchanged.

o_proj and the MoE see a smaller M, so they can pick other tiles. That includes a different Humming tuning interval
(m = 6,144 instead of 12,288), so the result is not bitwise. Greedy divergence matched off-vs-off restarts (commit
message).

**Memory:**
- 26c5906214 cost 0.7% KV (1,642,735 -> 1,630,697 tokens): `out` 10 KiB/token, a new residual 40 KiB/token and
  `ffn_in` 10 KiB/token stay live across the indexer's prefill peak.
- f506afa210 allocates the reduced output after the indexer, writes the residual in place, and makes the Engram mHC
  post / Engram output in place. Result: **1,675,566 tokens (+2.0% vs off)**. Part of that is the N-independent Engram
  in-place change (est.).
- Host: one more ~120 MiB shm instance plus 2 spinning reducers per rank.

**Failure modes:** the layer silently falls back to the unchunked path when a precondition fails (no log line). Hangs
end in the 600 s host-AR trap.

**Toggle:** `VLLM_DSV41_AR_OVERLAP_CHUNKS=0`.

**Benchmark:**
- `vllm bench serve --dataset-name random --random-input-len 12288 --random-output-len 2048 --num-prompts 64`, N = 0
  vs 2 on one pod;
- an nsys trace of one 2048-token step shows the side-stream overlap;
- unit tests: `tests/kernels/mhc/test_mhc_kernels.py -k row_chunks`, `tests/kernels/test_engram.py -k
  fused_engram_post_wkv`. Nothing tests `plan_mla_rows` / `run_mla_rows`.

## B. Sparse attention (DSA) and indexer

### How E128's DSA is wired (needed for B1-B4)

Sources: `attention.py` 315-360, `nvidia/model.py` 1070-1090, `compressor.py` 276-303.

**Compression codes per layer:**
- Layers 0-1 and the 3 MTP layers: code 0, SWA only.
- Layers 2-19: code 2 ("C2"), 2 tokens pooled per state. Commit c2c8736696 calls these "ratio-4", but the code uses
  `tokens_per_state = 2`.
- Layers 20-39: code 1 ("C1"), one compressed state per token.

**KV sources:** only layers 2, 8, 14 and 20 have compressors, compressed KV and their own indexer K cache. Every
compressed layer reuses the top-k of the newest index source at or below it.

**Two-level selection:**
- Layer 20's indexer (the *candidate writer*) scores the whole context densely.
- It keeps the top 2048 blocks of 8 positions (16,384 positions): `select_candidate_blocks`, `candidate_blocks.py`
  166.
- Consumers 24 / 28 / 32 / 36 pick their own top-512 inside those blocks.
- Indexers 2 / 8 / 14 are always dense.

**The indexer is replicated over TP:** `wq_b`, `weights_proj` and `wk` are `ReplicatedLinear` ("no tensor parallel,
just replicated"). So without B2, every rank scores every row.

**Indexer K cache:**

| | fork default | B32 |
|---|---|---|
| format (`dsa_indexer_uses_fp4` resolves "auto" to fp8) | FP8, 128 + 4 B per position | MXFP4, 64 + 4 B |
| all 4 caches, per token per rank (est.) | ~330 B | ~170 B |
| decode logits kernel | `_fp8_paged_mqa_logits_kernel` (`triton_mqa_logits.py` 159/509): FP8 tensor-core `tl.dot` | `_mxfp4_indexer_logits_kernel`, bound by in-register MXFP4 decoding at 64-71 GB/s (E042) |

- MXFP4 is **not usable on SM89 in the fork**: the Triton decode branch raises "does not support FP4 Q".
- The indexer caches sit in the same KV block as the main cache and are replicated on all 4 ranks.

### B1. `VLLM_TRITON_SPARSE_INDEXER_LOGITS=1` (shallow root; default off)

**What it does** (decode only; `sparse_attn_indexer.py` 1147-1180, `_triton_sparse_decode_topk` 1313-1359): for the
candidate *consumer* layers it replaces

> dense [rows, max_model_len] logits → `apply_candidate_mask` (a −inf mask written over the full width) → top-k

with

> `candidate_blocks_to_sparse_indices` (`sparse_mqa_logits.py` 184; drops invalid blocks, sorts) →
> `triton_fp8_paged_sparse_mqa_logits` (scores only the ≤ 16,384 candidate positions into compact
> [rows, 16384] logits) → top-512 → `remap_sparse_topk`.

So "indexer consumers score only layer 20's candidates" means the consumers skip the positions layer 20 rejected,
instead of scoring them and masking them out.

**When it applies:**
- consumer layers only, `next_n` = 1, a uniform batch, FP8 Q, no DeepGEMM, and no DCP (or the TP token shard);
- prefill always uses masked dense logits;
- it **only saves work once a context has more than 16,384 C1 positions** (tokens). Below that every block is a
  candidate.
- est.: at our 20-50k-token contexts the consumers read K for 1.2-3x fewer positions, and they skip the full-width
  mask pass.

**Measured:** no number found. est.: under 1-3 ms per step at 64 streams (consumers are 0.72 ms per layer dense at
B = 64 per the 5e1b message), larger at 100k+ contexts.

**Lossless?** Same scores per position and the same selected set ("Same selection as masking the dense logits"). Ties
can resolve differently. One edge case: a candidate past the logits width is clamped by the mask path but dropped by
the sparse path.

**Memory:** transient logits shrink to rows x 64 KB. est.: no KV gain, because the profile still reserves the dense
decode logits.

**Graphs:** compatible (fixed width 2048 x 8, no host sync).

**Toggle:** this env var only. It is unrelated to `attention_config.indexer_sparse_logits`, the DeepGEMM/SM100 path.

**Benchmark:**
- existing test: `tests/v1/attention/test_indexer_tp_token_shard.py::test_sharded_sparse_logits_are_bitwise_unsharded`;
- proposed: on one L4, a random paged FP8 cache, B = 64/80, C1 contexts U(20k, 50k), 2048 random candidate blocks
  (newest pinned). Time dense + mask + top-k against `_triton_sparse_decode_topk` with `triton.testing.do_bench` and
  assert equal sorted index sets.

### B2. `VLLM_DSA_DECODE_INDEXER_ROW_SHARD=1` (5e1bccd07c; `_MIN_REQS` default 8)

**What it does:**
- Rank r scores decode rows r, r+4, ...: dense logits plus candidate select or mask, or B1's sparse path for
  consumers.
- It takes the local top-512 ([B/4, 512] int32), `all_gather`s over TP (≤ 256 KiB, so it takes A2's host gather),
  then restores the row order.
- The Q projection and the K insert stay replicated; only scoring and top-k are split. Candidates stay rank-local
  (writer and consumers share the row order).

**Where:**
- builder: `v1/attention/backends/mla/indexer.py` 1207-1231 and `_build_decode_row_shard` 1306-1342 (it refills
  persistent buffers from the padded row count, so FULL decode graphs capture the sharded path, all-gather included);
- kernels: `sparse_attn_indexer.py` dispatch 1071-1095, `_row_sharded_decode_topk` 482-555;
- `envs.py` 64-65 and 1179-1188.

**When it applies:** CUDA, TP > 1, DCP = 1, no PCP, `next_n` = 1, no DeepGEMM, pure single-token decode steps with
`num_decodes % 4 == 0` and at least `MIN_REQS` (8) rows. Otherwise it falls back to the replicated path.

**Measured** (commit, 4x L4, B = 64, ctx 20k-102k), ms per layer, full → slowest shard:

| indexer | full | sharded |
|---|---|---|
| plain C2 (2/8/14) | 1.18 | 0.35 |
| candidate writer (20) | 2.44 | 0.77 |
| consumer (24/28/32/36) | 0.72 | 0.46 |

- That is **~5.1 ms per decode step** (est. 8.9 → 3.7 ms over the 8 indexers).
- TIR serving at concurrency 120: **+25.7 ± 2.8 tok/s (+2.8%)**, KV 3.78 → 3.91 GiB per rank (smaller graph-pool
  logits).
- E062b folded it into arm C (+9.8% decode per stream with host AR).
- For scale: B32's own indexer with the #156 skip was 3.9-5.3 ms per step at 48-64 rows on a 4090 (E054). The L4 is
  ~1.35-1.7x slower.

**Lossless?** "value-exact … only the order and the choice among tied keys may differ" (commit). The persistent top-k
is already nondeterministic on ties (E054).
- Phase H decode A/B, B1 + B2 + A2 on vs off: mean |dlogprob| decode-vs-prefill 0.0362 vs 0.0347, decode-A vs
  decode-B 0.0216. That is fp noise.
- Pod 3 run 3 paired accuracy with these on, at 55-80 concurrent: 74.5% vs eval 73.0% (204 pairs). No cost.

**Memory:** slightly *saves* KV (above).

**Failure modes:** silent fallback. Prefill row sharding (`tp_prefill_row_sharding_supported`) is off under
cudagraph_mode FULL. Not compatible with B3 on the caches B3 shards (B3 sets `dcp_world_size = token_shards` first).

**Toggle:** this env var; `_MIN_REQS`.

**Benchmark:**
- `tests/v1/attention/test_indexer_decode_row_shard.py` checks the row order only;
- kernel savings: time B vs B/4 rows on one GPU;
- the gather needs 4 GPUs;
- end to end: the same server, env flipped, at fixed concurrency.

### B3. `VLLM_DSV41_INDEXER_TP_SHARD` (c2c8736696; in the fork, **off in production**)

**What it does:**
- It shards the indexer K caches over TP by compressed position: 16 tokens per rank in turn, a DCP-style interleave
  (`indexer_tp_token_shards`, `indexer.py` 272-308).
  - Level 1 shards the C2 caches (2/8/14).
  - Level 2 also shards layer 20's C1 cache, read by 24-36.
- Each rank scores its shard at unsharded positions (bitwise-identical logits), takes a local top-k, then an exact
  all-gather merge (`merge_dcp_topk` / `merge_sparse_topk` / `merge_dcp_candidate_blocks`, `dcp_indexer_topk.py`
  244/288/320; ties go to the lowest global id).

**Gain:** KV capacity **13.63x → 14.75x (level 1) → 15.46x (level 2)** at 131k on 4x L4 (commit), i.e. +8% / +13%
pool. It is the largest untried KV lever on the no-DCP config.

**Costs:** an all-gather of (score, id) pairs per indexer layer per step (rows x 512 x 8 B, plus rows x 2048 x 8 B at
layer 20), and B2 is lost on the sharded caches.

**Requirements:** no DCP/PCP/spec decode/`indexer_sparse_logits`, and the block must be divisible by 16 x TP.

**Test:** `tests/v1/attention/test_indexer_tp_token_shard.py`.

### B4. Attention kernel: fork Triton sparse MLA vs B32's FlashInfer V4.1 kernel (automatic)

**The fork:** `_select_dsv4_attn_cls` (`nvidia/model.py` 229-288) returns `DeepseekV4TritonSparseAttention` on SM89
("Neither FlashMLA nor FlashInfer has sparse MLA kernels for SM89").
- It reuses the SM120 FlashInfer flow, with `triton_sparse_mla_dsv4` (`ops/triton_sparse_mla.py` 640): BLOCK_H 16,
  BLOCK_N 16, split-KV up to 16 plus a combine kernel.
- It reads the **584-B `fp8_ds_mla` record** per compressed position: 448 fp8 NoPE + 64 bf16 RoPE + 8 scale bytes.
- The production log's "DeepseekSparseSWAFlashInferBackend" is the SWA metadata backend; it forces
  `cudagraph_mode=FULL_DECODE_ONLY` (F1).

**B32's stack** (vision11 wheel): `DeepseekV41NativeMixedAttention` → FlashInfer's
`_deepseek_v41_mixed_sparse_attention_with_inv_rope`, BF16 Q, **288-B FP4 compressed rows** and 528-B SWA rows.

**Consequence:**
- est.: the fork stores ~2x the bytes per compressed position, plus ~1.9x in the indexer cache.
- Measured per GPU (available KV memory / pool tokens):
  - **fork no-DCP 3.87 GiB / 1,608,578 = 2.58 KB per token**;
  - **B32 2.29 GiB / 1,556,850 = 1.58 KB per token** (1.64x);
  - fork DCP4 3.47 GiB / 3,058,602 = 1.22 KB per token.
- So **the fork's no-DCP pool matches B32's only because it frees 1.6 GiB more GPU memory per rank** (weights 14.56
  vs 16.9 GiB: pad, embed UVA, RoPE cap, profiler trims). DCP4 was how it got a 2x pool.

**Speed:** attention was ~10-17 ms of a ~131 ms B32 decode step (E042 profiles). No same-workload kernel A/B exists.

**Lossless vs B32:** no. It uses a different KV quantization: fp8 + bf16 RoPE, *finer* than B32's FP4 rows. Across
stacks: teacher-forced NLL +0.004 per token vs B32 (E062), and accuracy tied (77.1 vs 76.5, 120 tasks).

**Test:** `tests/kernels/attention/test_triton_sparse_mla_dsv4.py`.

### B5. `--attention-config.sparse_mla_prefill_workspace_factor=2` (upstream option)

**What it does:**
- It sizes the V4.1 indexer prefill K-gather workspace: max_model_len x 8 x factor rows / compress ratio, 132 B per row
  (`indexer.py` 990-1001, `sparse_attn_indexer.py` 645-660), plus the per-chunk row cap of indexer prefill chunking.
- The default is None, which resolves to 5 at ≤ 256K (`config/vllm.py` 1882-1890). `ge=2`.

**Gain (est., C1 layers per rank):** 865 → 346 MB at 163,840 context, i.e. **~0.5 GB more for KV** (~0.4 GB at
131,072).

**Speed:** est. none, since a chunk still holds 16 full contexts and the 512 MiB logits cap binds first.

**Lossless:** yes (sizing only).

**Toggle:** remove the flag (factor 5).

**Benchmark:** the "Available KV cache memory" log line, plus one long-prefill smoke (160k-token prompt) to confirm no
workspace error.

## C. MoE and linear kernels

### C1. `VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64` (default 256,128; in the shallow-root tree)

**What it does:**
- It sets the N and K pad multiples that Humming applies once at load
  (`quantization/utils/humming/moe.py` `_prepare_and_transform_sublayer` 392, pad 412-418; `envs.py` 2037-2044).
  Upstream hard-codes 256,128.
- E128's TP4 expert shards (w13 [1152, 5120], w2 K = 576) are then stored unpadded, instead of 1280 rows and K = 640.
- The extra w13 rows were zeros whose outputs were discarded. w2's K loop changes from 640 to 576.

**Why it helps:**
- Memory: 11% fewer expert bytes, ≈ 1.17 GiB of 2-bit codes per GPU plus scales and zero points.
- Speed: at 24 or more streams almost every expert is read on every decode step, so the step is weight-read-bound
  (E042: 2-bit experts ≈ 13.2 GB/GPU, ~51 ms floor at 48 streams). Each MoE GEMM therefore streams 11% fewer bytes.

**Measured** (R1 = the same knob on B32's stack, L4 pod 2):
- weights 16.90 -> 15.64 GiB (**-1.26 GiB**);
- KV 1,556,850 -> **2,486,783 tokens (+59.7%)**;
- decode +3.6 / +4.5 / +3.0% at B = 48 / 56 / 64 (4K prompts);
- cold prefill +24% (one sample);
- 11 h replica: Kaggle-clock 77.92 vs B32 77.43, with 209 vs 106 second attempts.

**Lossless?**
- No: the w2 reduction length changes, so fp32 accumulation order changes.
- Measured as noise:
  - same-box 120-task A/B `acc_r1` vs `acc_b32`: -0.1 pp [-8.3, +8.3]. This supersedes the earlier cross-pod 71.1 vs
    76.0 warning;
  - on the fork, pad + UVA + interleave vs defaults: NLL +0.0001.

**Failure modes:**
- N must stay a multiple of 64 and K a multiple of 512 / activation bits.
- Humming 0.1.12 (B32's) and 0.1.18 both accept 64,64.
- No test covers the knob.

**Toggle:** unset it, which gives 256,128.

**Benchmark:**
- the "Model loading took" and "GPU KV cache size" log lines;
- `decode_window.py` at B = 48/56/64, as R1 did (`experiments/E062_manh_vllm_fork/r1/`).

### C2. `VLLM_HUMMING_MOE_TUNING_CONFIG` (cd8e75973e; in 5e1bccd07c+)

**What it does:**
- It takes a JSON string or file `{"NxK": [[lo, hi, cfg], ...]}`, with m = routed rows = tokens x 6.
- `HummingExpertsBase.get_tuning_config` (`fused_humming_moe.py` 172-193) puts the intervals *in front of* Humming's
  heuristic table. The first interval with lo < m ≤ hi wins, both for vLLM's `moe_align_block_size` block size
  (`prepare_humming_moe_kwargs` 702-721 / 782-787) and for Humming's kernel lookup.
- It is ignored under `VLLM_BATCH_INVARIANT`.
- Log: "Humming MoE 1152x5120: 4 tuned intervals".

**Production table:**
- w13: intervals (1926, 3680], (3888, 5504], (5504, 8192], (8192, 12288].
- w2: intervals (1840, 4248], (4248, 5592], (5592, 8192].
- `num_sms` 58 (L4), stream-k for large w13.

So it only affects **steps of about 321 tokens or more** (prefill and mixed). Decode steps (≤ 64 tokens, ≤ 384 rows)
and decode graphs are untouched.

**Measured:** commit message on L4 E128: gate/up 1.0-1.28x, down 1.0-1.35x, relative error vs fp32 unchanged
(0.00204-0.00207). There is no serving A/B. In E062b it was folded into arm C.

**Lossless?** No: other tiles mean another reduction order, at the noise level.

**Failures:**
- gaps in the table fall back to Humming's heuristics, or to block 64 with a warning;
- the configs are L4-specific (`num_sms` 58);
- AR-overlap chunks (A3) can move a step into another interval.

**Toggle:** unset the variable.

**Benchmark:** nsys per-GEMM at T = 512 / 1024 / 2048, or `vllm bench serve` with 2048 in / 1 out, A/B on one pod.

### C3. Other kernel changes on the E128 path (automatic, no knob)

- **Triton sparse MLA and Triton MQA logits for SM89**, where FlashMLA / DeepGEMM are missing:
  - files: `nvidia/ops/triton_sparse_mla.py`, `kernels/attention/dsa/triton_mqa_logits.py`; selected at
    `nvidia/model.py` 278-280;
  - the split-KV reduction order depends on the batch;
  - see B4 for which backend production actually reports.
- **o_proj through Humming MXFP8** (`HummingMxfp8BmmLinearKernel`, `kernels/linear/mxfp8/humming.py` 94;
  `deepseek_v4/nvidia/ops/o_proj.py` 38/65): with `--linear-backend humming` the grouped `wo_a` runs as real MXFP8
  GEMMs instead of a bf16-dequantising emulation. Test: `tests/kernels/quantization/test_humming_mxfp8_bmm.py`.
- **dsv4_topk router:** the 128-expert shape is whitelisted, so E128 uses the fused Triton top-6 with warmup. (Our
  dynamic-count PR widens this to 6..512.)
- **The mHC pre-norm split-k GEMM** (shallow root): without DeepGEMM, upstream ran one block per token. The fork uses
  8-token x 12-output blocks over 8 k-slices (4 above 128 tokens). The comment says "3-4x faster from 16 to 2048
  tokens" on L4. It is used by prefill/mixed steps and `mhc_pre_delayed`; its fp32 order is deterministic within a
  split bucket. See D1.
- **Not used by production E128:** `VLLM_MARLIN_PAD_M` / `M80` (Qwen/NVFP4), `VLLM_SM89_FP8_TILES` (Qwen3.8 CUTLASS
  tiles), the qwen3_dflash2 logits-processor refactor, and all qwen4_exp / GDN / DFlash / LoRA / hisparse changes.

## D. Model-level changes

### D1. The fused mHC kernel with token blocks (71dca8bc24; in wheel v11 only, not v10)

**What mHC is:**
- The residual is 4 streams ([T, 4, 5120] bf16).
- At each sublayer boundary, a *post* step mixes the sublayer output into the streams, and a *pre* step computes 24
  mix coefficients plus Σr'² for the RMS.
  - fn is [24, 20480] fp32 (1.97 MB).
  - The pre step then runs 20 Sinkhorn iterations on the 4x4 comb, collapses the streams and RMS-normalises.
- `mhc_fused_tilelang` (`kernels/mhc/tilelang_kernels.py` 744) fuses post with the pre-norm GEMM, but **only for
  T ≤ 32** (`_FUSED_POST_PRE_MAX_TOKENS`).

**What changed:**
- The grid goes from (T, n_tiles, split_k) to (ceil(T / block_m), n_tiles, split_k). Each thread loads its fn slice
  into registers once per hidden index and applies it to block_m tokens.
- The table `_FUSED_POST_PRE_TOKEN_BLOCKS` (102):
  - ≤ 4 tokens: 2x1 (unchanged);
  - 5-15: 8x2;
  - 16-17: 8x4;
  - 18-28: 12x4;
  - 29-32: 8x4.
- It applies only to CUDA below SM100 (`_fused_post_pre_token_blocks_supported` 117).
- est.: fn L2 traffic per call drops from ~63 MB to ~16 MB at 32 tokens, and the grid from 1024 to 192 blocks (on 58
  SMs).

**Measured:** commit message, graph of 80 calls:

| tokens | before | after |
|---|---|---|
| 8 | 14.7 µs | 13.4 µs |
| 16 | 21.6 µs | 18.0 µs |
| 24 | 32.2 µs | 22.2 µs |
| 32 | 42.2 µs | 28.1 µs |

- est.: about 77 fused calls per step, so **~1.1 ms per step at 32 running and ~0.3 ms at 16**.
- At 64 running, decode batches are above 32 tokens and never take the fused kernel. The gain lands only in low-load,
  ramp and KV-bound phases.
  - Those phases matter: in the KV-bound hours of Manh's v12 replica, running falls to 25-41.

**Lossless?** Bit-identical. The sum order depends only on n_thr / n_splits; `test_mhc_fused_token_blocks_are_exact`
checks it with atol = rtol = 0.

**Cost:** 4 instead of 2 fused configs to JIT at warmup.

**Toggle:** none. Compare wheel v10 against v11, or monkeypatch the table to the old `((15,2,1),(32,6,1))`.

**Benchmark:** `pytest tests/kernels/mhc/test_mhc_kernels.py -k "fused or token_blocks"`, plus a graph of 80
`mhc_fused_post_pre_delayed_tilelang` calls at T = 8-32. (Two pre-existing off-by-one failures:
`test_mhc_fused_post_pre[4-4096-128]`, `[4-7168-128]`.)

### D2. Engram host tables: `--engram-config {"cpu_offload": true}` and `--engram-config.use_thp=true`

**What it does** (`vllm/models/deepseek_v41/common/engram.py`):
- `cpu_offload` (upstream default) keeps the n-gram embedding tables in pinned host memory.
- Per step:
  - hashes are computed once in `DeepseekV4Model.forward` (1338-1401);
  - `_start_prefetch` (2022-2044) launches the Triton `_engram_lookup_kernel` (651-759) on a side stream (half the
    SMs). The GPU reads the rows straight from host memory through UVA and dequantises them into a persistent
    `staged_rows` buffer, which is graph-safe;
  - the rows are then all-gathered (12 KiB per token; A2's host gather), and the value-sharded WKV exchanges partials.
  - est.: ~3.2 KB per token per rank over PCIe for both layers. Layer 14's lookup hides behind layers 0-13; layer 1's
    has little to hide behind.
- **Per-rank sizes 2.8 / 10.8 / 20.7 / 29.7 GiB** (64.0 GiB in total). The 24 hash columns (n = 2/3/4 x 8 heads) are
  split 6 per rank:
  - rank 0 holds 2-grams, rank 3 holds 4-grams;
  - compact tables keep the touched rows (E043 census: 4-gram banks 95% touched, 2-gram 13%);
  - plus a `row_map` (est. ~0.7 GiB pinned per rank, not in the log line).
- `use_thp` (upstream; `_allocate_huge_page_storage` 945-979): mmap + `MADV_HUGEPAGE`, prefault, `cudaHostRegister`,
  then a best-effort `madvise(MADV_COLLAPSE)` (1219-1230). The checkpoint page cache is dropped before and after
  (dea3a3d504).

**Fork additions:** compact tables, `row_map`, the row-chunk loader a6c3ee8c2b (`model.py` `_engram_row_chunks`
1862-1920), the value-sharded WKV, and `out=` in-place paths (f506afa210).

**Why it helps:** 64 GiB of tables can never fit on 4x 24 GB. Huge pages cut TLB misses on the GPU's UVA reads of
random rows (est.), and Manh reports ~132 GiB of host RAM with THP.

**Measured:**
- Pod 2: 100% THP-backed, load 47-48 s per rank.
- Pod 3 (fragmented host): MADV_COLLAPSE failed on TP1-TP3, the load took **1,235 s on TP3** (server up after
  1,525 s), and synchronous compaction ran (compact_stall +260 in 10 s). On the same pod a later run took 495 s.
- Host RAM is the same either way: about 130-131 GiB of anonymous memory for the server at steady state.
- No throughput A/B of THP on vs off exists.

**Lossless:** the lookup is bit-exact. The value-sharded WKV is not bitwise (fp32 order; tests use a 1e-2 tolerance).

**Failure modes:**
- **startup time under host-memory fragmentation** (Kaggle runs the same `use_thp=true`);
- warnings: "Engram huge-page allocation failed", "cudaHostRegister failed", "MADV_COLLAPSE failed; keeping existing
  pages";
- the largest rank releases its GPU ~20-25 s after its process exits.

**Toggle:** `--engram-config.use_thp=false`.

**Benchmark:**
- `pytest tests/kernels/test_engram.py -k "lookup_matches_torch or use_thp or prefetch or value_sharded"`;
- time `layer.lookup` for T = 1-8192 with CUDA events;
- for startup, log `/proc/vmstat` compact_* during load (pod 3 `dcp4_slow_load/`).

### D3. `--cpu-offload-gb=0.35 --cpu-offload-params=embed_tokens --offload-backend=uva`

**What it does:**
- The fork's `UVAOffloader.wrap_input_embedding` (`offloader/uva.py` 74-100; called at `model.py` 1092-1105) moves the
  vocab-parallel embedding shard to pinned host memory and replaces `p.data` with a CUDA UVA view.
- `F.embedding` then reads only the batch's rows over PCIe.
- Log: "Offloader set to UVAOffloader", "Total CPU offloaded parameters: 0.31".

**Saved:** 32,320 x 5120 x 2 B = **0.31 GiB per rank** of GPU memory, est. ≈ 0.12M KV tokens. The same amount is
pinned on the host.

**Cost (est.):** 10 KiB per token per rank, tens of µs per decode step, ~20 MiB of reads per 2048-token prefill.

**Lossless:** yes; `test_vocab_embedding_uva_tp4_shards_and_graph_replay` uses zero tolerance. Graph-safe.

**Failures:** errors if lm_head is tied or UVA is disabled (`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`). `--cpu-offload-gb`
must cover 0.31 GiB.

**Toggle:** drop the three flags.

**Test:** `pytest tests/kernels/core/test_uva.py -k embedding`.

### D4. Smaller always-on memory fixes

- **RoPE cos/sin cap (b1a14abb7a):** the two fp32 [positions, 64] caches are sized at `max_model_len` instead of
  1,048,576 (`common/rope.py` 50-56). That is 256 -> 32 MiB each, **448 MiB freed per rank**: KV 1,604,011 ->
  1,776,501 tokens (+10.8%) at 131k (commit). Bit-identical; no toggle.
- **Page-cache drop (dea3a3d504):** `drop_checkpoint_cache` now recurses into `gptq/` and stops the safetensors
  prefetch threads. File cache after startup 34 -> 8.8 GiB (commit). Host RAM only; runs only with `use_thp`.

## E. Memory, KV pool, prefix cache, offload, scheduler

### E1. `VLLM_GPU_DRIVER_RESERVE_MB=256` and the allocator cap (b9ad670a67); expandable segments after the cap (46a2cb33de); `VLLM_GPU_MEMORY_LOG_INTERVAL=60`

**What it does:**
- `Worker._reserve_driver_memory` (`v1/worker/gpu_worker.py` 1150-1178) runs at the very end of startup: after the KV
  allocation, graph capture, sampler warmup and the gc freeze.
- It reads the driver's free memory and the allocator's reserved bytes, then calls
  `torch.cuda.set_per_process_memory_fraction((reserved + free - reserve) / total)`.
- So the caching allocator may grow by at most (startup free − 256 MiB). Past that it must free cached blocks and
  retry, instead of taking the last **256 MiB**, which stays for non-torch allocations: lazily loaded CUDA modules,
  cuBLAS/NCCL workspaces.
  - The notebook comment "keep 512 MiB" is stale; 512 was B32-M/E062's value.
- It raises at startup if less than the reserve is free.
- Log line: "Capped the PyTorch allocator at X GiB (Y GiB reserved), keeping 0.25 GiB free …". The v12 replica logged
  X − Y = 2.00-2.05 GiB.
- The same commit adds Humming `load_kernels` (`model_runner.py` 1087, `utils/humming.py` 116/149): it replays every
  kernel m-range seen in `profile_run` before the KV pool takes the device. Without it, the first prefill of a new
  size loads a module late and OOMs.

**46a2cb33de:**
- Right after the cap, the native allocator switches to `expandable_segments` (`gpu_worker.py` 1162-1170) via
  `set_alloc_conf`.
- Why: under a fixed cap, transients of varying sizes leave unreleasable holes in split segments. The commit records
  an OOM after 70 min at util 0.95, asking for 254 MiB with 695 MiB reserved but unallocated.
- The KV pool and profile keep ordinary segments, so the pool size is unchanged.
- **The notebook strips `expandable_segments` from `PYTORCH_CUDA_ALLOC_CONF` because upstream refuses to start with any
  KV connector while it is set** (`VllmConfig._verify_kv_transfer_compat`, `config/vllm.py` 1299-1317). The runtime
  switch bypasses that check after the KV tensors exist.

**Memory log:** `_log_gpu_memory` (1180-1195), checked at the top of `execute_model` every 60 s: "GPU memory: A GiB
allocated (peak P), R reserved (S in split segments), F free, N allocator retries, K OOMs."

**Measured:**
- v12 replica (2 h): allocated flat at 18.85 GiB; reserved and peak creep up ~0.2 GiB per hour.
- Extrapolated: cap − peak reaches 0.26 GiB in ~7 h, and cap − reserved reaches 0 in ~5 h. At the cap, expect allocator
  retries (latency spikes), not OOMs.
- 0 retries and 0 OOMs so far.

**Why it matters:** stability, not speed. It is what lets util 0.90-0.91 run 12 h. The B32-M 46.5 postmortem was a VRAM
problem.

**Lossless:** yes.

**Toggle:** `VLLM_GPU_DRIVER_RESERVE_MB=0` removes both the cap and the switch.

**Benchmark:** a long mixed-size prefill churn (our 12 h replica); track retries, OOMs and split bytes.

### E2. Startup-profile fixes: upstream 7ab756940a (#63047), f506afa210, 34339df964

**7ab756940a** is upstream, an ancestor of 34091a07c7. It is in wheels v10/v11 and **missing from 5e1bccd07c and from
B32's 0.28 lineage**. It fixes three things the DSv4.1 profile missed:
1. Engram activations: profile runs now use `dummy_hashes`.
2. The indexer prefill logits: `del logits` (`sparse_attn_indexer.py` 1047) frees each prefill chunk's logits before
   the next chunk allocates. Before, two 512 MiB chunk buffers could be live, against one profiled.
3. The indexer metadata `expanded_block_table_buffer`, now sized min(max_num_seqs·next_n, max_num_batched_tokens) rows.

The KV pool gets slightly smaller, but runtime peaks now match the profile. This is the "re-prefill of long requests"
OOM class that B32-M likely hit.

**f506afa210:** see A3 (+2.7% KV vs N = 2 before the trim).

**34339df964** (`VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES`, default 2, **not set in production**):
- The profiler captures 2 FULL graphs and extrapolates the rest.
- On DSv4.1 4x L4 that estimated 0.51 GiB against 0.18 GiB actual per rank, and the excess comes out of KV.
- est.: `=0` (capture all 11) returns ~0.3 GiB per rank to KV, about +0.12M tokens, for some startup time. A free KV
  lever to test.

### E3. `--watermark=0.1` (upstream #44594) and the fork's gate fix

**What it does:**
- `watermark_blocks = int(0.1 · num_blocks)` (`kv_cache_manager.py` 217).
- In `allocate_slots` (544-551, 572, 611), WAITING / PREEMPTED requests are admitted only if 10% of the pool stays free
  after their *full* prompt is placed (`scheduler_reserve_full_isl` defaults to True).
- Running requests' decode growth is never gated.
- Free blocks include evictable prefix-cache blocks, so the watermark limits concurrency, not cache retention.

**Fork fix** (shallow root, `scheduler.py` 1522-1526, `_holds_kv_blocks` 2779): a request that already holds blocks is
not re-gated. Examples are one with computed tokens, or one returning from an async CPU-offload load
(WAITING_FOR_REMOTE_KVS). Re-gating would leave its blocks idle.

**Why it helps:** it trades admission rate for preemptions.
- Measured: B32's replica had **~100-275 preemptions per 5-min window** at KV ≥ 0.88. It had no watermark (0.28 lineage
  default 0) and the OffloadingConnector.
- Manh's runs show **0 preemptions** at KV 0.88-0.93, with the waiting queue absorbing the pressure (up to 19
  waiting).
- Not isolated: our solver's KV gate (0.85 + queue hold) also acts in both.

**Lossless:** yes (scheduling only).

**Toggle:** `--watermark 0`.

**Benchmark:** the replay A/B, comparing preemptions, running count and gen tok/s in the KV-bound hours.

### E4. `--prefix-cache-retention-interval=128`: **does nothing on this stack**

- The setting is upstream (#52216; `CacheConfig.prefix_cache_retention_interval`, `config/cache.py` 275). It acts
  only on sliding-window and Mamba groups (`single_type_kv_cache_manager.py` `cache_blocks` 503, SWA mask 1297).
- With the V2 runner, upstream's `swa_bounded_replay=True` (`config/cache.py` 367, #56227) makes
  `SlidingWindowMLASpec.prefix_cacheable` False (`kv_cache_interface.py` 1102: `return not self.bounded_replay`).
  So the SWA group is never cached.
  - The compressed MLA and indexer K groups are full-attention groups, cached densely.
  - The compressor ring is never cached.
  - So 128 vs 512 changes nothing. The validator (`kv_cache_coordinator.py` 36-66) only needs a multiple of the
    scheduler block, which is why DCP4 needed 512.
- **Granularity instead:** hits end on a 128-token block, then the last 128-token window is recomputed
  (`prefix_replay_tokens`), about 128-255 tokens per hit.
- **On B32's stack** (no bounded replay; the vision11 base predates #56227) the interval *does* matter: SWA blocks are
  cached, and 512 gives 512-token hit granularity (E042 P11; "512 buys ~0.06 of KV headroom").
- Fork-only prefix envs (`VLLM_SWA_FINE_GRAINED_PREFIX_CACHE`, `VLLM_PREFIX_CACHE_HOLD_*`, `…_SKIP_RESUMED_TAIL`,
  `VLLM_KV_CACHE_GROUP_SIZE`; `envs.py` 1405-1477) are all off and target Mamba/GDN. Not relevant here.
- **Toggle:** remove the flag; expect identical pool size and hit rates. That is a cheap check.

### E5. SimpleCPUOffloadConnector (`--kv-offload-gib-per-rank 10`, eager: `lazy_offload: false`)

**What it is:** upstream #37160, fork-modified.
- Files: `distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` 80; scheduler side
  `v1/simple_kv_offload/manager.py` `SimpleCPUOffloadScheduler` 143; worker side `worker.py` 27, `DmaCopyBackend`.
- **Stores (eager):** `_prepare_eager_store_specs` (manager.py 679) stores every full, hashed, computed block about
  one step after it is computed (prefill and decode), plus boundary handoffs and the request's tail at finish.
  - Only prefix-cacheable groups are stored (compressed MLA KV + indexer K), not SWA. A host hit then replays the last
    window like a GPU hit.
  - Source blocks are pinned (`touch`) until all ranks confirm.
- **Loads:**
  1. `get_num_new_matched_tokens` (366) looks beyond the GPU hit and returns async; the request goes to
     WAITING_FOR_REMOTE_KVS.
  2. A background thread issues `cuMemcpyBatchAsync` on a lowest-priority stream, after the forward launch
     (`v1/worker/gpu/kv_connector.py` 187).
  3. Completion is polled with `event.query()`. The finished request rejoins through `kv_holding_waiting`, drained
     first.
  - est.: one or two extra engine steps per host hit.
- **Host memory:** `torch.zeros` pinned in place with `cudaHostRegister` (`worker.py` 240-249). That is 4 KiB pages,
  avoiding PyTorch's power-of-two pinned allocator, and it matches the pods' observation of no huge pages for the
  offload.
- **Fork changes:**
  - `_refresh_cpu_lru` (manager.py 995) re-touches CPU blocks the GPU still serves, so a long conversation's CPU prefix
    does not decay from its start;
  - `BlockPool.touch` cross-pool routing;
  - `kv_connector.disabled()` around graph capture and dummy runs.

**Why it helps:** when the GPU pool is full, re-sent transcripts come back over PCIe instead of being recomputed.

**Measured:**
- v12 replica: **19-98% of prompt tokens restored from host** in KV-bound windows, recompute 0.7-2.2%, **0
  preemptions**.
- Pod 2 (older cell, 120 workers): ~90% restored after GPU hits collapsed.
- B32's OffloadingConnector replica: 1.4-6.2k tok/s restored, plus 100-275 preemptions per window.

**Lossless:** yes (byte copies).

**Failure modes:**
- No `invalid_block_ids` / `failed_recving` reporting and **no timeout**.
- est.: the copy thread has no try/except, so a CUDA copy error kills it silently and requests hang in
  WAITING_FOR_REMOTE_KVS. Our solver's stall watchdog is the only guard.
- A preemption with pending stores triggers a blocking flush.
- Host RAM: 10 GiB/rank plus Engram's 64 GiB plus ~361 MiB of host-AR segments. That is ~127-131 GiB of server
  anonymous memory against Kaggle's 173 GiB.

**Toggle:** `--kv-offload-gib-per-rank 0`, or `--kv-offload-mode lazy`.

**Benchmark:**
- warm a prompt, `POST /reset_prefix_cache` (GPU only), resend, and compare TTFT and `prompt_tokens_cached_by_source`;
- metrics `vllm:simple_kv_offload_*`;
- est.: eager stores add steady device-to-host PCIe traffic that competes with the host AR, so eager vs lazy deserves
  an A/B.

## F. Runner, graphs, sampling, batching

### F1. CUDA graphs: `--compilation-config {"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64}` with `--max-num-seqs 64`

**What happens:**
- Capture sizes are [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], identical to B32's.
- On SM89 the DSv4.1 builders (Triton sparse MLA, SWA, compressor, indexer) declare UNIFORM_BATCH, so
  `resolve_cudagraph_mode_and_sizes` (`config/compilation.py` 1395-1419) **silently downgrades FULL to
  FULL_DECODE_ONLY**. The log says "…DeepseekSparseSWAFlashInferBackend … setting cudagraph_mode=FULL_DECODE_ONLY", as
  in B32 ("DeepseekV41IndexerBackend").
- **Any step with a prefill chunk runs fully eager, decode rows included.** Only pure uniform-decode batches replay
  graphs. That is the same as B32, so it is not a source of the speed-up.
- Upstream's `cudagraph_decoder_replay` (default on) captures separate graphs for the decoder-replay layers (F3). In
  the fork's log, capture took 10 s ("Capturing CUDA graphs (FULL): 2/2"), against B32's 79 s.
- `max-num-seqs` 64 = the graph cap = the solver's workers. At 80 (B32-M) the pool per stream is thinner, and decode
  steps above 64 would run eager.

**Memory:** 0.08 GiB of graph pool (v12 log), against B32's 0.12 GiB. See E2 for the profiler over-estimate.

**Toggle / benchmark:** none needed; FULL and FULL_DECODE_ONLY behave the same here.

### F2. `--enable-batch-sharded-sampling` (upstream #50465) and `VLLM_BATCH_SHARD_TOPK` (fork; set by the launcher to the profile's top_k)

**What it does:**
- **Off** (B32: present in its base, but not enabled): `compute_logits` all-gathers the vocab-parallel logits, so every
  rank holds [n, 129280] and samples the whole batch. est.: ≈ 16.5 MB received per rank at 64 rows.
- **On:** each request has an owner rank (`req_idx % TP`, `sample/batch_shard.py` 68).
  - Every rank computes its local-vocab logits (`compute_logits_local`, `nvidia/model.py` 2100);
  - `all_to_all_logits` (batch_shard.py 383, NCCL `all_to_all_single`) gives each owner full rows for its ~n/4
    requests (est. ≈ 4 MB per rank);
  - the owner samples, then `gather_sampler_output` (753) all-gathers the results.
- **Fork top-K:** if `Sampler.samples_from_top_k` (`sampler.py` 137) passes, each rank sends only its top-K
  (ids + fp32 bits) per row, ≈ 25 KB at 64 rows (`exchange_top_k_logits`, batch_shard.py 405). It passes only if
  every request in the step is greedy or has top_k ≤ K, with no logprobs, min_p, logits processors or thinking budget.
  - Unset top_k is stored as vocab_size (`states.py` 32, 54), so it does not pass.
  - **B32's solver sends temperature 1.0, top_p 1.0 and no top_k** (`pinned_candb/solver_dsv41.py:205`), so in our runs
    the top-K path never fires. The upstream all-to-all runs.

**Gain (est.):** 1-3 ms per decode step at 64 rows from the smaller exchange, plus 3/4 less sampling work per rank.

**Lossless:** greedy and top-k-only sampling give identical tokens (Gumbel noise keyed on seed, position and column).
With top-p, ties at the cut can differ. `VLLM_USE_FLASHINFER_SAMPLER=0` keeps tie order defined.

**Toggle:** the flag; `VLLM_BATCH_SHARD_TOPK=0`.

**Benchmark:** decode ITL at 64 concurrent, flag on vs off. A top_k = 64 request mix exercises the fork path.

### F3. Upstream features B32's 0.28-lineage stack lacks (automatic, no flag)

These came with the upstream merges, not from Manh's commits, but they are part of "his stack vs ours".

- **Decoder-side SWA bounded replay:** #58132, #59532 (graphs), #59894, #63155; `decoder_replay_layers.py`,
  `nvidia/model.py` 1174-1215.
  - Log: "layer 20 writes its KV for every token, and the rest of it and layers 21-39 run on each request's last 128
    tokens only".
  - In eager prefill steps, **19 of 40 layers process at most 128 rows per request.** est.: a single-request 2048-token
    chunk then costs ~55% of the full compute, so prefill runs ~1.8x faster before any comm change.
  - B32's base has only a prototype gated by `ced_prefill` in config.json. E128 does not set it, and B32's log has no
    replay line.
  - Lossy? The upstream design recomputes the window. E042 found the B32 prototype "not an equivalence
    transformation". The fork's accuracy (77.1 on 120 tasks, E062) covers it.
- **Encoder-side bounded replay / SWA not prefix-cached:** E4 above.
- **The 7ab756940a profile fix:** E2 above.
- **The V2 runner:** both stacks run `VLLM_USE_V2_MODEL_RUNNER=1`, so it is not a difference.
- `--enable-scale-out` (upstream `entrypoints/common/cli_args.py` 216) only adds HTTP routes (`/inference/v1/generate`
  and others). **No engine effect.**
- `--enable-prompt-tokens-details` only adds cached-token counts to usage.

### F4. `--max-num-batched-tokens 2048` (both stacks)

It sets the chunked-prefill step budget and the `profile_run` size, i.e. the activation peak subtracted from KV. A
2048-token chunk is ≥ 2 MiB per all-reduce (the 32x32 host-AR instance) and is split 2 x 1024 by A3. Lower values give
more KV and slower prefill. Unchanged from B32.

### F5. Misc env in the production cell

- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` (ours): upstream, default 300. It is the executor's RPC timeout and must
  exceed the 600 s host-AR trap.
- `VLLM_MAIN_STREAM_PRIORITY`: fork, default 0, not set; it is for GDN side streams.
- `VLLM_DCP_ASYNC_QUERY_GATHER=1`: DCP only (A2b).

## G. DCP: `--decode-context-parallel-size 4 --cp-kv-cache-interleave-size 64` (B32-M; **not** in production)

**What it shards.** Specs with `dcp_sharded=True` (`kv_cache_interface.py` 558):
- the main compressed C1/C2 caches and the indexer K caches, interleaved by `cp_kv_cache_interleave_size` tokens;
- the fork keeps the SWA cache (`sparse_swa.py`) and the compressor state (`compressor.py`) replicated.

**Per compressed layer (38 layers) per decode step** (`triton_sparse.py` 185-376):
1. Each rank packs its 16 heads' bf16 queries into fp8 rows + fp32 scale (`dcp_quantize_heads`, 516 B per token-head)
   and all-gathers them over pynccl. With `VLLM_DCP_ASYNC_QUERY_GATHER=1` this happens on a side-stream communicator
   during the indexer.
2. Each rank attends all 64 heads over its local shard plus a disjoint quarter of the SWA window, writing LSE.
3. Partials (fp8 + scale + LSE, 520 B per row) go to every peer by grouped send/recv, an all-to-all on the critical
   path.
4. `dcp_merge_partials` merges by LSE and applies the sink.

Each indexer layer also does a local top-k plus an all-gather merge (Triton `merge_dcp_topk`); layer 20 adds a
2048-block candidate merge. Under DCP, B1 and B2 are off.

**Constraints:**
- the interleave must divide block_size (`config/parallel.py` 452);
- it must be a multiple of the compress ratio (`indexer.py` 1239-1258), so the default of 1 fails;
- interleave / ratio must be a multiple of the candidate block 8. So 16/32/64/128 are valid; we used 64.

**Scheduler block = 128 x 4 = 512** (`resolve_dcp_kv_block_size`, `kv_cache_utils.py` 741; LCM over groups,
826-870):
- prefix hits align to 512 tokens (`single_type_kv_cache_manager.py` 102-110). est.: ~256 tokens of re-prefill per
  turn vs ~64 at 128;
- `--prefix-cache-retention-interval` must be a multiple of 512 (`kv_cache_coordinator.py` 36-65). Production's 128
  is rejected under DCP4.

**Why it is slow on PCIe without P2P:**
- est.: at B = 64, ~3 MB per rank per compressed layer, ~120 MB per step across 38 layers;
- 76+ extra NCCL-over-SHM collectives per step on top of the ~80 TP all-reduces, each with a latency floor. The
  partial exchange has no host path (E063: "no alternative path, so the autotune leaves it on NCCL").

**Measured:**
- 4090 no-P2P two-socket box: the fork with DCP4 ran at **0.51x B32** at x56 (E062).
- L4 pod 1: per-step fit ≈ 32 ms + 1.09 ms per stream (the fit table above), i.e. ~119 ms at 80. That ties B32's
  extrapolated step; the gain came only from the 2x pool (3.06M tokens, 0 preemptions).
- **Kaggle (B32-M V3 commit smoke, `experiments/E064_manh_notebook_b32_harness/kaggle_smoke_v3_load/`): 405-416 gen
  tok/s at 80 running**, against ~595 on the pod replica b32m_x2t at 70-97 running. That is **~0.70x** (the fix
  README's "0.72x"). Kaggle has PCIe Gen3 (and 50 W in the scored rerun; our commit smokes logged 72 W) and possibly a
  different GPU pairing, while DCP adds per-layer collectives.
  Kaggle's cold prefill was ~1,600 uncached tok/s.
- **B32-M's LB 46.5** is not explained by DCP speed: it fits a run that stopped answering at Kaggle hour 4-6. Its
  wheel 5e1bccd07c lacked the prefill-memory fix (E2), it had ~1.1 GiB VRAM free, and the solver's restart path burned
  tries (`kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/README.md`, `MANH_SAFETY_EDITS.md`).

**Lossless?** No. Queries and partials cross as fp8: `test_dcp_fp8_transport_matches_full` allows rtol 0.07 on peer
queries and a merged error below 0.04. Measured: NLL +0.004 per token vs B32, accuracy tied (E062).

**Memory:** the compressed bytes per token per rank drop 4x; SWA and compressor state stay replicated. Measured 1.22
vs 2.58 KB per token per GPU.

**Toggle:** drop both flags, and set retention back to 128 if wanted.

**Benchmark:**
- `tests/kernels/attention/test_dcp_indexer_topk.py` and `test_triton_sparse_mla_dsv4.py -k dcp`;
- end to end: the replay A/B with only these flags flipped.
- Manh reported (2026-10-29) running TP4 + **DCP2** with `--gpu-ids 0,2,1,3`, since by his report Kaggle's PHB pairs
  are 0-2 / 1-3; we have no Kaggle `nvidia-smi topo -m`. His production cell 2m (v12) passes no DCP flag, and DCP2 was
  never measured by us.

## H. `--supervise N` (his restart supervisor; `qwen38_server.py`, used by `dsv41_vllm.py main()`)

**What it does:**
- `supervise()` (qwen38_server.py:974) re-executes the launcher as a detached child in its own session
  (`start_new_session`), with stdout going to `supervisor.log`. It prints `Supervisor PID:` and returns once the first
  server pid exists.
- The child first stops a previous supervisor of the same port. It finds it by the recorded pid plus start time in
  `$TMPDIR/arc-supervisor-<port>.json` (`stop_previous_supervisor`, line 963), never by name.
- It then runs `run_supervisor()` (line 880). Per start:
  1. `free_port`: kill the port holder's whole session, unless that session is protected (the notebook kernel's own);
  2. `wait_gpu_free`: no nvidia-smi compute apps, 180 s;
  3. `launch()`: a fresh output dir, `server_restart<n>`;
  4. poll `/v1/models` until `dsv41` is listed (state "up", `up_s`);
  5. on exit, or no "up" within `--supervise-up-timeout`: stop the process, kill its session, and restart.
- It stops restarting when any of these holds:
  - N restarts are used;
  - start 0 failed before a process existed;
  - `now + backoff 10 s + last startup time > --supervise-deadline` (the cell passes ARC_STOP_NEW_S).
- Status JSON: start, state (starting / up / restarting / failed / stopped), pid, log, up_s, error, history. It is
  written atomically and read by our solver S26 as `ARC_EXTERNAL_SERVER_STATUS`.

**What it does not do:**
- It does not detect hangs. A wedged engine whose process stays alive is never restarted. Our cell-2s supervisor adds
  "25 min with no generated tokens while requests run"; see `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/`.
- It does not confirm a death twice.
- It does not wait for the old group's GPU memory beyond nvidia-smi's app list.

**Measured (pod 3 run 3; r1_rerun on pod 2):**
- One SIGKILLed TP worker → the API server exits on its own after 11-13 s → relaunch → "up" after 95-150 s. The
  outage was 116-174 s in total.
- The two engines' host RAM never overlapped.
- The cost is on the client side: the dying server answers every in-flight request with a plain HTTP 500, which
  solvers must treat as a connection loss (fixed in solver 1ea2b25d / S26).

**Risk:** the deadline rule uses the *last* startup time. A Kaggle restart is ~8-25 min under FUSE plus graph capture
and THP compaction (pod 3 saw a 1525 s load), so a late death still restarts if the previous start was fast.

**Toggle:** omit `--supervise`: one start, no restart.

**Test:** our kill tests (`per_layer/pod3_scripts/kill_tp.sh`, `phase_h/scripts/killtest.sh`).

## Ablation plan: one flag at a time, on a fixed workload

**Goal:** turn the ranking above into measured shares. It needs a 4x L4 pod rented by the user, and nothing on Kaggle.

**Fixed for every arm:**
- one pod, its GPU order (`--gpu-ids 0,2,1,3` as in production), and `nvidia-smi -q` topology/power recorded;
- model E128 (Kaggle copy) and wheel v11;
- context 163,840, max-num-seqs 64, chunk 2048;
- **KV pool pinned with `--num-gpu-blocks-override`** to the production pool. Memory-saving flags then change no
  concurrency; their memory effect is read separately from the startup lines ("Model loading took", "Available KV
  cache memory", "GPU KV cache size", "Capped the PyTorch allocator").

**Three workloads per arm** (all exist in the repo):
1. **Decode step:** `experiments/E042_dsv41_l4_synthetic/scripts/decode_window.py` at B = 16 / 32 / 56 / 64, 32k
   context, a 30-s window after warm-up. It yields `itl_median_ms`, the pure decode step with no prefill.
2. **Prefill:** 64 x 12,288-token random prompts with 1 output token (`vllm bench serve --random-input-len 12288
   --random-output-len 1`), plus cold 32 x 40k from phase H's `ab_decode/prompts_index.json`. It yields prefill tok/s.
3. **Shape-exact tool replay:** `E042 scripts/tool_replay.py` (returns generated ids, sends the recorded tool outputs
   and turn lengths with `ignore_eos`) from one fixed trace set (b32m_x2t or r1_rerun attempts), x64, 45 min, so that
   it enters the KV-bound phase.
   - It yields gen tok/s, per-stream at matched running count, preemptions, GPU-hit / host-restored / recomputed
     prompt tokens, and the fitted a + b·n step (`experiments/E062_manh_vllm_fork/per_layer/phase_h/scripts/fit_steps.py`).

**Also logged:**
- `pidstat -t` CPU per reducer thread and sandbox;
- GPU power and clock;
- the "GPU memory" lines.

**Numerics:** for every arm that can reorder sums (A1 vs NCCL, A3, C1, C2, F2), run `phase_h/scripts/ab_logprob.py`
(decode vs prefill logprobs, 32 prompts). The gate is mean |d| ≤ the stack's repeat noise (~0.022-0.037).

**Arms, leave-one-out from production P:**
- One change from P per arm, because several features depend on A1: prefetch, overlap and the inner instance need the
  host AR.
- Each arm is ~2-4 min of startup with warm compile caches, plus workloads 1-2 (~15 min), plus workload 3 for the
  Tier-1 arms.
- Run arms ABAB with P re-run in between for drift.

| tier | arm | change vs P | answers |
|---|---|---|---|
| 1 | P | none (production cell) | baseline; repeat it 2x for noise |
| 1 | −AR | `VLLM_HOST_ALLREDUCE=0` (also disables prefetch and overlap) | rank 1 in total |
| 1 | −PF | `VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=0`, or the PREFETCH_MODE_FILE flipped *on the same server* | prefetch vs pure transport |
| 1 | −ROW | `VLLM_DSA_DECODE_INDEXER_ROW_SHARD=0` | rank 4 |
| 1 | −BSS | drop `--enable-batch-sharded-sampling` | rank 7 |
| 1 | B32 | B32's stack, the same replay (and R1) | the total gap on one pod |
| 2 | −GATHER | `VLLM_TP_HOST_GATHER=0` | A2 |
| 2 | −WARM | MODE 44 | bit 512 without Q8 (never measured) |
| 2 | −PACK/SIMD | MODE 32 | packing and AVX-512 |
| 2 | THREADS=1 | `VLLM_HOST_ALLREDUCE_THREADS=1` | CPU load vs speed |
| 2 | −OVL | `VLLM_DSV41_AR_OVERLAP_CHUNKS=0` | A3 (prefill workload) |
| 2 | −TUNE | unset `VLLM_HUMMING_MOE_TUNING_CONFIG` | C2 (prefill workload) |
| 2 | −SPLOG | `VLLM_TRITON_SPARSE_INDEXER_LOGITS=0` | B1; add a 100k-context decode window |
| 2 | v10 | wheel v10 (the only difference is 71dca8bc24) | D1, at B ≤ 32 |
| 2 | −WM | `--watermark 0` | E3 (replay, KV-bound phase) |
| 2 | −OFF / lazy | `--kv-offload-gib-per-rank 0`, or `--kv-offload-mode lazy` | E5 |
| 2 | −PAD | pad 256,128 (pool pinned) | C1 speed alone |
| 3 | +B3 | `VLLM_DSV41_INDEXER_TP_SHARD=1/2` (unpinned pool) | KV gain and cost; loses B2 |
| 3 | +CGS0 | `VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES=0` (unpinned) | E2 KV gain |
| 3 | ret512 | `--prefix-cache-retention-interval 512` | confirm E4 is inert (same pool and hits) |
| 3 | −THP | `--engram-config.use_thp=false` | startup time and steady speed |

**Cost:**
- Tier 1 is 6 arms x (~5 min start + 15 min micro + 45 min replay), ≈ 6.5 h, or ≈ 9 h with the P repeats.
- Tier 2 runs micro workloads only, ≈ 20 min each, ≈ 4 h.
- Tier 3 is startup plus micro, ≈ 1.5 h.

**Kaggle transfer:**
- Repeat P vs −AR vs −PF with `nvidia-smi -pl 50` on the pod if the host allows it. Kaggle runs at 50 W, and its NCCL
  is ~2x slower than pod 1's, so A1's share is expected to grow there.
- est. ~1 h. Any Kaggle-side check needs the user's OK per run.

## Open items, doc errors and cheap wins found while reading

**Doc and comment errors:**
- `envs.py`'s bit-16 comment ("power-of-two scale per 32") is stale; the code uses an fp16 scale per 16.
- c2c8736696 calls the C2 caches "ratio-4", but the code pools 2 tokens per state.
- The notebook's "keep 512 MiB" comment is stale; the reserve is 256 MiB.
- Our E062 README's "24 MiB per rank" for the host AR understates it: ~120 MiB per instance, 3 instances.

**Inert or misleading flags in production:**
- `--prefix-cache-retention-interval 128` (E4);
- `cudagraph_mode FULL` (becomes FULL_DECODE_ONLY);
- `VLLM_DCP_ASYNC_QUERY_GATHER` (no DCP);
- `VLLM_BATCH_SHARD_TOPK` (never fires with our T = 1.0 / top_p = 1.0 requests; a top_k ≤ 64 would make it fire but
  changes the sampling distribution).

**Untried KV levers on the no-DCP config:**
- `VLLM_DSV41_INDEXER_TP_SHARD=2` (+13% pool, commit; it loses B2 on the sharded caches);
- `VLLM_MEMORY_PROFILER_CUDAGRAPH_SAMPLES=0` (est. +0.3 GiB/rank);
- MXFP4 indexer cache: not usable on SM89 in this fork ("does not support FP4 Q").

**Unquantified risks:**
- Host AR reducers can busy-spin up to 24 threads with no affinity. That has to fit the CPU quota next to 64 tool
  sandboxes, and whether AVX-512 is active is not logged.
- SimpleCPUOffload has no load timeout and no failure reporting.
- The host AR and host gather trap only after 600 s.
- THP compaction can make startup 3-25x slower on a fragmented host.
- Reserved VRAM creeps ~0.2 GiB/h, toward the allocator cap, at ~5-7 h.

**The fork's per-token KV footprint is 1.64x B32's** (B4: fp8 + bf16-RoPE compressed rows and an FP8 indexer, against
B32's FP4 rows and MXFP4 indexer). That is a capacity cost. It may also be part of why the fork's accuracy is no worse,
since the quantization is finer (est.; untested).

**Not covered here:** Manh's own `dsv41-vllm-prod.md`, which the cell cites, is not in our workspace. Neither is any
Kaggle-side measurement of the production stack's throughput: every production-stack number in this note is from pods.
