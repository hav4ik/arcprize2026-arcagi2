# E051: KV CPU offloading for DeepSeek-V4.1 in the SM89 vLLM fork

Status (2026-10-27 13:00 UTC): **B+ done on a 4090 box; overlay v2 = v1 on GPU; 16 GiB host-capacity cliff
found, 32 GiB clears it at 64 streams.**
- v2 byte parity passed, and v2 tied v1 at 48 late-phase streams (968.6 vs 893.4 tok/s; the gap is a preemption burst,
  and preemption-free bins are equal).
- At 16 GiB both overlays collapse at 56 heavy streams (~170-216 tok/s, ~0% host hits): the in-flight context outgrows
  the 37.7k host blocks.
- At 32 GiB, v2 at 64 streams ran at 970.7 tok/s and 67.2 attempts/h.
- Recommendation for B/B+: 32 GiB (`cpu_bytes_to_use=34359738368`), plus a solver-side cap on in-flight context.

Status (2026-10-27 06:15 UTC): GPU work done; offload validated and handed to dsv4-infer for candidate B.
- Byte-level KV parity (shared and private memory, to 120k).
- Capacity A/B: 3.4-4.0× gen tok/s at 56-72 streams, 6.3× under hour-8 late-run load (within box).
- No overhead at light load; a 16 GiB host pool matched 32 GiB in throughput up to 48 streams.
- Recommended flag: `cpu_bytes_to_use=17179869184, shared_region=false, swa_store_policy=checkpoint`. Superseded by
  32 GiB above.
- Results: "GPU results" and "Host memory" below.
Earlier status: CPU-designed and tested (26 unit tests, mini-engine, mutation check).

- Why it failed on 10-24: the 8-row compressor ring group trips the block-size divisibility check
  (`offloading/config.py:57-64`, fixed upstream in #57145 after the fork's 2 Sep snapshot) and the scheduler's
  `FullAttentionSpec` assertion (`offloading/scheduler.py:125`).
- Design (`DESIGN.md`): never offload the ring; MLA+indexer dense (one 128-token block per host slot, 3,584 B/token for
  all 4 ranks); SWA as checkpoints only (`swa_store_policy=checkpoint`: the 4-chunk window at the prompt end and a
  rolling latest 128-aligned boundary; dense SWA would be 28x the MLA bytes); hits rounded down to 128 so a restore
  resumes where a GPU hit would. ~205 MB per 50k-token conversation; a 28 GiB budget holds ~3x the GPU pool.
- Patch: `overlay/vllm/` (4 full replacement files: offloading config, scheduler, cpu manager, cpu spec) and
  `patch/dsv41_kv_offload.diff`; byte-identical job logs with offloading off or at default options; installer
  `gpu/install_overlay.sh` (sha-checked, `--revert`).
- GPU plan (`GPU_TEST_PLAN.md`): parity (greedy ids + top-5 logprobs identical for GPU hit vs host restore), then
  capacity (tool replay at 36/56/72 streams with 1 s / 20 s tool idle, offload off vs on), then a 3-4 h pod soak.
- Effort to a submit-ready kit: ~4-7 days. Main risk: untested combination of the offload connector with async
  scheduling, Model Runner V2 and FULL CUDA graphs on a hybrid model. Phase 2: upstream #57652 single-copy layout
  (4x less host memory).

Handoff for a separate agent: `HANDOFF.md` (2026-10-26).

## GPU results (box 1 = Vast 57816747, 4×RTX 4090, PCIe Gen4, owned by E051 from 2026-10-26 19:50 UTC)

Box layout: `/root/dsv41/venv_offload` is a copy of the shared venv with only the 4 overlay files installed
(`gpu/install_overlay.sh`, `--revert` ready). The shared `/root/dsv41/venv` is untouched. **Gotcha:** a copied venv's
`bin/` scripts keep `#!/root/dsv41/venv/bin/python`, so `venv_offload/bin/vllm` would silently run the unpatched code.
The 53 shebangs were rewritten in the copy. All E051 files and runs are under `/root/e051/`. Host RAM is shared, so the
plan is at most half of 503 GB; `gpu/box_serve.sh` refuses to start above that.

- **Gate 1 (install), 19:58 UTC:** 26/26 CPU tests pass against the installed tree.
- **Gate 2 (server start + smoke):**
  - Kaggle server command plus `--kv-transfer-config` (28 GiB, `swa_store_policy=checkpoint`) and
    `--num-gpu-blocks-override 21490`; ready in 210-230 s.
  - Log: `KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups [8]`.
  - Engram pins 23.6 GiB per rank; the 30.06 GB shm region is mapped by all 4 ranks.
  - A 3,000-token prompt after a GPU reset hits 2,944 tokens from the host. 0 tracebacks, 0 preemptions.
- **Run-to-run nondeterminism (blocks the logprob gate):** two GPU-prefix-hit runs of the same prompt already differ at
  token 0 (logprob −4.436 vs −4.495). Two cold runs differ too, also with `VLLM_BATCH_INVARIANT=1`. So the planned
  bit-identical logprob check (`gpu/parity_test.py`) cannot pass on this stack, offloading or not
  (`gpu/diag_parity.py`).
- **Gate 3 (parity), replaced by a byte-level check: PASS.**
  - Tools: `gpu/kv_parity.py` + `gpu/e051_ext.py`, a worker extension loaded with
    `--worker-extension-cls e051_ext.E051Ext` and called via dev-mode `/collective_rpc`, test servers only.
  - Method: a host restore must put back exactly the KV pages a GPU prefix hit uses. The KV pool is filled with 0xA5
    before every restore, so a load that silently writes nothing cannot pass.
  - All 4 ranks, every compressed-attention block below the hit and each sliding-window group's 4-block checkpoint,
    byte-identical, 0 poison pages:

    | L | prompt case: hit (pages per rank) | resume case after a 300-token decode: hit (pages per rank) |
    |---|---|---|
    | 3,000 | 2,944 (51) | 3,200 (53) |
    | 20,000 | 19,968 (184) | 20,224 (186) |
    | 60,000 | 59,904 (496) | 60,288 (499) |
    | 120,000 | 119,936 (965) | 120,192 (967) |

  - Equivalence argument: a GPU hit and a host restore give the same kernels identical KV bytes and the same fresh ring
    and per-request state, so the remaining difference is the stack's own nondeterminism.
- **Transfers (Gen4):**
  - loads 1.81 GB in 0.42 s (~4.3 GB/s, ~7.5 ms per load), so a 120k-token restore takes ~0.12 s;
  - stores ~0.77 GB/s in the background;
  - lookup sync delay 0.09 ms mean.
- **Host memory:** Shmem 119 GiB (Engram 94 + offload 28); box-wide used 166 GB.
- **Gate 4 (capacity A/B), started 20:59 UTC:** `gpu/run_capacity.sh` runs `gpu/tool_replay_kv.py`, an E042 replay
  copy with offload accounting. Workload: E128 tool shapes, 600 s warm-up, 2,400 s window, pool 21,490 blocks.
  Arms: offload on vs off at 56 streams / 20 s tool idle, 72 / 20 and 36 / 1.
  - Workload file: the converted shapes `E042 metadata/trace_shapes/tools_e043_e128t.json` (334 attempts, `p0` +
    `turns`). `GPU_TEST_PLAN.md` named E043's raw `tool_shapes_*.json`, which the replay cannot read; the first launch
    (cap1) failed on it within seconds.
  - Driver: `cap2`. Box 1 runs offload on; box 3 (Vast 57867594, rented by the user for E051's off arms) runs offload
    off with the same patched venv, inert without a connector. Smoke runs (4 streams, seed 1): 253 vs 267 tok/s.
  - **B pair, 56 streams / 20 s tool idle** (40-min windows):

    | | off (B0, box 3) | on (B1, box 1) |
    |---|---|---|
    | gen tok/s | 220.2 | **742.2 (3.37×)** |
    | GPU prefix hit | 0.0% | 18.1% (+ 20.98M tokens restored from host) |
    | true uncached tok/s | 3,344 | **394 (8.5× fewer)** |
    | uncached per turn, mean / p99 | 34.3k / 103.9k | 1.3k / 14.4k |
    | turns with > 20k uncached / full re-prefills (≥ 50k, ≤ 640 cached) | 148 / 47 | 7 / 1 |
    | TTFT p50 / p99 | 40.5 / 399 s | 1.3 / 7.8 s |
    | turn p50 / p99 | 290 / 3,463 s | 59 / 1,591 s |
    | turns / attempts in window | 231 / 16 | 898 / 62 |
    | KV mean / peak (window) | 0.88 / 0.997 | 0.72 / 0.94 |
    | preemptions | 0 | 0 |
    | host traffic in / out | – | 34.7 / 83.6 MB/s (2,692 loads, 3.7 ms and 30.9 MB each) |
    | host memory | Shmem 91 GiB | Shmem 119-121 GiB, server PSS 133 GiB |

    - Without offload the 56-stream load thrashes completely: every turn re-prefills, and the slower turns lengthen
      the idle gaps, which evicts more. With offload the effective prefix hit (GPU + host) is ~96.5%.
    - This passes the plan's decision rule (more than 50% fewer uncached tokens, at least 5% more tok/s) and the
      handoff's ≥ 5× re-prefill reduction by a wide margin.
  - **C pair, 72 streams / 20 s tool idle:**

    | | off (C0, box 3) | on (C1, box 1) |
    |---|---|---|
    | gen tok/s | 208.5 | **842.3 (4.04×)** |
    | GPU hit / host-restored tokens | 0.0% / – | 0.3% / 31.5M |
    | true uncached tok/s | 3,379 | 493 |
    | turns with > 20k uncached / full re-prefills | 155 / 54 | 9 / 2 |
    | TTFT p50 / p99 | 171 / 488 s | 3.0 / 51 s |
    | turns / attempts in window | 231 / 17 | 1,044 / 72 |
    | KV mean / peak | 0.965 / 0.997 | 0.929 / 1.0 |
    | running mean / waiting mean (max) | 55.7 / 14.5 (20) | 60.9 / 3.0 (43) |
    | preemptions | 0 | **541** |
    | host in / out | – | 80.4 / 95.4 MB/s (4.1 ms, 33.6 MB per load) |

    - With offload, throughput still rises from 56 to 72 streams (+13%). At 72 the running decode set overflows the pool
      (541 preemptions): offload cannot hold running contexts, so the useful concurrency is ~56-64 on this pool.
    - Without offload, throughput collapses to ~210 tok/s from 56 streams up.
  - **A, 36 streams / 1 s tool idle (overhead check; offload restores nothing at this load):**

    | | gen tok/s | GPU hit | stores | KV mean | turn p50 |
    |---|---|---|---|---|---|
    | off, box 3 (A0) | 814.7 | 96.4% | – | 0.49 | 38.6 s |
    | on 28 GiB shared, box 1 (A1) | 585.8 | 96.3% | 66 MB/s | 0.49 | 53.9 s |
    | **on 32 GiB private, box 3** | **844.9** | 96.3% | 96 MB/s | 0.47 | 35.8 s |

    - Within one box, offload costs nothing measurable (+3.7%, noise).
    - **Box effect:** box 1 is ~28% slower than box 3 at 36 streams, but only 5% slower at 4 streams. Its load average
      was 11-13, probably other tenants' CPU contention on the CPU-bound scheduler. So the cross-box B and C gains
      (on = box 1) are conservative. All later comparisons are kept within one box.
    - CPU benchmark of the connector's scheduler side (`tests/bench_scheduler.py`, 36 requests × 40k tokens, decode):
      3.3 ms per step. Of that, 1.9 ms is `update_offload_keys` (an `islice` from 0 on every step, O(context) per
      request) and 1.3 ms is store planning. Optional fix: slice the list directly. Not applied, to keep the overlay
      frozen.
    - Tools for later profiling: in-process stack sampler `gpu/sampler/sitecustomize.py` (`E051_SAMPLE=1`). The
      containers have no ptrace, so py-spy cannot attach.
  - **Late-phase "Kaggle hour-8" arms: results.** Settings: heaviest 30% of shapes ×1.25, starts at ≥ 30k
    tokens, 20 s idle, 600 s warm-up + 1,500 s window, pool 21,490 blocks.

    | box | arm | streams | gen tok/s | attempts/h | turns/h | GPU hit | host tokens | full re-prefills | preemptions | TTFT p50 / p99 | KV mean |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 1 | off | 36 | 96.0 | 7.2 | 115 | 0.0% | – | 30 | 0 | 148 / 1,163 s | 0.966 |
    | 1 | off | 44 | 95.5 | 9.6 | 101 | 0.0% | – | 31 | 0 | 533 / 1,421 s | 0.957 |
    | 1 | off | 48 | 93.0 | 9.6 | 103 | 0.0% | – | 32 | 0 | 675 / 1,571 s | 0.961 |
    | 1 | on 28 GiB private | 36 | **602.7** | 43.2 | 811 | 37.4% | 10.2M | 4 | 55 | 0.65 / 197 s | 0.78 |
    | 3 | on 32 GiB private | 36 | **874.0** | 55.2 | 1,147 | 16.2% | 19.3M | 2 | 0 | 0.57 / 5 s | 0.685 |
    | 3 | on 32 GiB private | 44 | 893.1 | 60.0 | 1,217 | 0.0% | 25.2M | 2 | 47 | 0.65 / 49 s | 0.838 |
    | 3 | on 32 GiB private | 48 | 942.7 | 64.8 | 1,284 | 0.0% | 27.1M | 3 | 85 | 1.45 / 77 s | 0.934 |

    - Without offload the late heavy load thrashes at every concurrency tested (~93-96 tok/s flat).
    - Offload: within box 1 at 36 streams, 6.3× gen tok/s and 6× attempts/h.
    - On box 3, throughput keeps rising slowly with streams (+8% from 36 to 48), while preemptions and p99 TTFT grow.
      Late hours fit ~36-40 running heavy contexts on the pool; preempted streams come back from host instead of being
      recomputed. So a gate that also holds while the server has waiting requests (dsv4-infer's S17) is the right
      control.
    - Off at 16/24 streams was skipped at the user's request to scale the 4090s down; the off curve is flat, and
      dsv4-infer has L4 evidence for its no-offload fallback.
    - **16 vs 32 GiB (box 3, private memory, same hour-8 load):**

      | streams | 16 GiB tok/s | 32 GiB tok/s | attempts/h (both) | full re-prefills 16 / 32 | preemptions 16 / 32 | TTFT p99 16 / 32 |
      |---|---|---|---|---|---|---|
      | 36 | 895 | 874 | 55.2 | 2 / 2 | 0 / 0 | 4 / 5 s |
      | 44 | 930 | 893 | 60.0 | 2 / 2 | 101 / 47 | 41 / 49 s |
      | 48 | 948 | 943 | 64.8 | 3 / 3 | 71 / 85 | 73 / 77 s |

      16 GiB is as good as 32 GiB up to 48 heavy streams, and Shmem is 107 vs 123 GiB. **Recommendation:**
      `cpu_bytes_to_use = 17179869184` (16 GiB, an exact power of two per rank), `shared_region=false`. **Superseded (B+, 13:00 UTC):** 16 GiB has a host-capacity cliff at
      ~56 heavy streams; use 32 GiB (see overlay v2 / B+ below).
    - A first chained launch of the 16 GiB arms (late3b) misfired, because its waiter's
      pattern "ALL DONE" matched another driver's "wait over: 'ALL DONE'" line; the GPU-busy check stopped it, and
      the driver no longer echoes the pattern.
  - Box 1 was wound down at 04:23 UTC (`gpu/stop_box1_after_x36.sh`); box 3 finished at 06:10 UTC. Runs:
    `runs/box1_57816747/`, `runs/box3_57867594/` (secret scan clean). Both boxes were handed back to dsv4-infer to stop.
  - Original plan for the hour-8 arms (dsv4-infer's E055 idea; options `--tail-frac` / `--min-start-ctx`
    merged into `gpu/tool_replay_kv.py`). Settings: heaviest 30% of shapes ×1.25, starts at ≥ 30k tokens, 20 s idle,
    36 / 44 / 48 streams, private host memory (`shared_region=false`).
    - Box 1: private-memory parity, then off and on 28 GiB (`gpu/chain_box1b.sh`, TAG late1).
    - Box 3: on 20 GiB and on 28 GiB (TAG late3).
    - These replace the planned uniform heavy pair and the 56/20 20-vs-28 runs.

## Host memory on Kaggle: pool size and accounting (2026-10-27)

- **Private pinned memory is rounded up to a power of two per rank.** With `shared_region=false` each rank's buffer is
  `torch.zeros(..., pin_memory=True)` (fork `v1/kv_offload/cpu/gpu_worker.py:792`). That goes through PyTorch's
  CachingHostAllocator, which rounds every pinned allocation up to the next power of two: in torch 2.13
  `pinned_max_round_threshold` and `pinned_max_cached_size` default to SIZE_MAX (`ATen/core/CachingHostAllocator.h:300-308`).
  - Real host cost ≈ 4 × PowerOf2Ceil(cpu_bytes_to_use / 4): 28 GiB → 32 GiB, 20 GiB → 32 GiB, 16 GiB → 16 GiB,
    32 GiB exactly (34359738368) → 32 GiB. Slots are floor-divided, so a rank never exceeds its share.
  - Measured on the L4 candidate replica (hegn, 28 GiB private): used 178 → 212 GiB, process RSS 105.9 → 139.5 GiB
    (+33.6 GiB) when the pool came up.
  - **Use 32 GiB (same RAM as 28, +14% cache) or 16 GiB; never 20-28.** Shared mode (`/dev/shm` mmap +
    `cudaHostRegister`) is exact: box 1 Shmem rose by 28-30 GiB for 28 GiB.
- **What Kaggle's cgroup (memory.max 173 GiB) counts:**
  - Both the Engram table (~94 GiB pinned) and the private offload pool are pinned, unreclaimable host memory. On hegn
    both appear under Shmem in /proc/meminfo: 123 GiB = Engram ~91 + offload 32. Both count against memory.max,
    together with the processes' anonymous memory.
  - Page cache (model files read from /kaggle/input) is charged too, but it is reclaimable, so it need not fit.
  - Budget: V6 runs at ~105-123 GiB + ~34 GiB for a 32 GiB pool ≈ 139-157 GiB vs 173 GiB; with 16 GiB, ≈ 122-140 GiB.
    dsv4-infer logs candB's host memory over the 12 h (mem.log) to check for creep.

## Long-run host memory (server process group PSS, excluding the first 10 min)

| run | duration | PSS start → end (GiB) | Shmem |
|---|---|---|---|
| on 28 GiB shared, box 1 | 3.3 h | 131.3 → 134.9 | flat 119 |
| on 32 GiB private, box 3 | 3.1 h | 135.6 → 139.0 | flat 123 |
| on 16 GiB private, box 3 | 2.0 h | 119.3 → 121.4 | flat 107 |
| off, box 3 | 3.0 h | 103.6 → 106.3 | – |
| off, box 1 | 3.2 h | 102.5 → 102.8 | – |

- **The growth is in the 4 workers and is slowing down:** per-process snapshots at 2.4 and 2.7 h
  (`runs/box3_57867594/late3c_on16p/pss_snap{1,2}.txt`) show each worker's anon at 2.73 → 2.75 GiB, while EngineCore
  and the API server stay flat. That is ≈ 0.3 GiB/h total, down from ~1 GiB/h in the first 2 h. 12 h projection:
  +4-6 GiB.
- **Code audit of the offload path, no unbounded allocations after startup:**
  - Transfer descriptor buffers are pooled pinned int64 arrays (KBs).
  - No GPU allocations per transfer (cuMemcpyBatchAsync between preallocated pools); streams and events are pooled.
  - The host LRU index is bounded by the slot count (~37k slots at 16 GiB, ~75k at 32 GiB).
  - Per-request key state is deleted after the last store job, the discard queue is capped at 65,536 keys, and KV
    events are off.

## Overlay v2: incremental `update_offload_keys` (2026-10-27, for candidate "B+")

- **What changed:** `RequestOffloadState.update_offload_keys` runs for every scheduled request on every step. v1 (and
  the fork) islice `block_hashes` from index 0, which is O(context) per call. v2 indexes the append-only list directly
  (O(new hashes)) and skips groups with no new hash after a length check. That is the only change: one function in
  `offloading/scheduler.py`, plus dropping the now-unused `islice` import. Diff: `patch/dsv41_kv_offload_v1_to_v2.diff`.
  Candidate B's frozen kit (`experiments/E056_candidate_b/kit`) is untouched.
- **Files (`overlay_v2/vllm/`), sha256:**
  - `offloading/scheduler.py` **94b06f850d3706d154d7d51b835e913af923cee1b5784bed0825c96ba489c8fa** (v1: dffd7d86…)
  - `offloading/config.py` 664b4b01…, `cpu/manager.py` eda86f25…, `cpu/spec.py` a0dbbafc… (unchanged from v1)
- **Install:** `gpu/install_overlay_v2.sh VENV` installs over the wheel original or upgrades a v1-patched venv in
  place, and refuses anything else; `--revert` restores the wheel originals. `OVERLAY=v2 gpu/setup_offload_venv.sh`
  does the venv copy, install and tests. All paths were tested on a throwaway venv.
- **CPU tests:**
  - The 26 v1 tests pass on the v2 tree.
  - New `tests/test_dsv41_kv_offload_v2.py` (5 tests, pass):
    - Randomized histories on the real 9-group layout: 60 seeds, more than 2,000 calls. Prompts of any length; growth
      by 0, 1 or a partial chunk, or crossing a chunk; repeated calls; calls through `_update_req_states` with the
      preempted/resumed flag. After every call the keys equal both the verbatim v1 code and a from-scratch rebuild.
    - The ring group never gets keys.
    - End to end, multi-turn MiniEngine workloads with checkpoint and dense SWA policies emit identical store and load
      job sequences under v1 and v2 key builders, including host restores that pass the content check.
  - **Mutation check** (`scripts/mutate_v2.py`): 3/3 mutants caught (start off by one, step ignored, stale length).
  - **Default behaviour** (`tests/diff_default_behavior.py`): v1 and v2 digests are identical.
- **Scheduler-side connector cost per decode step** (`tests/bench_scheduler.py`, CPU, 256 steps):

  | requests × context | v1 ms/step (mean / p99) | v2 ms/step (mean / p99) |
  |---|---|---|
  | 36 × 40k | 2.45 / 2.57 | **0.59** / 0.67 |
  | 36 × 80k | 4.45 / 4.59 | **0.71** / 0.70 |
  | 56 × 40k | 3.88 / 4.39 | **0.93** / 1.01 |
  | 56 × 80k | 6.85 / 7.00 | **1.11** / 1.04 |

  - v2 is 4-6× cheaper and no longer grows with context. The max (17-55 ms in both) is a rare step with a large store
    batch.
  - On the L4s (~90-110 ms decode steps at 48-64 streams) this removes ~3-6 ms of scheduler time per step at 56-64
    long contexts; how much of it was on the critical path with async scheduling is for the GPU A/B to show.
- **GPU, 4090 box (Vast 57929406), candidate B flags** (`gpu/serve_bplus.sh`: retention 512, util 0.92, pool
  21,490 blocks, 16 GiB private; driver `gpu/run_bplus.sh`):
  - The v1 arm is `venv_candb`. The v2 arm is `venv_candbp`, a copy with `install_overlay_v2.sh` applied; 31/31 CPU
    tests pass on it.
  - **Byte-level KV parity with v2: PASS.** All 8 cases (prompt and resume at 3k/20k/60k/120k): 0 mismatches,
    0 poisoned pages, 0 tracebacks. Resume hits land on 128-aligned boundaries (for example 3,200) under retention 512.
  - **Host-capacity cliff at 16 GiB (v2, 56 late-phase streams):**
    - 170.6 gen tok/s and 16.8 attempts/h. External hits were 232k tokens over the window, vs 27M on box 3 at 48.
    - 48/69 turns were full re-prefills. Running about 30, waiting about 26; TTFT p50 622 s.
    - Store totalled 204 GB, load 3 GB. No preemptions.
    - Cause: `cpu_bytes_to_use` is the total over the 4 ranks, so 16 GiB = 16 GiB / (113,920 B × 4) = 37.7k host
      blocks, about 4.8M MLA tokens.
    - The late-phase prompts have a p50 of 82.5k (p90 114.5k), so 56 streams need about 36k blocks, plus 28 SWA
      checkpoint blocks per turn.
    - The first queue cycle already overflows the host. Under FCFS waiting plus LRU, each waiting stream's prefix is
      the oldest entry and is evicted before the stream is admitted. That stream re-prefills and re-stores, evicting
      the next one: a self-sustaining loop with about 0% hits.
    - 48 streams (about 31k blocks) fit, and box 3 got 948 tok/s there with 16 GiB.
    - So offload stops helping, abruptly, once the in-flight total context passes about 4.5M tokens per 16 GiB.
    - 32 GiB (8 GiB per rank, a power of two) doubles the cliff.
    - v2 at 64 was stopped, since it was deeper in the cliff.
  - Re-planned A/B below the cliff (`gpu/bplus_chain2.sh`): v2 @ 48, v1 @ 48, then v1 @ 56 as the control showing the
    collapse is capacity and not v2. Then, at dsv4-infer's request, v2 @ 64 @ 32 GiB (`gpu/bplus_chain3.sh`),
    B+'s likely config.
  - **v1 vs v2 at 48 streams (same box, 16 GiB): a tie.**

    | arm | gen tok/s | attempts/h | turns/h | preemptions | host-hit tokens | full re-prefills | TTFT p50 / p99 |
    |---|---|---|---|---|---|---|---|
    | v1 (`venv_candb`) | 968.6 | 64.8 | 1,310 | 139 | 27.5M | 3 | 1.6 / 81 s |
    | v2 (`venv_candbp`) | 893.4 | 62.4 | 1,195 | 354 | 25.5M | 3 | 1.7 / 93 s |

    - Gen tok/s and preemptions per 300 s bin:
      - v1: 873/84, 902/9, 1020/4, 1029/42, 1010/0.
      - v2: 820/135, 719/150, 912/68, 1018/0, 997/1.
    - Preemption-free bins are equal (1000-1030 tok/s). The 8% gap is v2's preemption burst early in its window.
      v2's keys are proven identical to v1's, so v2 cannot change scheduling decisions; the burst is run-to-run KV
      pressure variation.
    - Shmem 107 GiB and PSS 121-122 GiB in both arms; 0 tracebacks.
    - **Verdict:** v2 is safe but is not a throughput lever on this box. The ~5 ms/step scheduler CPU saving is off
      the critical path. Take v2 only if the kit is rebuilt anyway (for example for 32 GiB), as CPU headroom for
      Kaggle.
  - **Control, v1 @ 56 (16 GiB): the cliff reproduces without v2.**
    - 215.7 gen tok/s, 21.6 attempts/h.
    - 2.7M host-hit tokens; 55 full re-prefills out of 100 turns.
    - Waiting 25.4, TTFT p50 587 s, 48 preemptions.
    - v2 @ 56 had 170.6 tok/s and 0.23M host-hit tokens, the same regime. The cliff is host capacity, not the overlay.
  - **v2 @ 64 @ 32 GiB (B+'s likely config): healthy.**
    - 970.7 gen tok/s, 67.2 attempts/h, 1,274 turns/h.
    - 28.1M host-hit tokens; 3 full re-prefills out of 531 turns.
    - 498 preemptions. TTFT p50 54 s, p99 150 s. Waiting 19.3 on average.
    - Shmem 123 GiB, PSS 138 GiB; 0 tracebacks.
    - 64 × 82.5k ≈ 41k blocks, well under the 75.4k host blocks.
    - Throughput plateaus at ~970 tok/s: about 38 heavy streams run on the GPU pool. Going from 48 to 64 streams adds
      only +4% attempts/h, and median TTFT rises from 1.6 s to 54 s.
    - So late-phase concurrency above ~48 buys little. 32 GiB is insurance against the cliff, not a throughput gain.
  - Runs: `runs/box4090_57929406/` (`bplus_v2`, `bplus_v2b`, `bplus_v1`, `bplus_v2x32`; secret scan clean). The box
    was handed back to dsv4-infer at 13:00 UTC.
  - **Host-pressure signals** (180 s windows with waiting > 0):

    | run | host-hit ratio Δext_hits/Δext_queries, p5 / p50 / p95 | windows < 0.5 | longest low run |
    |---|---|---|---|
    | box 3, 16 GiB, 44 streams (healthy) | 0.00 / 0.98 / 0.99 | 15.9% | 182 s |
    | box 3, 16 GiB, 48 streams (healthy) | 0.00 / 0.98 / 0.99 | 16.6% | 249 s |
    | 4090, 16 GiB, v2, 56 streams (cliff) | 0.00 / 0.00 / 0.18 | 100% | 3,116 s |

    - Healthy low windows are new attempts' cold prefills.
    - Load/store byte ratio: about 0.6-1.1 when healthy, below 0.06 in the cliff.
    - `vllm:kv_offload_cpu_cache_usage_perc` is not occupancy. It counts only pinned (in-flight) blocks and read 0.2%
      in the cliff.
    - Suggested to dsv4-infer:
      - Primary: a solver-side cap of in-flight total context ≤ 0.6 × host tokens (cpu_bytes / ~3,584 B per token).
      - Secondary: a reactive hold on admissions while the ratio stays below 0.5 for 360 s or more with waiting > 0.
- **Still to do:** the 12 h L4 replica (dsv4-infer).

