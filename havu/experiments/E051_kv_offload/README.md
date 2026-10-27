# E051: KV CPU offloading for DeepSeek-V4.1 in the SM89 vLLM fork

Status (2026-10-26 21:00 UTC): **GPU gates 1-3 passed on Vast 57816747 (4×4090)**: install, server start, byte-level
KV parity up to 120k tokens. Gate 4 (capacity A/B) running. Earlier: designed, patched and tested on CPU (26 unit
tests, mini-engine over the fork's real KV manager, mutation check).
mutation check); GPU parity and capacity untested. Not on the 24-h critical path for the projected-75 goal
(`experiments/E046_multi_test_rollout/goal75/`): once thrash is removed, more workers add ~0.2 pts in the simulator.

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
  - **Next: late-phase "Kaggle hour-8" arms** (dsv4-infer's E055 idea; options `--tail-frac` / `--min-start-ctx`
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
