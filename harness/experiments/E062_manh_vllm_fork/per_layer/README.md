# Dynamic-count MoE on Manh's fork: tests and Kaggle KV pool (2026-10-29, Runpod 4xL4 pod 2)

Branch `dynamic-count-moe` (commit b3a3f642a9 on Manh's main 26c5906214). The patch is applied as a runtime patch to
his Kaggle wheel g5e1bccd07c, the B32-M stack. See `../README.md` "Dynamic-count MoE".

## Setup (pod 2, 5z3srm3108wcwl, restarted with an empty container disk)
- **Wheels:** Manh's environment as in E064b. All 294 wheels of `nguyennguyen599/vllm-wheels` v8 were downloaded
  sequentially and version-pinned in 1,191 s, with 0 failures. The vLLM wheel's sha256 is 85e05736…, as before.
- **Install:** `install_b32m.sh`, offline. torch 2.14.0+cu130, triton 3.8.0, vllm 0.32.1rc1.dev0+g5e1bccd07c,
  humming-kernels 0.1.18, Python 3.13.18.
- **Patch** (`pod_scripts/apply_patch.sh`):

  | File | Wheel sha256 | Patched sha256 | Change |
  |---|---|---|---|
  | `router/dsv4_topk.py` | 7e1e101a | ab32ea7e | branch hunk |
  | `deepseek_v41/nvidia/model.py` | ead1e087 | 616bb6aa | `../port/04b_per_layer_model_only.diff`, applies with offsets |

  The row-chunk plan-B patch (ead1e087 → b2a04fc2) is NOT applied: dummy loading doesn't need it. A real Kaggle-copy
  load needs both patches stacked on model.py.
- **Tokenizer files:** `tokenizer.json` (sha c90dfa01, = manifest), `tokenizer_config.json` and `encoding/`, from the
  Kaggle Model copy (v1). The HF token can't read the private chankhavu repo (404).

## Unit tests: all pass (L4, patched install)
- `tests/kernels/moe/test_topk_softplus_sqrt.py` (whole file): **1,720 passed, 10 skipped**. That includes the new
  `test_dsv4_topk_per_layer_expert_counts` (E 72-224; 1/80/2,048 tokens; with and without image rows) and
  `test_dsv4_topk_expert_count_limits`.
- `tests/models/deepseek_v41/test_per_layer_experts.py`: 5 passed.
- `pod_scripts/crosscheck_cuda.py`, the Triton `dsv4_topk` vs the CUDA `topk_softplus_sqrt` at E128, random fp32
  logits + bias, image rows: expert ids identical (same order) at 1/80/2,048 rows, weights within 1.2e-7.

## Pool check (dummy weights; B32-M's exact server command, util 0.938, DCP4, order 0,1,2,3)
- **Method:** `make_dummy_ckpt.py` writes config.json + tokenizer files + per-layer gptq files with e0 only.
  `--load-format dummy` then dummy-initializes every tensor at its real shape.
- **Server command:** B32-M's exact command from the E064b replica's `server_cmd.json` (`pod_scripts/server_args.txt`,
  `server_env.json`), without `--language-model-only` so the vision tower loads.
- **Validation:** text-only C1 with `--language-model-only` reproduces the real B32-M replica's pool **exactly**
  (3,058,602 tokens, 14.8 GiB/GPU).
- **Kaggle-equivalent:** pod × 1.0377 (Kaggle's 3,174,039 at util 0.945 / the pod's 3,058,602 at 0.938).
- **Synthetic load:** `vllm bench serve`, random 1,024 in / 512 out, 160 prompts at concurrency 80, ignore-eos.

| Layout (experts) | Load GiB/GPU | KV avail GiB | KV pool (pod) | Kaggle-eq | Peak out tok/s | Median ITL | Out tok/s (whole run) |
|---|---|---|---|---|---|---|---|
| C1 128 text-only (5,120) | 14.80 | 3.47 | 3,058,602 | 3.17M | 720 | 116.3 ms | 447 |
| C1 128 + vision (5,120) | 15.03 | 3.21 | 2,819,085 | **2.93M** | 720 | 116.2 ms | 447 |
| f72 cap160 avg120 + vision (4,800) | 14.62 | 3.58 | 3,150,222 | **3.27M** | 720 | 113.2 ms | 457 |
| f72 cap160 avg124 + vision (4,960) | 15.03 | 3.16 | 2,774,907 | **2.88M** | 720 | 114.4 ms | 453 |

- **Vision tower:** +0.23 GiB/GPU (TP-sharded ViT, FLASH_ATTN MM encoder). The encoder cache budget is 2,048 tokens,
  profiled with 2 images of the maximum feature size. Graph capture uses 0.19 + 0.34 GiB.
- **Host memory:** unreclaimable peak ~137-140 GiB in every run (Kaggle's limit is 173).
- **Answer: f72 avg124 is NOT >= 3.1M.** It is 2.88M Kaggle-eq, slightly below even uniform 128 + vision. f72 avg120
  reaches 3.27M. dsv4-infer's estimates were 2.7-3.0M (128 + vision, confirmed), 3.4-3.6M (120) and 3.0-3.3M (124).
  The two per-layer layouts come in lower because of the overhead below.

## Step-16 layout (fewer distinct counts), same flags + vision
- `f72c152s16_avg120`: 4,800 experts, 6 distinct counts (72, 88, 104, 120, 136, 152), max 152.
  text_config.n_routed_experts = 152 (= max), as dsv4-pruning's exporter will write it: **loads and serves**, so the
  export contract is OK.

| Layout | Distinct counts | Load GiB/GPU | KV avail GiB | KV pool (pod) | Kaggle-eq |
|---|---|---|---|---|---|
| f72c152s16 avg120 | 6 | 14.50 | 3.70 | **3,260,857** | **3.38M** |
| f72 cap160 step 8 avg120 | 11 | 14.62 | 3.58 | 3,150,222 | 3.27M |

- **What step 16 buys:** +110,635 tokens (+3.5%). The overhead over the linear prediction (14.30 GiB) falls from
  ~0.32 to ~0.20 GiB/GPU, so fewer distinct counts give back ~1/3 of it, not most.

## Overhead of per-layer layouts (found, partly located)
- **The expert parameters scale exactly linearly.** Logged in Humming's `process_weights_after_loading` (a temporary
  diagnostic, removed): 2.329 MiB per expert per GPU in every layer, at every count. f72_124 holds 11.282 GiB of
  expert params vs C1's 11.646 GiB.
- **The extra is elsewhere.** "Model loading took" (live allocated, measured after `empty_cache`) carries
  ~+0.32 GiB (f72_120) and ~+0.36 GiB (f72_124) on top of the linear prediction.
- **Diagnostic starts** (vision on, no load):

  | Layout | Experts | Load GiB | vs linear |
  |---|---|---|---|
  | per-layer list, all 128 | 5,120 | 15.03 | **= C1**: the patch path adds nothing |
  | 20×96 + 20×128 | 4,480 | 13.58 | **exactly linear** |
  | 20×160 + 20×96 | 5,120 | 15.18 | +0.15 |
  | 20×136 + 20×120 | 5,120 | 15.08 | +0.05 |

- **Partly located:** some is retained during Humming's weight conversion the first time each new expert count
  appears. That is 15-34 MiB per new count (E 88-152), vs ~4 MiB per layer in uniform C1.
- **Not driver memory.** The excess is in torch-allocated live tensors: "Model loading took" and the snapshots below
  both count torch blocks only. Non-torch extra, i.e. KV-available loss beyond the load difference, is only
  0.03-0.05 GiB in every run.
- **Torch memory snapshot at the end of loading** (a temporary diagnostic in `v1/worker/gpu/model_runner.py`,
  removed; `pod_runs/diag/snap_*.pkl`, `pod_scripts/snap_diff*.py`):
  - Total live is equal: f72_124 15.039 GiB vs C1 15.042 GiB.
  - The difference sits in Humming's `repack_weight` outputs (`humming/ops/utils.py:98 _prepare_output`): 12,543 MiB
    vs 12,453 MiB. f72_124 should be ~337 MiB lower there (160 experts × 2.109 MiB of w13+w2).
  - Both have 290 repack blocks. C1's are exactly 40 × 180 MiB (w13) and 40 × 90 MiB (w2), plus dense tensors.
  - f72_124's block sizes don't map one-to-one onto its per-layer counts. For example there are four 128 MiB blocks
    that fit no expert count, and the 68/108 MiB blocks expected for E 96/152 w2 are missing.
  - Each tensor also gets up to 2 MiB of caching-allocator rounding when its size isn't 2 MiB-aligned (C1's are
    aligned).
  - So the excess (~0.36-0.48 GiB) is in how Humming repacks or allocates non-uniform layers, not in vLLM's per-layer
    construction.
- **Still open:** the exact mechanism. Next step: log each layer's repacked tensor shapes (sublayer, E, shape, numel)
  in `_prepare_and_transform_sublayer`.
- **Without the overhead,** f72_124 would be ~3.13M and f72_120 ~3.6M Kaggle-eq.
- **Practical lever:** fewer distinct per-layer counts (steps of 16/32, or a few values) should cut it.
  Diagnostics with 2 distinct counts cost 0-0.15 GiB vs ~0.35 GiB with 10-11.

## Files
- `pod_runs/runs/<tag>/`: server.log, bench.log/json, gpu_host.log, summary.json, meta.txt.
- `pod_runs/diag/`: the per-layer param and alloc logs.
- `pod_scripts/`: apply_patch.sh, crosscheck_cuda.py, serve_dummy.sh, measure.sh, chain_pool.sh, server args/env.
- Secret scan: 101 files, 0 hits.

## Phase G: GPU order / DCP (C1 text-only dummy, B32-M's exact command, util 0.938)

| Arm | KV pool | Out tok/s (whole run) | Peak | Median ITL | Mean TTFT |
|---|---|---|---|---|---|
| DCP4, CUDA_VISIBLE_DEVICES=0,1,2,3 | 3,058,602 | 447 | 720 | 116.27 ms | 10.31 s |
| DCP4, 0,2,1,3 | 3,058,602 | 447 | 720 | 116.21 ms | 10.33 s |
| DCP2, 0,1,2,3 | **hung** at communicator init | - | - | - | - |
| DCP2, 0,1,2,3, `--disable-custom-all-reduce` | **hung** (same point) | - | - | - | - |
| DCP2, 0,2,1,3 | not run (DCP2 doesn't start) | | | | |

- **DCP4: GPU order makes no difference on this pod.** This pod's PHB pairs are 0-1 / 2-3, and NCCL here uses
  SHM/host paths.
- **DCP2 hangs** after the TP group's all-reduce backend selection (`cuda_communicator.py:377 ... group 'tp:0'`).
  The next step in a DCP4 start is the same line for group 'dcp:0', and it never appears. No error is logged in 16
  min; disabling custom all-reduce doesn't change it. Manh runs DCP2 with his own launcher on Kaggle, so the cause is
  probably a flag combination of B32-M's command with DCP2, or this pod's 2-GPU P2P/IPC path. Unresolved; needs his
  exact DCP2 command.
- So the DCP2 + `--gpu-ids 0,2,1,3` gain is NOT measured. Both DCP2 starts were stopped by process group.
