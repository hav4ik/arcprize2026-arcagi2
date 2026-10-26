# E053 sub-report: upstream SM89 support for V4.1, overlay port, versions (fork worker, 2026-10-26)

Paths: U = scratchpad/e053/up_main/src (upstream main af6665b5dd20, 2026-10-26 19:34 UTC); F = scratchpad/src_vllm/vllm
(fork vision11 wheel source); O = kaggle/dsv41_tools/vllm_overlay_src/vllm.

## 1. Fork base

- Fork main merged upstream at 1b60fdb6bc5a ("Merge upstream Qwen4Exp support", 2026-09-02 14:18 UTC). Its 2nd parent is
  upstream **2a4e3cc3dbdd (2026-09-02 11:09 UTC, #54759)**. That commit is 305 commits after the v0.28.1rc0 tag (79651d60,
  2026-08-27); 305 upstream + ~212 fork commits ≈ dev517. (gh compare v0.28.1rc0...2a4e3cc3: ahead 305.)
- Upstream main is now **3,088 commits ahead** of that base (gh compare 2a4e3cc3...main).
- The Vision11 wheel was built from fork commit 86d35745 (2026-09-15; release body). Fork pins = upstream v0.29.0's pins
  (flashinfer 0.6.18, cutlass-dsl 4.6.2, quack 0.6.4, tokenspeed-mla 0.1.8, humming 0.1.12, torch 2.13.0).
- Fork's V4.1 is its own implementation: c897ad5a (2026-09-12) adds 19,396 lines in 116 files (8,346 in
  vllm/models/deepseek_v4_1, 1,720 in model_executor/kernels, 1,041 in layers). Upstream's V4.1 model definitions landed
  2026-09-10 (#56228) and were renamed deepseek_v4_1 -> deepseek_v41 on 09-14 (#56741).

## 2. Upstream DeepSeek-V4.1 on SM89: does not start

| component | upstream main | SM89? |
|---|---|---|
| V4.1 attention class selection | U/vllm/models/deepseek_v41/nvidia/model.py:227-255: SM12 -> FlashInfer SM120 class, else MegaAttn (SM100 only) else `DeepseekV4FlashMLAAttention` | falls to FlashMLA |
| FlashMLA sparse backend | U/vllm/models/deepseek_v41/sparse_mla.py:150-151 `capability.major in [9, 10]`; build U/cmake/external_projects/flashmla.cmake:55-71 archs 9.0a/10.0f/10.7f | **no** |
| FlashMLA mega attention | sparse_mla.py:305-306 `major == 10`; nvidia/flash_mla_mega_attn.py:98 | no |
| FlashInfer sparse DSV41 | U/.../nvidia/flashinfer_sparse.py:137-138 `major in [10, 12]`; uses `flashinfer_trtllm_batch_decode_sparse_mla_dsv4` (:29) | no |
| sparse indexer | U/vllm/model_executor/layers/sparse_attn_indexer.py:1060-1064 raises unless DeepGEMM; DeepGEMM `support_deep_gemm` = SM90 / SM100 / SM120 only (U/vllm/platforms/cuda.py:691-697; deepgemm.cmake:71-79) | **no** |
| O projection (fp8 einsum) | U/vllm/models/deepseek_v4/nvidia/ops/o_proj.py:28-33 `deep_gemm_fp8_o_proj` | no (DeepGEMM) |
| DeepSelect prefill top-k (#62240) | U/cmake/external_projects/deepselect.cmake:33-38: 10.0f only | no |
| mHC | fast paths SM100 (nvidia/ops/mhc.py:50, mega_mhc.py:21); tilelang + torch fallback (U/vllm/model_executor/kernels/mhc/torch.py) | probably (untested) |
| Engram | cpu_offload default True (U/vllm/config/engram.py:37); tuned thresholds only for SM100 (common/engram.py:94) | yes (generic) |
| MoE MXFP4 / Humming / Marlin | `"humming": [Mxfp4MoeBackend.HUMMING]` (U/.../fused_moe/oracle/mxfp4.py:377) | yes |
| MXFP8 dense Humming | `--linear-backend humming` includes HummingMxfp8LinearKernel (U/.../kernels/linear/__init__.py:324-331) | yes |
| router 128 experts | fast Triton `dsv4_topk` only for 256/384 experts (U/.../fused_moe/router/dsv4_topk.py:26); 128 uses generic fused_topk_bias | yes (generic path) |
| specialized decode | SM103 (GB300) only (nvidia/specialized_decode/__init__.py:83) | no |

- No SM89/Ada string anywhere in U/vllm/models/deepseek_v4{,1} or the indexer (grep). The V4.1 tracking issue #56400 lists
  verified H200, GB200, GB300, MI350X (quoted in issue #57144). Release notes v0.30-v0.32 mention V4.1 only for SM100/SM90;
  v0.32's "NVFP4 KV on SM8x" (#46963) is for generic backends, not DSv4.
- The upstream wheel itself does build generic `_C` for 8.9 (U/CMakeLists.txt:123-134; U/docker/Dockerfile:274), so a
  source build is not needed for the generic kernels. The V4.1-specific kernels are the gap.
- Open upstream PRs for pre-SM90 DSv4 (none merged): #55184 (software-FP8 + sparse-MLA omnibus), #55177, #47629 / #38476
  (TRITON_MLA_SPARSE for SM8x), #52955 ("Sm89", empty). #56120 (portable Triton fallbacks for Ampere/Ada) closed unmerged 09-24.
- **Community branch zenglanmu/vllm `dsv41-sm80`** (last commit 2026-10-21): upstream main fce64bc5c2 (2026-10-20) + #55184
  + thomaslwang's `dsv41-ampere`; 66 commits, 122 files, +16,130 lines. It reports V4.1-Flash-1017 running end to end on
  8xA100 (issue #57144 comment, 2026-10-20) with prefix caching, CUDA graphs, Engram cpu_offload. Its Ampere attention is
  selected for **any** `major == 8` (its nvidia/model.py:251, :259), so it would also be picked on SM89; its
  TRITON_MLA_SPARSE accepts `major >= 8` (triton_mla_sparse.py:187-189). It is BF16 Triton sparse MLA (no FP8 MMA) and
  is untested on SM89. Its last commits are "sm80: cover SWA bounded replay through the Ampere attention path" and
  "offloading: restore DeepSeek-V4.1 SWA replay windows from the host tier" (offloading/scheduler.py +248/-17,
  config.py +7/-1, tests +709): relevant to the offload question (other worker). Its launch used `--no-swa-bounded-replay`.
  Reported throughput is low and power-limited (60 tok/s c1 2k, 45 W A100s): no evidence at our 36-64 stream batches.

What the fork has that upstream lacks (for SM89): native SM89/SM12x mixed FP8/FP4-cache attention through the fork's
own FlashInfer build (`flashinfer.mla.deepseek_v41`, F/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py:38-57,
:185-204; no such module in upstream FlashInfer, whose flashinfer/mla has `_sparse_mla_sm120`, `cake_dsv4`); SM89 query
quantization (F/.../common/ops/query_quant.py:103-106); the Triton MXFP4 indexer (F/vllm/model_executor/kernels/attention/
dsa/dsv41_indexer.py, no upstream counterpart); the router fast path for (128,3)/(384,6) (F/.../nvidia/router.py:14);
CED prefill; DSpark adaptive verification on SM89.

What upstream has that the fork lacks: SWA bounded replay (`swa_bounded_replay: bool = True`, U/vllm/config/cache.py:367;
10 files reference it; 0 in F), decoder replay (#58132) and trimming (#59894), Engram rewrite (1,033 -> 1,814 lines:
huge pages, sorted rows, async prefetch, DP sharing, peer HBM), offloading fixes after 09-02 (#57145), PCP, Humming
refactor (utils/humming/ package, native backends), MegaMoE / mega attention / DeepSelect (SM100-only, irrelevant).

## 3. Overlay files: our change vs fork, and fork vs upstream main

| overlay file | our change (vs fork) | fork -> upstream counterpart | port to upstream main |
|---|---|---|---|
| e043_humming.py | new, 97 lines: MXFP4->BF16 expert decode, GPTQ precomputed reader | none | trivial; adapt tuple -> dict (upstream `weight_schema.quant_tensor` returns a dict, U/.../quantization/humming.py:686-691) |
| models/deepseek_v4_1/synth_init.py | new, 168 lines, synthetic-weight tests only | none | drop (not used in production) |
| model_loader/dummy_loader.py | +14, DSV41_SYNTH_INIT hook | 64 -> 68 lines, 12 diff lines | drop / trivial |
| deepseek_v4_1/quant_config.py | +14, DSV41_EXPERT_HUMMING hook (+ w2 schema) | deepseek_v41/quant_config.py, 230 -> 264, 370 diff lines (rename, NVFP4/Quark paths) | trivial: get_quant_method still returns Mxfp4MoEMethod for experts (U/.../deepseek_v41/quant_config.py:221-261); HummingConfig(full_config=) unchanged (U humming.py:155) |
| deepseek_v4_1/common/engram.py | +47/-6: compact table + row_map in the Triton gather (E043C) | 1,033 -> 1,814, 979 diff lines | **rewrite** (1-2 d): sorted, inline and prefetch lookup paths are new. Required: uncompacted tables at TP4 are ~2x 23.6 GiB/rank and exceed the 173 GiB cgroup |
| deepseek_v4_1/nvidia/router.py | +38/-1: shape whitelist (128,6) etc. + debug stats | no file; upstream dsv4_topk.py (169 lines) only 256/384 | trivial: generic path works; optionally add 128 at dsv4_topk.py:26 after a kernel check |
| deepseek_v4_1/nvidia/model.py | +39: Humming dequant + Engram row-chunk loader in load_weights | deepseek_v41/nvidia/model.py, 1,225 -> 1,788, 1,105 diff lines | moderate: re-insert into load_weights (expert mapping U model.py:1480-1488, name map :1565) |
| quantization/humming.py | +22/-12: humming 0.1.12 API shims, per-w2 schema, precomputed GPTQ | 815 -> 868, 165 diff lines | moderate: shims obsolete with humming 0.1.18; hooks redone on the new schema API |
| kernels/linear/__init__.py | +2: humming MXFP8 opt-in | 1,276 -> 1,452 | **drop**: native `--linear-backend humming` (U :324-331) |
| fused_moe/oracle/mxfp4.py | +3: humming MoE backend | 2,056 -> 2,327 | **drop**: native `moe_backend` "humming" (U :377) |
| quantization/utils/humming_utils.py | +5/-2: GROUP_TENSOR shim, per-sublayer schema dict | split into utils/humming/{moe,schema,linear,...}.py (moe.py 625 lines) | small-moderate: per-sublayer schema into utils/humming/moe.py |
| kernels/attention/dsa/dsv41_indexer.py | +108/-14: Q predecode (= fork PR #154) | **no upstream file**; upstream indexer needs DeepGEMM | part of porting the whole fork SM89 indexer |

The overlay itself ports in ~2-3 days. The blocker is not the overlay. Upstream has no SM89 V4.1 attention, indexer or
o_proj, so moving to upstream means also porting the fork's SM89 path: about 11k lines of model and kernel code, plus the
custom FlashInfer build, onto upstream's very different V4.1 files (model.py 1,105 diff lines, attention.py 777,
compressor.py 128). Alternatively, adopt and SM89-validate the dsv41-sm80 Triton path, at unknown speed. The fork
maintainer has not done this rebase: fork issue #153 (2026-10-22), "will update when appropriate; no machine to verify
recently".

## 4. Version constraints and packaging

| | fork vision11 | upstream v0.32.0 (2026-10-18) | upstream main (10-26) |
|---|---|---|---|
| torch | 2.13.0 (+cu130) | 2.13.0 | **2.14.0** (U/requirements/cuda.txt; pyproject.toml:10) |
| FlashInfer | 0.6.18+glm53.dsv41.vision2 (custom, 22 MiB, fork release) | 0.7.0.post1 | 0.7.2.post1 (59.2 MiB) |
| humming-kernels | 0.1.12 | 0.1.17 | 0.1.18 (0.5 MiB) |
| cutlass-dsl / quack / tokenspeed-mla | 4.6.2 / 0.6.4 / 0.1.8 | 4.8.0 / 0.6.5 / 0.2.15 | 4.8.0 / 0.6.5 / 2.0.3 |
| CUDA | 13.0 | 13.0 default PyPI | 13.0 |
| vllm wheel | 235 MiB (cp312, sm89+sm120, includes DeepGEMM 875 files) | 311.4 MiB x86_64 abi3 | nightly |

- Upstream wheel arch list includes 8.9 for generic `_C` (CMakeLists.txt:123-134), but FlashMLA, DeepGEMM and DeepSelect
  are not built for 8.9 (cmake external projects). A V4.1-on-SM89 upstream stack therefore needs our own wheel: a source
  build of the rebased tree for TORCH_CUDA_ARCH_LIST=8.9. Estimate ~1-2 h on this 24-core host (unmeasured); the fork's
  build also compiles a custom FlashInfer.
- Kaggle: the wheel set comes from kernel output chankhavu/dsv41-sm89-fork-py312-cu130 (kaggle/dsv41_tools/build.py:32,
  :350). A new stack means a new run of that kernel and a new output version (mount-error risk on fresh versions, notes §8),
  plus a re-check of the CUDA 13 shim (build.py:157-170; nvrtc/cudart from the nvidia cu13 wheels). torch 2.14 (main)
  changes the torch wheel (~529 MiB) and every compiled extension. v0.32.0 keeps torch 2.13.
- The production server needs `--linear-backend humming` (server_cmd.json), FULL cudagraphs, `VLLM_USE_V2_MODEL_RUNNER=1`,
  `FLASHINFER_CUDA_ARCH_LIST=8.9` (FlashInfer JIT on Kaggle).
