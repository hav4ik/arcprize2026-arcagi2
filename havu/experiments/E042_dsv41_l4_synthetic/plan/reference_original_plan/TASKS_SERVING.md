# Serving implementation tasks: DeepSeek-V4.1-Flash-1017 on 4× NVIDIA L4

## Goal and acceptance criteria

Implement and evaluate a text-only, pruned DeepSeek-V4.1-Flash-1017 server on the actual four-L4 competition environment. **E64/top6 is primary; E72 is the larger quality candidate.** Preserve expert width, shared experts, attention topology and quantization semantics. Deploy the validated, frozen Engram row mask produced by TASKS_ENGRAM_FIRST.md; preserve original hash addressing and the declared missing-row behavior.

The current GPU backbone estimates are **64.49 GB (60.06 GiB) for E64** and **70.86 GB (65.99 GiB) for E72**, before cache/workspace. The earlier 32–56 GB envelope and E48/E40/E32 experiments are archived smaller-envelope cases. **Tool-free behavior is the required deployment/pruning objective.** No Python-tool control or broad coding-retention requirement is added. Compare candidate wall time only on the same local hardware; off-site teacher/API controls establish quality retention rather than comparable local serving speed.

This document contains source-verified findings and implementation instructions. **No L4 benchmark or full-model execution was performed during preparation.** All new scripts, patches, fixtures and command sequences described as proposed must be implemented and run. Do not report proposed commands, theoretical memory arithmetic, or another GPU's benchmark as a successful L4 test.

## Working layout

Use an isolated working directory. Paths below are relative to its root and are proposed outputs unless already supplied in this package.

| Path | Purpose |
|---|---|
| `vendor/vllm-sm89/` | Pinned vLLM fork and tracked implementation changes |
| `vendor/flashinfer-v41/` | Pinned native V4.1 SM89 kernel source and tests |
| `vendor/engram-reference/` | Pinned reference offload patch repository; source material, not an overlay |
| `locks/serving.json` | Model, repositories, wheels, build flags, driver and package provenance |
| `patches/serving/` | Reviewable patch series exported from the pinned base |
| `tools/serving/` | New benchmark, fixture, validation and launch scripts |
| `tests/serving/` | Focused tests for new integration behavior |
| `fixtures/serving/` | Synthetic/small weights, tensor fixtures and exact selected real rows |
| `configs/serving/` | Explicit launch and benchmark configurations |
| `runs/serving/<run_id>/` | Environment, logs, metrics, profiler traces and test results |
| `checkpoints/arc-v41-e64/` | First real pruned checkpoint supplied by the pruning workstream |
| `engram/` | Compact retained original rows, mapping, frozen mask and checksums |

Each run must record the model revision, patch commit, software lock, seed, prompt hashes, context/output lengths, concurrency, CED/DSpark/prefix/graph settings, and GPU allocation. Do not put credentials or a complete environment-variable dump in logs.

The package already supplies `tools/probe_environment.py` and `tools/estimate_budget.py`; use those as initial inventory and arithmetic helpers. They do not replace device tests or an engine memory profiler.

## Fixed baseline and provenance

### Model and repository pins

| Component | Pin and source |
|---|---|
| Exact model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at Hugging Face revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, last modified 2026-10-17 05:46:55 UTC |
| Editable vLLM baseline | `yhfgyyf/vllm-deepseek-v4-sm89` at **`3f3bbaea916707c851ca027b87b97fb6605998b8`**, September 16, 2026 |
| Released Vision11 wheel Python source | **`86d35745c8797c9d2524877fb235084a6a5f5a7d`**; distinguish this from the editable baseline |
| FlashInfer source inspected | `yhfgyyf/flashinfer`, branch `deepseek-v4.1-native-sm89-sm120`, commit **`d4eef3c44c70594ff18b9126381ae6a978694e3f`**, September 12, 2026 |
| Engram reference repository | `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` at **`a2664a7e4ac7bc823ef29c8d7725abfe4d286f8a`**, October 19, 2026 |

Primary sources:

- [Exact model revision and configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)
- [vLLM pinned source](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/tree/3f3bbaea916707c851ca027b87b97fb6605998b8)
- [Vision11 release and installation provenance](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/tag/v0.28.1rc1-vision9-sm89-sm120-cu130)
- [FlashInfer pinned source](https://github.com/yhfgyyf/flashinfer/tree/d4eef3c44c70594ff18b9126381ae6a978694e3f)
- [Engram reference repository pin](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/a2664a7e4ac7bc823ef29c8d7725abfe4d286f8a)

The release tag retains the word **Vision9**, but its recommended wheel is **Vision11**. The tag's automatic source archive does not equal the Vision11 wheel's Python source. The wheel reused audited binaries from prior builds and rebuilt its top-k extension. Keep its exact SHA256 and do not infer reproducibility from the retained tag name.

The FlashInfer source commit above was inspected separately. A byte-for-byte comparison of every paired wheel file against that source tree was not performed. Record the actual installed package, import paths, wheel checksum and any differences before development.

### Published wheel stack

- vLLM: `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`
- FlashInfer: `0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213`
- Linux x86_64; Python 3.12; CUDA 13.0; PyTorch 2.13.0+cu130; Triton 3.7.1; Transformers 5.16.1.
- vLLM wheel SHA256: `e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed`.
- FlashInfer wheel SHA256: `078aa7682d699c67dce619fc8acf29ec41934ab5ba916268d1fe6f8850459de8`.
- Published container tag: `ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130`. Resolve and record its immutable image digest if containers are usable; a tag is not a content pin.

**Evidence limits:** the README advertises full-model operation on modified 48 GB RTX4090s and 96 GB RTX PRO6000s. Latest Vision11 packaging tests ran on SM120; they did not repeat physical SM89 full-model validation. [Issue 110](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/110) documents a real 8× L20 SM89 deployment and a top-k launch limit tied to SM count/shared memory. Neither establishes this exact 1017/E64/L4 combination. Treat L4 device tests as mandatory.

## Source map for implementers

Paths in the first table are relative to `vendor/vllm-sm89/` at the pinned baseline.

| Path / symbol | Why it matters |
|---|---|
| `vllm/models/deepseek_v4_1/nvidia/model.py`, `DeepseekV4MoE`, `_select_dsv4_attn_cls` | V4.1 router installation, separate draft counts, modular-MoE requirement, exact attention dispatch |
| `vllm/models/deepseek_v4_1/nvidia/router.py`, `_SUPPORTED_ROUTING_SHAPES`, `deepseek_v41_topk` | Existing shapes are only `(384,6)` and `(128,3)`; new counts need a patch |
| `vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | Native mixed-cache API, BF16 queries, local head-count requirements and workspace |
| `vllm/models/deepseek_v4_1/attention.py` | Shared-cache source topology, replicated indexer, cache specifications and BF16 output projection layout |
| `vllm/v1/attention/backends/mla/indexer.py`, `dsv41_indexer_uses_fp4` | Native V4.1 requires FP4 index cache; architecture and compressed-context constraints |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | Candidate scoring, top-k dispatch and staging buffers |
| `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `select_deepseek_v41_mxfp4_moe_backend` | SM89 selects Marlin rather than hardware-native FP4 kernels |
| `vllm/model_executor/layers/quantization/mxfp4.py`, `DeepseekV41Mxfp4MoEMethod` | Official MXFP8 activation boundaries and router-weight ordering |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` | Packed expert weights remain four-bit; scale buffers become BF16 |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` | MoE GEMM calls, intermediate buffers and quantization semantics |
| `vllm/models/deepseek_v4_1/quant_config.py`, `DeepseekV41WoALinearMethod` | `wo_a` loads into BF16; other linears use the model-specific MXFP8 profile |
| `vllm/models/deepseek_v4_1/common/engram.py` | Hash state, TP head ownership, exact row offsets, pinned-UVA storage and lookup |
| `vllm/models/deepseek_v4_1/nvidia/model_state.py` | Runner integration point for live Engram prestaging |
| `vllm/config/engram.py` | Baseline schema contains `cpu_offload`; disk configuration is new work |
| `vllm/engine/arg_utils.py` | Verified CLI includes language-model-only, BooleanOptional prefix caching and Engram configuration |

FlashInfer paths are relative to `vendor/flashinfer-v41/`.

| Path | Role |
|---|---|
| `flashinfer/mla/deepseek_v41.py` | Native format validation, API and workspace sizing |
| `flashinfer/mla/_deepseek_v41_sparse_triton.py` | FP4/FP8 tile decoding and BF16 MMA attention |
| `tests/attention/test_deepseek_v41_mixed_sparse.py` | Existing numerical, layout, boundary and CUDA-graph tests |

## Memory contracts to preserve and measure

### Experts and dense modules

The routed-expert parameter count is `40 × E × 3 × 5120 × 2304`. The inspected Marlin path retains packed four-bit expert matrices but expands group-32 scales into BF16:

`resident expert bytes ≈ parameter_count × (0.5 + 2/32)`.

| E | Routed experts only | With illustrative 12.6 GiB fixed reserve |
|---:|---:|---:|
| 32 | 23.73 GiB | 36.33 GiB |
| 40 | 29.66 GiB | 42.26 GiB |
| 48 | 35.60 GiB | 48.20 GiB |
| **64** | **47.46 GiB** | **60.06 GiB** |
| **72** | **53.39 GiB** | **65.99 GiB** |

The fixed reserve is an estimate, not a measured allocation. Account for replicated modules, padding, scales, LM head/embeddings and shared experts explicitly. The 40 backbone `wo_a` tensors occupy approximately 2.539 GiB as BF16 plus expanded scales in this implementation: 1.288 GiB more than their native checkpoint representation. Other MXFP8 linears remain eight-bit after Marlin repacking, with rowwise scale expansion.

Exclude DSpark and vision from the initial operating baseline using supported configuration and loader behavior. Keep `VLLM_MARLIN_INPUT_DTYPE` unset. The specific V4.1 method requires packed W4A16 after its official group-32 MXFP8 activation quantize/dequantize boundary. Generic W4A16 or a different activation rounding rule is not an equivalent replacement.

### KV cache under ordinary TP4

| Cache | Stored row | Owners |
|---|---:|---|
| Sliding window | 528 B: FP8 values + group-32 UE8M0 scales | Every backbone layer, per TP rank |
| Global attention | 288 B: FP4 values + group-16 FP8 scales | Layers 2, 8, 14 at ratio 2; layer 20 at ratio 1 |
| Indexer K | 68 B: FP4 values + four UE8M0 scales | The same four source layers; index-only reindex layers share them |

Queries divide across TP ranks, but these single-KV-head caches are replicated. Consequently the linear payload is **890 bytes/token/request/GPU**, or 3560 bytes across TP4. At 65,536 tokens it is 55.625 MiB/request/GPU. Add a logical SWA floor of `40×128×528` bytes, approximately 2.58 MiB/request/GPU, plus page padding, retained blocks, prefix policy, replay state, workspace and speculation.

See KV_MEMORY_NOTES.md for explicit runtime components: default indexer scratch is **577.03125 MiB/rank**, shared candidate/top-k arrays add **80 MiB/rank at 8192 scheduled-token capacity**, and compressor rings add 96 KiB/request/rank. Optional CED tails reserve approximately 5.003 MiB/slot/rank at 128 tokens, or 45.026 MiB with a retained 1024-token vision reserve. Measure actual normalized text-only settings, extra SWA pages and in-flight prefill rows. The current estimator counts known buffers separately from residual workspace; do not add them twice.

The native command-line `--kv-cache-dtype fp8` selects this **mixed** layout. Do not replace it with all-FP8 caches or count the generic 132-byte indexer branch; V4.1's validator requires native FP4 index rows. Native attention passes separate SWA and global slots. Use up to 128 SWA rows and 512 selected compressed global rows in the baseline; a compression ratio of two changes original-token coverage, not a requirement to double the number of native global slots.

## Task 0 — Inventory the actual target environment

**Dependencies:** none. **Outputs:** `runs/serving/inventory/environment.json`, `topology.txt`, `storage.json`, and `locks/serving.json` with unresolved fields clearly marked.

Run the supplied inventory tool in the actual Kaggle environment. Record effective cgroup memory limits as well as physical host memory. Record each GPU's usable bytes, architecture, SM count, shared-memory limits, driver version and PCIe topology. L4 is **Ada, compute capability 8.9**, not SM86.

Verified supplied-tool command, from the package root:

```bash
mkdir -p runs/serving/inventory
python tools/probe_environment.py --paths . /tmp /dev/shm > runs/serving/inventory/environment.json
nvidia-smi topo -m > runs/serving/inventory/topology.txt
```

The supplied probe does not benchmark P2P, collectives or storage. Implement focused probes for those. For storage, measure random reads on a disposable file in the same intended storage tier and record cache conditions. Never run a write benchmark against model files or other existing user data. Reading from a mounted competition dataset may behave differently from local scratch or NVMe.

**Pass gate:** identify the real driver/toolchain compatibility, usable GPU bytes, effective host limit, accessible storage capacity and whether the intended local model/Engram footprint can be made available. If the CUDA 13 wheel cannot run on the target driver, resolve a compatible build before spending on pruning. Installing a toolkit does not upgrade the host driver. If all four GPUs are reserved during this inventory, count all four in allocated GPU-hours.

## Task 1 — Create a reproducible pinned environment

**Dependencies:** Task 0. **Outputs:** `locks/serving.json`, package freeze, wheel checksums, import paths, `patches/serving/0000-baseline-provenance.md` and one successful CUDA smoke-test log.

Use a fresh virtual environment. The following installation command is **copied from the publisher's verified release instructions; it was not executed in this research**. It requires the compatible runtime established in Task 0:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130/vllm-0.28.1rc1.dev517%2Bglm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed' \
  'transformers==5.16.1' 'triton==3.7.1'
```

Its dependency installs the paired FlashInfer wheel. Do not overwrite it with latest upstream FlashInfer. For modifications, create an editable checkout at the pinned vLLM baseline and follow its source-build instructions, preserving `requirements/cuda.txt` and its exact FlashInfer URL. The published source build requires CUDA 13.0, C++ compilation, Rust supporting the 2024 edition, and the pinned Python dependencies. Build on a compatible external host if target notebook startup time is constrained, then test the produced wheel on the actual target.

Proposed checkout sequence using standard Git commands, not an executed environment build:

```bash
mkdir -p vendor
git clone --filter=blob:none --no-checkout https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git vendor/vllm-sm89
git -C vendor/vllm-sm89 checkout 3f3bbaea916707c851ca027b87b97fb6605998b8
git -C vendor/vllm-sm89 switch -c arc-v41-serving
git clone --filter=blob:none --no-checkout https://github.com/yhfgyyf/flashinfer.git vendor/flashinfer-v41
git -C vendor/flashinfer-v41 checkout d4eef3c44c70594ff18b9126381ae6a978694e3f
```

On L4 set `FLASHINFER_CUDA_ARCH_LIST=8.9`. Keep `VLLM_USE_V2_MODEL_RUNNER=1`. Unset `VLLM_MARLIN_INPUT_DTYPE` and `TRITON_PTXAS_BLACKWELL_PATH`; the latter belongs to an SM120-specific recipe. For a source build, use the fork's build requirements and confirm any change to `TORCH_CUDA_ARCH_LIST` against its CMake build rather than assuming a generic wheel recipe suffices.

**Meaningful checks:** CUDA BF16 matmul executes; expected vLLM and FlashInfer symbols import from the intended environment; native V4.1 workspace API exists; no unsupported-architecture or missing-cubin error occurs when the first real V4.1 operator is launched. Record wheel/package hashes and import paths so an old site-packages installation cannot silently shadow the checkout.

**Pass gate:** imports and an actual L4 CUDA operation succeed with a reproducible lock. A source compile without device execution is not a numerical pass. If blocked, evaluate [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport), pin its selected revision/image digest, and repeat these gates. Do not change serving frameworks without documenting the specific blocker.

## Task 2 — Prove native attention on one L4

**Dependencies:** Task 1. **Outputs:** `tools/serving/bench_attention.py`, `runs/serving/operator-attention/{tests.xml,metrics.json,profile.json}` and any narrowly scoped kernel patch.

**Use a 16-query-head fixture corresponding to one TP4 shard. Do not instantiate the full 64-head backbone at TP1.** The native kernel currently accepts local query heads 8, 16 or 32. The full model's supported corresponding TP degrees are 8, 4 and 2.

Run existing FlashInfer tests first. Their path and test names are source verified; this invocation is a proposed execution against the installed paired build:

```bash
python -m pytest vendor/flashinfer-v41/tests/attention/test_deepseek_v41_mixed_sparse.py \
  -k 'pack_boundaries or selected_row_reference or prefill_candidate_boundaries or fused_inverse_rope or cuda_graph_changed_inputs' \
  --junitxml=runs/serving/operator-attention/tests.xml
```

Create the output directory first and install the test-only dependencies required by that repository. Verify the test module imports the intended paired FlashInfer package. Preserve existing tolerances; do not loosen them simply to make a new device pass. Existing selected-row comparisons use rtol/atol 0.03, while fused/unfused inverse-RoPE comparisons at the same BF16 boundary use exact equality.

Benchmark the actual 16-head kernel with SWA-only and native mixed caches, page sizes 64/128 where applicable, 128 SWA slots, and up to 512 selected global slots. Include decode token counts 1/4/8/16/32 and backing context lengths 8K/32K/64K; add 128K only if the intended harness requires it. Include the real indexer/candidate path separately so a fixed selected-row attention benchmark does not hide context-dependent indexing work.

**Required cases:** pack/unpack boundary rows, final valid row in padded storage, zero-width global slots, all-invalid/padded query rows, causal/pre-fill boundaries, nonzero sink parameters, correct inverse RoPE, changed inputs and positions across repeated graph replays, mixed decode/prefill slices. Measure compile time separately from warmed latency; use CUDA events or synchronized profiling correctly. Record actual shared-memory and workspace needs on L4, not an RTX4090 assumption.

**Pass gate:** all selected supported-device tests pass unchanged, finite inputs produce finite valid outputs, changed-input replay matches eager behavior, and no unsupported launch shape or out-of-bounds access appears. Report p50/p95 timing and peak memory without extrapolating a full-model tokens/sec number. If a failure is only in an explicitly unused backend, prove dispatch excludes it; otherwise fix the failure before proceeding.

## Task 3 — Patch pruning-compatible routing and verify packed MoE

**Dependencies:** Task 1; may run alongside Task 2. **Outputs:** `patches/serving/0001-pruned-router.patch`, `tests/serving/test_v41_pruned_router.py`, `tools/serving/bench_moe.py`, and `runs/serving/operator-moe/` metrics.

Extend `_SUPPORTED_ROUTING_SHAPES` in `vllm/models/deepseek_v4_1/nvidia/router.py` for `(64,6)` and `(72,6)` while keeping original backbone/draft cases. Update error messages to describe the actual contract. The existing Triton kernel already has masked `NUM_EXPERTS` and `BLOCK_N=next_power_of_2(NUM_EXPERTS)`. Avoid introducing a new router implementation unless the existing one fails a demonstrated requirement.

Test E64 first, then E72. E48/E40/E32 are optional archived smaller-envelope cases. The reference must preserve:

1. Stable `sqrt(softplus(logits))` scores.
2. Text/image correction bias for selection only, selected from the same token-ID conditions.
3. Lowest expert ID as the exact tie-breaker.
4. Original unbiased selected scores, normalization and configured 1.5 routing scale.
5. Correct masking of padded lanes and documented behavior for exceptional logits.
6. Consistent local/global expert mapping and unchanged DSpark `(128,3)` routing.

Require exact selected IDs for deterministic constructed cases, including ties, and close FP32 weights against a reference with the same semantics. For randomly generated finite inputs near a decision boundary, diagnose numerical differences rather than treating arbitrary PyTorch top-k tie order as the reference. Negative tests should still reject unsupported dtypes/layouts and inconsistent biases. Include NaN-handling tests without declaring NaN inputs a valid inference workload.

Build one full-width MoE layer: hidden 5120, intermediate 2304, original SwiGLU/clamp behavior, E64, top6. Use synthetic legal quantized tensors first, then selected real expert tensors plus captured hidden states. Reproduce the model's group-32 activation quantize/dequantize and placement of router weights before the second projection's quantization. A generic BF16 SwiGLU reference without those boundaries is inadequate.

Inspect GPU buffers after `prepare_moe_mxfp4_layer_for_marlin`. Confirm expert matrix storage is packed four-bit and expert scale storage is BF16. Record expected payload, actual parameter bytes, padding, reserved memory, temporary repack peak and intermediate workspace. The one-layer experiment does not require downloading the complete model; select tensors by safetensor header/byte ranges or download only necessary shards and verify declared lengths.

**Pass gate:** all new router shapes pass semantic tests; packed residency matches accounting; the MoE output meets the established same-quantization reference tolerance; real tensors produce finite stable outputs; no whole expert bank expands to BF16/FP8. Keep top6 and expert width unchanged. Leave TP4/EP-specific acceptance to Task 6; do not assume 10 or 12 local experts work merely because the repacker iterates over arbitrary E.

## Task 4 — Integrate the frozen compact Engram table in native CPU offload

**Dependencies:** Task 1 and the E01–E03 address/selection/adapter work in TASKS_ENGRAM_FIRST.md; final acceptance uses the validated frozen E04 manifest. CPU census and adapter development can run alongside Tasks 2–3, and this adapter supports E04 validation. Main REAP scoring waits for the frozen mask. **Outputs:** `patches/serving/0002-engram-compact.patch`, `tests/serving/test_engram_compact.py`, `tools/serving/bench_engram.py`, `engram/manifest.json` and measured host/GPU memory reports.

The pinned `ParallelEngramEmbedding` already supports `cpu_offload:true`: TP-local FP8 tables and scale shards reside in pinned CPU memory, UVA kernels gather required rows, and TP all-gather assembles outputs. The full source payload is 188.83 GiB aggregate; the current plan deploys only selected original rows. Implement an original-ID-to-compact-row adapter while preserving native hash geometry, original bank/head ownership and lookup arithmetic. Do not allocate or pin the original full-size table before replacing it with the compact storage.

Use `masks/engram/frozen/manifest.json` from the Engram-first workstream. Retained rows preserve their original FP8 values and scales. Missing lookups produce **zero embedding rows before Wkv/projection**; the remainder of the module runs unchanged. This is the already-validated model behavior, not a cache miss falling through to a hidden full-table backing. Do not rehash, change prime moduli, substitute another row, zero the final residual, or silently load extra rows. Record the frozen mask checksum in every serving run.

The supplied `tools/compact_engram_reference.py` is a CPU-only synthetic reference for byte-preserving compaction, original-ID mapping, a zero sentinel and TP4 head ownership. It is not a native hash census, streaming checkpoint exporter, GPU dequantizer or vLLM integration. Use it as a contract fixture while implementing the adapter.

**Required tests:** original/compact retained row byte and scale parity; missing-row parity against a full-table reference with identical pre-projection zeroing; first/last and empty bank handling; nonzero offsets; original TP head ownership; hash collisions; repeated n-grams; chunk/prefix lookbacks; changed tokens; target-harness continuation; request interleaving; rollback if speculation is later enabled; finite padding and changed-input graph replay. Confirm no full-table CPU/GPU allocation occurs and no out-of-range index reaches a gather kernel.

CPU table bulk does not consume VRAM. Engram Wkv/projection weights, GPU ID maps, gathered embeddings, staging, transfer buffers and any hot cache still do. Record map representation and bytes, host RSS/pinned pages, UVA traffic, TP collectives, p50/p95 lookup delay and per-rank transient peaks under realistic batches. Keep host allocation and GPU gather output ownership explicit.

**Pass gate:** the native adapter reproduces the frozen mask's declared arithmetic, respects host limits and survives state/concurrency tests. Complete held-out quality must already have passed E04 and must be checked again in the integrated candidate; a high hit rate is not the quality gate.

### Archived fallback: exact full-table disk prestaging

If needed for an exact reference or if even compact pinned tables exceed the host budget, inspect the [pinned Engram stager](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/a2664a7e4ac7bc823ef29c8d7725abfe4d286f8a/patch/cudagraph-prestage/engram.py), [runner adapter](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/a2664a7e4ac7bc823ef29c8d7725abfe4d286f8a/patch/cudagraph-prestage/model_state.py) and associated tests. Port a small adapter, not complete foreign attention/runner files. CPU reads and staging must execute outside CUDA graph replay and refresh every step. If serving the frozen compact model from disk, preserve its exact mask and miss policy; full backing with fallback on every missing row is a different model. This disk path is not mandatory for the current native CPU-resident PoC.

## Task 5 — Exercise the cross-layer topology with a small integration fixture

**Dependencies:** Tasks 2, 3 and 4. **Outputs:** `tools/serving/build_fixture.py`, `fixtures/serving/topology/`, `tests/serving/test_topology_state.py`, and an eager integration trace.

A single attention layer cannot prove shared cache/index state. Build a small-weight fixture that preserves the relevant original layer IDs and ownership, or instrument selected layers from the full architecture. Include pure-SWA layers, KV+index sources, index-only reindex sources, reuse consumers and the transition at layer20. Avoid renumbering source IDs in a way that changes the tested relationship.

For a one-L4 numerical fixture use 16 query heads representing a TP4 shard, and explicitly mark altered dimensions/weights as fixtures. This is not a TP1 full-model result. For engine-wide testing, move to TP4 with reduced/synthetic weights and preserved 64-head topology. A 40-layer real E64 checkpoint does not fit on one L4.

Verify sources 2/8/14 publish ratio-2 global and index caches; source20 publishes ratio-1 caches; later reindex layers read source20's K cache; reuse consumers see the correct published top-k. Exercise full prefill then decode, multiple chunks and request interleaving. Retain CED off, DSpark off, prefix caching off and eager execution. Verify language-model-only loading does not accidentally retain unnecessary vision parameters, and record which parameters were excluded.

**Pass gate:** intermediate attention outputs, top-k indices, cache contents and Engram output agree with a reference executing the same fixture and arithmetic. State does not leak between requests. Real selected tensors produce finite logits. A fluent completion by itself is not a parity test.

## Task 6 — Prove TP4, EP choices and runtime memory on four L4s

**Dependencies:** Task 5 and actual four-L4 access. **Outputs:** `configs/serving/tp4-e64-eager.json`, `runs/serving/tp4-fixture/`, per-rank memory reports and a parallelism decision.

Begin with TP4 and test **EP off and EP on** using the same inputs/configuration. E64/E72 imply 16/18 local routed experts under EP4; verify initialization, expert mapping, token dispatch, padding and output reduction for each supported candidate. Inspect actual Marlin calls and collective traffic. Do not assume the generic fused-router table used by another fork applies to this native router.

Use measured topology/P2P and a short NCCL collective benchmark before changing NCCL settings. Record per-layer attention, indexer, MoE, Engram, collectives and CPU scheduler time. Tune only a demonstrated bottleneck. Avoid starting with pipeline parallelism: this fork raises when a pipeline split separates a consumer from its KV/index source. PP requires deliberate cache transport or verified sharing-group placement, not a launch flag alone.

Measure parameter bytes, live allocation, reserved allocation and peak separately, per rank and aggregate. Report at least: packed experts/scales, shared/dense modules, replicated modules, `wo_a`, global/index/SWA caches, prefill workspace, temporary quantization/repacking, hash/replay buffers and future graph reserve. Confirm ordinary TP4 global/index cache replication experimentally rather than dividing 890 B/token by four.

Start with maximum context 32K or 64K, **max-seqs 8**, a conservative prefill budget, eager execution and no optional acceleration. Run memory and state tests before large concurrency sweeps.

**Pass gate:** both selected TP/EP configurations initialize and agree within the established distributed numerical tolerance; chosen expert counts route correctly; no rank exceeds the actual device budget; all host/disk state tests hold under concurrency. Choose the configuration using measured latency/throughput and memory. If only one EP setting is correct, use it and document the other as unsupported until fixed.

## Task 7 — Serve the first real E64 checkpoint and run the batch sweep

**Dependencies:** Task 6 plus a verified E64 checkpoint from the pruning workstream. **Outputs:** `tools/serving/launch_e64.sh`, `tools/serving/bench_arc_serving.py`, `runs/serving/e64-baseline/`, and an explicit accept/fix/reduce-model decision.

Validate the checkpoint's revision, expert-retention manifest, remapped expert IDs, gate rows and both correction-bias vectors before loading. Verify safetensor declared data extents against actual files. Use the exact 1017 tokenizer and prompt encoding; do not substitute V4 templates. The baseline includes the original shared experts and attention, plus the validated frozen compact Engram rows and declared miss behavior. Do not label a randomly initialized or topology fixture as a pruned model.

The following is an **illustrative launch skeleton**, not an out-of-the-box command. `ARC_DSV41_ENGRAM_MANIFEST` is a proposed interface to implement in Task 4; the unmodified fork does not recognize it. Finalize the script against `vllm serve --help` and the implemented config. The other options shown exist in the inspected baseline:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASHINFER_CUDA_ARCH_LIST=8.9
unset VLLM_MARLIN_INPUT_DTYPE
unset TRITON_PTXAS_BLACKWELL_PATH
export ARC_DSV41_ENGRAM_MANIFEST=/absolute/path/to/engram/manifest.json

vllm serve /absolute/path/to/checkpoints/arc-v41-e64 \
  --served-model-name arc-v41-e64 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --moe-backend auto \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --engram-config '{"cpu_offload":true}' \
  --hf-overrides '{"ced_prefill":false}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41
```

Add `--enable-expert-parallel` only when Task 6 selects EP. Add tool-parser options only when the actual ARC harness uses that interface and its round-trip test passes. No speculative configuration means DSpark remains off. Log final resolved settings, because prefix caching is enabled by default in the baseline unless explicitly disabled.

First confirm finite logits and real ARC prompt/tokenization behavior at batch8. Then sweep concurrency **16/32/64** across the actual context distribution, for example 8K/32K/64K, stopping each configuration on a concrete OOM or correctness failure. Measure prompt/prefill tokens and generated tokens separately. Include real novel tool-free reasoning from the target harness, not only counting, repeated text or cache hits.

Record TTFT, aggregate generated tokens/sec, per-request TPOT, p50/p95 completion latency, token-length distribution, failures/timeouts, per-rank peak memory, Engram read behavior and useful task solves per runtime. Include cold startup/JIT/cache behavior that matters to the submission. Do not invent a throughput target from GPU specifications; compare measured results to the existing baseline and the accepted competition runtime budget.

**Pass gate:** total GPU-resident model weights meet the chosen envelope, with bytes and units explicit; per-rank headroom supports the selected batch/context configuration; no invalid logits or state errors occur; held-out ARC quality is acceptable under the pruning protocol; and measured useful throughput justifies the deployment. If E64 quality passes but batch memory fails, first reduce configured concurrency/prefill reservations; smaller expert counts are a subsequent controlled experiment. Evaluate E72 only when its quality benefit and measured headroom justify it. If serving passes but pruning quality fails, return the result to the pruning workstream rather than masking it with more samples.

## Task 8 — Add only measured accelerations, one at a time

**Dependencies:** Task 7 baseline passes. **Outputs:** one configuration and comparison report per accepted feature; final `configs/serving/competition.json` and locked launch script.

Evaluate in this order: CUDA graphs, prefix caching, CED, DSpark. Each feature gets a same-input baseline comparison and the relevant state tests. Keep only changes that improve the actual solve/runtime objective.

- **Graphs:** native UVA lookup must consume fresh hashes, compact IDs and request state on every step. Test changed inputs, padding, positions, batch boundaries and output reuse. If the optional host/disk stager is enabled, refresh it outside graph replay with persistent buffers and explicit ordering; replaying GPU nodes cannot execute fresh Python reads. Detect stale rows numerically.
- **Prefix caching:** verify n-gram lookback, encoder/decoder state, cache ownership and tool-turn continuation. Count cold misses as well as hits. Shared prefixes should come from the harness, not a synthetic benchmark-only assumption.
- **CED:** this fork calls its decoder-prefill shortcut approximate. Assess held-out ARC quality and parity expectations explicitly. Do not demand bitwise equality with full prefill or silently assume the shortcut is distribution preserving. Include chunk and replay-tail boundaries.
- **DSpark:** initially retain the original separate draft expert count/top3 and measure its additional resident bytes. Validate target/draft state, rejection/rollback and token distribution behavior for the pruned target. Record acceptance length and total end-to-end speed, including draft cost. The fork's adaptive verification requires FULL graphs; it is incompatible with an eager/PIECEWISE baseline. Use the original working k5 configuration before changing speculative block sizes. Do not infer ARC acceptance from counting benchmarks.

**Pass gate:** accepted features preserve the required correctness/quality standard, fit memory, and improve measured useful throughput under the intended harness. Stop optional tuning when the submission goal is met. Do not broaden testing without a concrete unresolved risk.

## Allocated GPU-hour budget and stop points

These are **planning caps for the first engineering pass**, not measured execution durations or provider prices. GPU-hours mean reserved GPUs multiplied by wall-clock hours, including compilation and idle time while reserved. CPU-only source reading, editing, fixture generation and preparation should occur before reserving GPUs when possible. Downloading the complete checkpoint is not required for Tasks 2–3.

| Stage | Initial allocation cap | What must exist before expanding |
|---|---:|---|
| Actual-target inventory | 1 aggregate L4 GPU-hour | Environment, driver, memory and storage decision |
| Paired environment plus one-L4 attention and MoE tests | 8 L4 GPU-hours total | Real-device numerical pass and packed-memory proof |
| Engram single-device integration | 3 L4 GPU-hours | Exact row/state parity and bounded host memory |
| TP4 small integration and EP/memory gate | 8 aggregate L4 GPU-hours, e.g. 4 GPUs × 2 hours | Working topology, cache replication and E64 EP decision |
| Real E64 baseline and first batch sweep | 12 aggregate L4 GPU-hours, e.g. 4 × 3 hours | Checkpoint exists; prior gates pass; measured solve/runtime comparison |
| Optional accelerations | 12 additional aggregate L4 GPU-hours | Baseline already useful and a measured bottleneck motivates tuning |

Thus the first real-baseline pass is capped at **32 allocated L4 GPU-hours**, with **12 additional GPU-hours** for justified optional tuning. This is an allocation decision, not a promise that all engineering will finish within it. If a gate fails, save a minimal reproducer, release the allocation, and do the next source fix on CPU before rerunning only affected tests. Report actual allocated hours separately from kernel execution time. Larger hardware used for teacher calibration or pruning belongs to the pruning budget, not this serving budget.

## Handoff checklist

The serving workstream is complete when another terminal agent can reproduce the result from the package without relying on a mutable branch, an undocumented system installation or a chat-only instruction:

- Exact source/model/wheel/image provenance and all local patches are recorded.
- Driver compatibility was proved on the actual competition environment.
- Native attention, new router shapes, packed MoE, Engram rows/state, cross-layer sharing and TP4 behavior passed meaningful tests.
- GPU and host memory reports distinguish weights, caches, staging, workspaces and transient peaks.
- E64/E72 comparisons use identical local hardware and solve rules; tool-free is the required target, with retention/accuracy curves against the original model.
- Final launch and benchmark scripts exist and were actually run; illustrative commands have been replaced with verified commands or retained as explicitly untested alternatives.
- Measured ARC quality and useful throughput, cold-start behavior, exact allocated GPU-hours and unresolved limitations are reported together.
