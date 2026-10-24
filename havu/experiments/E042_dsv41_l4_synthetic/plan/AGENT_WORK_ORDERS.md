# Engineering work orders: synthetic DeepSeek-V4.1 on 4×L4

## Scope and status

Prepare a checkpoint-free, full-shape inference fixture before starting the rental clock. The fixture must allocate and execute all 40 original backbone layers (20 encoder and 20 decoder layers), using native quantization, attention, cache and communication paths. Synthetic weights eliminate checkpoint downloads and checkpoint disk storage; they do not eliminate GPU weight residency, repacking peaks, cache/workspaces or Engram host memory.

These are **proposed implementation tasks**, based on inspection of the source pins below. No GPU execution or complete synthetic launch has been validated. `prepare_synthetic_metadata.py`, supplied separately, prepares metadata only: it does not implement GPU weight initialization, router support or compact Engram lookup. Any new manifest fields, modules, scripts or switches described here must be implemented and documented before use; they are not existing launch flags. There is deliberately no purported working full-model launch command.

### Frozen baseline

| Item | Required baseline |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017; metadata revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| vLLM | `yhfgyyf/vllm-deepseek-v4-sm89`, commit `3f3bbaea916707c851ca027b87b97fb6605998b8` |
| FlashInfer | `yhfgyyf/flashinfer`, commit `d4eef3c44c70594ff18b9126381ae6a978694e3f` |
| Preparation-machine audit paths | `/tmp/arc-sm89-review`, `/tmp/arc-flashinfer-review`; these are paths used during preparation, not assumed paths on the user's PC or rented host |
| Geometry | 40 layers, hidden 5120, global expert intermediate 2304, 64 query heads |
| Parallelism | TP4; local expert intermediate 576; local query heads 16 |
| Pruning fixture | E64 first, E72 second; top6 and shared-expert geometry unchanged |
| Execution | Language-only; DSpark/speculation, CED and prefix caching off |
| Quantization | Native MXFP4 experts, dense MXFP8, native activation boundaries and cache layouts |
| Existing environment settings | `VLLM_USE_V2_MODEL_RUNNER=1`; `FLASHINFER_CUDA_ARCH_LIST=8.9`; leave `VLLM_MARLIN_INPUT_DTYPE` unset |

Record source commits, local diff, metadata hash, dependency versions and compiled extension locations. Do not substitute upstream vLLM/SGLang behavior for evidence about these forks.

## Work order 1 — Extend the native router to E64/E72

**Owner:** runtime/kernel engineer. **Dependency:** none. **Deliverable:** focused patch plus reference comparisons.

The pinned router accepts only `(experts, topk) = (384, 6)` and `(128, 3)`. Editing model metadata alone therefore cannot construct E64/E72. Its Triton kernel already masks expert offsets and uses a power-of-two block, but smaller shapes remain unvalidated. See [native router](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/router.py).

Add `(64, 6)` and `(72, 6)` to `_SUPPORTED_ROUTING_SHAPES`; update both rejection messages and tests. Retain original supported shapes. Preserve `sqrt(softplus(logit))`, correction-bias selection, original-score normalization, routed scaling, deterministic lowest-ID tie handling, text/image bias selection and shared-expert handling. Do not replace routing with softmax or reduce top6. E72 must correctly mask lanes 72–127; E64 uses a 64-lane block.

Extend [existing router tests](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/tests/models/deepseek_v4_1/test_moe_router.py) for E64/E72 construction and GPU comparison to its PyTorch reference. Include zero tokens, single-token decoding, boundary batch sizes, ties, biased choices, valid ID range, unique selected IDs and int32/int64 IDs. Compare routed weights with the reference; verify their sum matches the configured scaling factor. Retain original-shape regressions and sentinel tests.

**Acceptance:** exact selected IDs on specified deterministic cases and reference-agreeing weights; all original shapes still work. **Stop:** masked experts selected, changed scoring semantics, silent fallback, or unsupported launch. A source-level allowlist edit alone is not GPU acceptance.

## Work order 2 — Implement bounded native synthetic loading

**Owner:** model-loading engineer. **Dependencies:** metadata fixture; router patch for full construction. **Deliverable:** explicitly named synthetic loader, initialization manifest and focused tests.

### Why stock dummy is insufficient

[BaseModelLoader](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/model_loader/base_loader.py) constructs the model before loading weights. [DummyModelLoader](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/model_loader/dummy_loader.py) avoids checkpoint downloads but rejects nonempty `model_loader_extra_config`. Its [initializer](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/model_loader/weight_utils.py#L1294) skips integer tensors on CUDA, resets the seed for each parameter and makes a full FP16 temporary for each sub-16-bit floating parameter. The module traversal also recursively visits state dictionaries. Consequently, packed MXFP4 weights/scales remain constructor zeros, some MXFP8 scales remain uninitialized, same-shaped layers receive identical values and initialization can create large memory peaks.

Implement a separate explicit loader or a tightly scoped synthetic branch. Initialize each unique parameter once and preserve semantic buffers. Use a counter-based generator keyed by **seed, global parameter name and global coordinates**. Replicas must match across ranks; TP shards must be slices of the same conceptual global tensor. Values must be invariant to generation tile size. Different layer names must generate different values. Initialize bounded, quant-block-aligned tiles, with a declared temporary-memory ceiling; never materialize the entire model or expert bank in BF16.

| Parameter | Required initialization |
|---|---|
| Routed expert packed weights/scales | Generate fan-in-scaled BF16 tiles; quantize into valid MXFP4/E2M1 bytes and group-32 UE8M0 scales. Use global fan-in, not the local TP width. |
| Dense MXFP8 | Generate finite E4M3 values and explicitly initialize valid uint8 group-32 scales; avoid sentinel255 and arbitrary random byte patterns. Preserve the checkpoint/runtime scale-layout contract. |
| Engram | Fill compact FP8 rows and scales through work order 3, in bounded host tiles. |
| Other linear/router weights | Controlled finite fan-in scaling; initialize router correction biases explicitly and record them. |
| Norm and mHC | Explicit sensible norm weights; controlled mHC functions, bases and scales. Record the recipe and inspect residual statistics through all layers. Tiny uniform values everywhere are not a stability recipe. |
| Attention sinks | Initialize valid head slots intentionally; preserve invalid padded slots at `-inf`. |
| Semantic buffers | Preserve positions, masks, lookup structures and execution metadata rather than blanket-randomizing a state dictionary. |

The [MXFP4 method](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/quantization/mxfp4.py) creates uint8 weights/scales with zeros. Its native V4.1 path preserves group-32 activation boundaries and router weighting before FC2 quantization. Local intermediate 576 already satisfies its 64-element rounding requirement; do not pad to 640 as an assumed requirement.

The [ModelOpt `KMxfp8Static` scheme](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/layers/quantization/modelopt.py#L2126) creates expanded row-major uint8 scales with `torch.empty`. Its checkpoint loader expands 32×32 block scales along output rows; preserve that relationship when emulating checkpoint quantization. Reuse the finite-value and full quantized-reference approach in [native Marlin MoE tests](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/tests/kernels/moe/test_deepseek_v41_marlin_moe.py), extending coverage to E64/E72, H5120, local I576 and top6.

### Keep normal finalization

Fill pre-finalization tensors, then run the standard [post-load sequence](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/model_loader/utils.py#L97): Marlin repacking/scale conversion, dense MXFP8 runtime preparation, attention preparation and model-level finalizers. The [Ada dense kernel](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/model_executor/kernels/linear/mxfp8/ada.py#L172) transposes the weight. [`wo_a`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/quant_config.py#L50) expands to BF16; retain this allocation. Its special dummy sentinel-scale fallback does not repair general dummy initialization.

The [model finalizer](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/model.py) prepares first-layer mHC broadcast weights. The [VL wrapper](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/nvidia/vl_model.py) includes a hook for loaders that skip the child's checkpoint-loading method. Verify required finalization occurs exactly as intended.

**Acceptance:** bounded initialization peak, valid scales/nonfinite checks, chunk-size-invariant generation, matching replicated checksums and global-slice shard references, normal finalization and finite nontrivial prefill/decode outputs. **Stop:** arbitrary integer contents, zeroed expert banks, unexplained all-token routing collapse, unstable layer statistics or a bypassed native operation.

## Work order 3 — Replace Engram backing before allocation

**Owner:** Engram/host-memory engineer. **Dependency:** original metadata; can proceed alongside work orders 1–2. **Deliverable:** constructor hook, explicit mapping, fixtures and memory accounting.

[`ParallelEngramEmbedding`](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/common/engram.py#L622) allocates original-range FP8 weights and uint8 scales during construction, explicitly pinned on CPU when offloaded. The original table is approximately 188.83 GiB aggregate in this model. A later loader hook cannot prevent that allocation. Install the synthetic-backing choice **before the constructor allocates its backing tensors**.

Preserve original prime-modulus hash geometry, head ownership, original row IDs and all Engram/Wkv/gating modules. Add an explicit original-ID-to-compact-row lookup. Misses produce zero embeddings **before Wkv**. Do not modulo IDs into a smaller table or silently shrink `engram_vocab_size`; decreasing only `engram_num_embeddings` also violates the existing original-head-size capacity check. See [layout construction](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/common/engram.py#L178).

| Proposed fixture name | Backing and purpose | Required labeling |
|---|---|---|
| E0 | Zero retained rows or minimal sentinel storage; all lookups miss; preserve complete Engram computation after lookup. | All-miss semantic smoke only. Does not characterize table access. |
| E1 | Tiny retained set containing known original IDs; explicit hits, misses and mixed sequences. | Semantic correctness against a small full-address reference. |
| E2 | Physically allocated, pinned and touched **8 GiB aggregate** FP8-plus-scale backing initially; optional **32 GiB aggregate** after headroom is measured. | Capacity/UVA fixture with declared original-ID trace, hit fraction, locality and bytes actually accessed. |

E2 byte targets cover row values plus their scale payload, not mappings or other process memory. Derive row counts from row stride and report rounding. Record per-rank ownership, resident/pinned bytes and page-touch completion; an uncommitted virtual allocation is insufficient. Keep separate totals for mapping keys/values, sentinels, hash state, staging buffers and temporary initialization. State whether each map is sharded or replicated and on CPU or GPU. Do not accidentally replicate a dense full-ID map across four ranks; budget the chosen representation explicitly before allocation.

Use declared original-ID traces compatible with the retained mapping. Report unique rows touched and access skew. **Filling 8 GiB does not prove that a benchmark reads 8 GiB.** An optional synthetic address stress test may force broad access, but label it separately from tokenizer-driven traffic and compact-mask fidelity. No synthetic fixture establishes the real model's trained lookup locality.

Extend [existing Engram tests](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/tests/kernels/test_engram.py): TP head reconstruction, CPU/UVA lookup, hash-cache replay, lookback windows, prepared rows across graph breaks and collectives. Add compact-map boundary/miss tests, zero-before-Wkv comparison and concurrent request isolation. Test ordinary text boundaries, chunked prefill and decode transitions.

**Acceptance:** exact mapping/ownership, reference-agreeing hits and misses, no original table allocation, declared bounded mapping memory, successful physical pin/touch. **Stop:** original 188.83 GiB allocation, changed hash geometry, post-Wkv zeroing, invalid indices, excessive map replication or unlabeled locality substitution.

## Work order 4 — Full-model gates and storage evidence

**Owner:** integration engineer. **Dependencies:** work orders 1–3. **Deliverable:** machine-readable storage/run manifest and logs.

Produce a manifest before initialization, after initialization, after finalization and after cache/graph setup. For each rank record parameter name, shape, dtype, device, logical bytes, storage identity/pointer, storage bytes and intentional ties. Sum unique storage within each process; identical numerical pointers in different processes do not imply sharing. Confirm distinct persistent allocations for all 40 layers, apart from documented architectural ties. Never reuse one layer's weight buffer or loop a single layer and call it full-model residency.

Verify actual E64/top6 and shared-expert geometry per layer, local I576, 16 local heads, actual native cache formats, BF16 `wo_a`, allocated router tensors, mHC finalization, Engram physical/logical footprints, absent vision tower and absent speculative buffers. Inspect actual modules: a text-only request alone does not prove vision weights were omitted.

Run focused existing native tests first, retaining their reference tolerances:

- Router and Marlin MoE tests linked above, plus added pruned shapes.
- [vLLM native operations](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/tests/kernels/test_deepseek_v41_native_ops.py): quantization, compression, cache/index operations and mHC.
- Paired FlashInfer [mixed sparse attention tests](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/tests/attention/test_deepseek_v41_mixed_sparse.py): native cache packing, 16-head reference comparisons, invalid/padded slots, prefill/decode boundaries and graph replay with changed inputs. Implementation lives in [`flashinfer/mla/deepseek_v41.py`](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/flashinfer/mla/deepseek_v41.py) and [`_deepseek_v41_sparse_triton.py`](https://github.com/yhfgyyf/flashinfer/blob/d4eef3c44c70594ff18b9126381ae6a978694e3f/flashinfer/mla/_deepseek_v41_sparse_triton.py).

Proceed sequentially: native-kernel references → one full-shape synthetic layer → full 40-layer E64/E0 → E1 → physically touched E2 → concurrent mixed prefill/decode → graph replay comparison. Measure startup/finalization peaks and steady memory on every GPU; include host resident/pinned memory. Then begin the workload sweep. E72 follows only if E64 passes with measured headroom.

**Acceptance:** no illegal access/nonfinite state, consistent request accounting, expected cache growth/reuse, finite outputs through every layer, graph/eager agreement within declared tolerances and no monotonic memory growth after warmup. Record routing occupancy and Engram access statistics alongside throughput. **Stop:** full-model OOM, unexplained storage mismatch, hidden CPU weight offload, changed quantization/cache path, request-state contamination or persistent growth. Resolve the specific failure before expanding the sweep.

## Interpretation boundary

This fixture can establish native compatibility, real full-depth residency, initialization peaks, scheduler/graph behavior and preliminary performance under declared synthetic routing and memory-access distributions. It cannot establish pruning quality, trained routing skew, real Engram locality, real-checkpoint loading behavior or final production throughput. Keep these unresolved items in the handoff to the pruning team; repeat the relevant gates with actual retained weights and representative requests when available.
