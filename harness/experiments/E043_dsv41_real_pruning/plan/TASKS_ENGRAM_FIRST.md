# Engram first: CPU row census, validation, freeze, then expert pruning

Research plan updated 19 October 2026 for DeepSeek-V4.1-Flash-1017. This file specifies implementation work; no row-pruned model, GPU forward or ARC benchmark was executed in preparing the pack.

## Current decision

Use **tool-free reasoning as the required pruning/calibration objective**. The user supplies approximately 80% pass@2 without tools and 84% with tools as unpruned-model context; these are not measurements from this pack. No tool-assisted control or general coding-retention requirement is added. Run **CPU row census → select candidate rows → validate model behavior → freeze the Engram mask → collect fresh REAP statistics with that mask active → E64 export**, with E72 as a larger quality candidate. Keep top-6 routing and the shared experts unchanged.

Engram and routed-expert storage budgets can be chosen independently. Their effects on hidden states and subsequent expert routing are coupled. A smaller Engram table is not justified solely by high lookup hit rate. Select and validate masks against the actual tool-free workload; tool-assisted specialization is an optional later branch only if the user chooses it.

This plan uses **ordinary domain-specific row retention**. It does not require tensor decomposition, new hash moduli, table requantization or a new training architecture.

## Phase E00 — Pin encoding and define held-out roots

**Outputs:** `locks/engram.json`, `data/roots.jsonl`, `data/splits.json`, `data/traces_manifest.jsonl`.

Pin the exact 1017 model revision, tokenizer files, chat/role encoding, Engram compressed-token mapping and hash helper. Save the actual compressed-token map arrays and their checksums, not just the tokenizer name. Use a proposed 10,000-task coverage pool, seeded by existing compatible traces, and verify how many independent roots/generator lineages it contains. Split roots into census/calibration, model-selection validation and final audit before augmentation or retries. Keep color permutations, rotations, generator descendants and any variants of a root together; add a generator-family holdout where provenance permits.

Record root ID, generator lineage, target harness, prompt-template revision, token-file checksum, success/failure label, role boundaries and trace length. Keep the complete tool-free token history with exact formatting. Existing compatible API text is enough to start; expert activations are not required for address census. Tool trajectories are not a prerequisite for this plan.

**Gate:** trace encoding round-trips through the deployed tokenizer and message formatter; no validation/audit root contributes to row selection. Do not concatenate unrelated traces and accidentally hash across their boundary.

## Phase E01 — Reproduce native row addresses on CPU

**Outputs:** `tools/engram/census_rows.py`, `tests/engram/test_hash_parity.py`, `fixtures/engram/hash_cases.jsonl`, `runs/engram/hash_parity.json`.

At each token position, addresses use the canonical last 2, 3 and 4 compressed token IDs, with the reference's exact beginning-of-sequence padding/history rules. There are **2 Engram layers × 3 orders × 8 heads = 48 banks/lookups per token**. Identify a row by `(original_layer_id, ngram_order, head_id, original_bucket_id)`. Preserve the original layer IDs 1 and 14, prime table sizes, multipliers, offsets and compressed token map. Never replace the original modulus with the retained row count.

Port the existing helper to a CPU batch/streaming implementation. Retain at most the last three preceding canonical token IDs across chunks, and reset or restore history according to the native request/session semantics. Addresses depend only on token IDs; Engram gates and resulting contributions require hidden states. A CPU count cannot determine that a frequently requested row has a large gate, or that a rare row is unimportant.

Test native/helper address equality on real ARC traces, beginning/end boundaries, punctuation and numbers, special/role tokens, repeated n-grams, long traces, chunk cuts and target-harness continuation. The source bounds multiplication below INT64_MAX; intentional integer overflow is not its hashing mechanism. Preserve these bounds, integer operations and tokenizer normalization explicitly. Validate the fixture against the actual runtime hash path when GPU integration becomes available.

Count only Engram-eligible input positions: ds4 may return addresses even where the native token mask suppresses Engram. A normal role delimiter does not itself reset n-gram history; use exact native request/continuation semantics. Validate every address against its original module and head-bank interval. Invalid global/head IDs are errors, distinct from valid rows omitted by pruning or owned by another rank.

**CED accounting:** both Engram modules are in the encoder, which runs on prompt and consumed generated inputs. Use the exact consumed-input/next-target alignment in [CED_CALIBRATION.md](CED_CALIBRATION.md). Do not count decoder warmup as another lookup, shift hashes onto prediction targets, or multiply counts when physical cache rebuilds rerun the same position. Including an unconsumed final output token is an allowed named conservative-coverage variant, not exact execution frequency.

**Gate:** exact original bank-row IDs agree on the fixtures; arbitrary chunking does not change the census.

**Reusable implementation:** [DS4_REUSE.md](DS4_REUSE.md) identifies `ds4_engram_hash()` plus `compressed_token_map()`/`engram_layout()` as a standalone CPU base for this phase. Load exact 1017 metadata; no GGUF or model GPU is needed. Its `mask` parameter controls token eligibility/history and is not the row keep-mask. A census wrapper and actual 1017 parity fixtures still need implementation.

## Phase E02 — Stream counts and select rows under a global budget

**Outputs:** `stats/engram/<harness>/counts.*`, `stats/engram/coverage.json`, `masks/engram/<candidate>/keep_ids.*`, `masks/engram/<candidate>/manifest.json`.

Count original row lookups on CPU before fetching table weights. Stream token shards and avoid retaining a list of all token-row events. The source has 768,022,850 rows across its two tables. A full uint32 count array is approximately **2.861 GiB**; uint64 is approximately **5.722 GiB**. Separate arrays per harness multiply this footprint. Bankwise or memory-mapped arrays and reduction of shard-local counts can keep RAM bounded. Choose count dtype from the maximum corpus multiplicity and detect overflow; weighted statistics require a separate declared representation.

For up to 1B tokens there are up to 48B lookup events. This is CPU work, not free work: benchmark hash/count throughput on a representative 1–10M-token shard, record wall time and peak RAM, then estimate the full census from measured tokens/s. Repeated n-grams can be counted once with multiplicity if that produces identical row counts. Do not rent GPUs for token-only counting.

Use this first selector:

Default to one count per eligible token lookup, with no per-bank frequency normalization. For equal 264-byte row costs, global top-frequency retention maximizes retained lookup count on this census. It does not maximize whole-token coverage or task accuracy. Root-uniform/role-weighted counting is a separately named objective.

1. Determine a **global native row-payload budget** from the measured host limit, leaving room for maps, pinning, workers, token batches, gather outputs and model loading.
2. Each retained row costs **264 bytes** in the source format: 256 FP8 values plus eight scale bytes. Thus `max_rows = floor(payload_budget_bytes / 264)`.
3. If the global union of observed `(bank, original_row_id)` pairs fits, retain all of it. Otherwise keep the highest-frequency rows globally; break equal counts deterministically by original bank and row ID.
4. Store per-bank compact segments and offsets. Retained banks may have different sizes. Equal per-bank fractions are an optional comparison, not a constraint of the global budget.
5. Report target-harness row-hit rate, all-48-lookups-hit rate, unique unseen rows, miss bursts and metrics by layer/order/depth/role. Compute validation metrics with a frozen candidate; never silently add validation rows to it.

If `U = U2 + U3 + U4` distinct canonical n-grams are observed, the global retained union is at most `16*U` rows and at most **4224*U bytes**. Collisions reduce this bound. It is not an equal-per-bank capacity bound. Different n-grams can share one row: retain/count the row once and accumulate all its requests. Pruning a row affects every colliding n-gram that uses it.

An expert-proportional table budget can be a baseline: E64/384 would retain approximately **31.47 GiB**, while E72/384 would retain **35.41 GiB** of native row payload. It is not a law or a required fraction. Choose the actual row budget from coverage, host feasibility and held-out model quality. Retaining a fraction of rows does not proportionally reduce the 48 lookup requests per token.

**Gate:** exact count totals and deterministic keep sets agree with a simple fixture; retained payload and map overhead fit the proposed host budget. Quality remains unproven at this gate.

### Optional alternative to frequency selection: local contribution scores

Frequency is the required first selector. If needed later, define the proposed local damage for one accessed row r as `D[t,r] = mean((G_layer(h_t, z_t) - G_layer(h_t, z_t with slot r zero))**2)`, over the full native `[hc_mult,dim]` module output. Hold incoming h fixed, use the native projection/activation quantization/output cast, and recompute affected keys, values, per-HC gate normalization and signed-sqrt gate. Do not normalize across HC copies when the native gate normalizes over dim separately for each copy.

The default proposed selector ranks by **sum of D over accesses** under token-uniform observation; a common global-token denominator gives the same order. Save access counts and conditional `sum(D)/access_count` as a different, frequency-agnostic diagnostic. Copying REAP's conditional denominator changes this Engram objective. Do not rescale retained embedding vectors by inverse frequency, inverse retained fraction or token hit rate. Bare embedding norms or gate-times-row norms do not capture the complete module change.

This is a proposed local surrogate, not an existing Engram REAP method or a task-loss guarantee. Single-row damages need not add when many rows are removed. Raw cross-layer damage ranking is also a declared surrogate; it is not proven to represent equal task importance across layers. Validate whole masks. Float64 damage sums plus uint64 counts over all 768,022,850 original rows require 11.44 GiB; adding a second moment requires 17.17 GiB, before census/maps. Budget those separately and shard by bank or visited rows.

This needs an instrumented model forward and Engram-specific states/statistics. It does not require MoE expert-routing statistics unless the same pass also collects a separate expert-pruning baseline. Reduce the added cost by first using the CPU census to exclude unseen rows and restricting contribution scoring to a representative replay subset. Once an Engram mask is chosen and validated, collect the final REAP expert statistics with that mask active; statistics from the original table are not automatically equivalent.

## Phase E03 — Implement compact original-ID lookup and export retained rows

**Outputs:** `patches/engram/compact_lookup.patch`, `tools/engram/export_rows.py`, `engram/<candidate>/tables.*`, `engram/<candidate>/manifest.json`, `runs/engram/row_parity.json`.

Use the native SM89 `cpu_offload:true` pinned-UVA path as the starting runtime. Add a lookup adapter from original bank-row IDs to compact retained rows. The baseline loader assumes original table geometry, so changing only a configuration size is insufficient. Avoid allocating or pinning the full original table before replacement. Fetch retained safetensor byte ranges or the necessary source shards and preserve selected FP8 values and scales exactly.

The supplied `tools/compact_engram_reference.py` is a CPU-tested synthetic mapping/byte-preservation fixture with a zero sentinel and TP4 ownership. It does not implement native hashing, streaming safetensor export, GPU dequantization or vLLM integration.

**Reusable export/I/O:** [DS4_REUSE.md](DS4_REUSE.md) maps the existing bounded row reader and streaming `write_engram()`/`SourceDB.iter_read()` into this phase. The writer currently exports every row; add retained-ID selection and compact metadata. Its row reader returns F32 storage of BF16-rounded values, so use native packed GPU lookup or validated BF16 staging rather than doubling transfer bytes unintentionally.

A sorted retained-ID index plus compact offsets or an explicit original-to-compact map are reasonable first implementations. Record where each index lives and its actual bytes; a uint32 map over all original rows costs another 2.861 GiB globally, independent of retention. A uint32 retained-ID list costs four bytes per retained row, excluding search structures. Python dictionaries are not a compact representation at this scale. TP ownership is by the original banks/heads; preserve it through compaction and all-gather.

Declare the initial miss behavior as **zero the missing 256-dimensional embedding row before Wkv/projection**, then run the unchanged native projection, gate and combination code. Zeroing the final Engram residual or treating a miss as a hit to an arbitrary row is a different model. Ensure missing IDs never become invalid memory accesses. Do not make miss behavior depend on whether a row happens to remain in the OS page cache.

Validate retained real row bytes/scales, first/last and empty-bank cases, original/compact offset boundaries, TP ownership, collision cases, missing rows, repeated requests, chunk/prefix lookback and changed-input graph replay. With a mask retaining every fixture row, output must agree with the original arithmetic path. With missing rows, compact output must agree with a full-table reference explicitly zeroing the same rows before projection.

Bulk table bytes remain in CPU RAM. **Engram Wkv/projection weights, ID metadata located on GPU, gathered embeddings, staging tensors, transfer buffers and any GPU hot cache still count toward VRAM or transient peaks.** Measure host RSS, pinned bytes, per-rank allocation, lookup latency and output-buffer lifetime. Disk staging is an optional fallback if compact pinned storage is still infeasible; it must preserve the same frozen mask and miss policy.

**Gate:** the compact representation faithfully executes the declared mask and fits measured host/GPU limits. This proves implementation parity, not task-quality retention.

## Phase E04 — Validate physical model quality and freeze the mask

**Outputs:** `eval/engram/<candidate>/metrics.json`, complete rollout logs, `masks/engram/frozen/manifest.json`, `eval/engram/selection_decision.md`.

Use original full-expert 1017 weights and the P02a forward from TASKS_PRUNING.md to isolate the row-mask effect. P02a can be developed before this mask is frozen; P02b validates its final configuration afterwards. On a small representative replay set, compare full-original Engram and candidate mask using logit/loss changes and native Engram contributions. Then run complete held-out tasks at intended reasoning lengths. Begin with a manageable panel, for example 32–64 diverse roots, and expand survivors. This small panel screens large failures; it does not establish a small accuracy-loss margin. A 2K-token smoke completion or high row-hit rate cannot validate 64K reasoning.

Evaluate the physical masked model on the tool-free workload. Report exact-grid success, valid completion, loops, generated length and deadline failures. Maintain **table-bytes → lookup coverage → actual solve accuracy** curves. Candidate serving comparisons on the same local hardware can use equal wall-clock; equal-token comparisons are diagnostics. An off-site API/full-teacher control establishes quality-retention deltas, not comparable local serving speed. No Python-tool comparison is a required gate.

Select a row budget using predeclared model-selection validation criteria and paired root-level uncertainty. Keep final audit roots untouched. If row masking damages useful reasoning, increase retained rows or change the training-corpus coverage and rerun this gate before expert calibration. Original Engram paper experiments do not certify post-hoc V4.1 domain-row deletion.

Freeze model revision, tokenizer/hash revision, census split/checksum, selection rule, per-bank original IDs and offsets, row bytes/scales checksums, map representation, miss policy, cache arithmetic, harness settings and validation decision. Use a content checksum as `engram_mask_id`.

**Gate:** the chosen mask has acceptable complete-task quality and feasible measured memory. Lookup coverage alone cannot pass this gate.

## Phase E05 — Handoff to fresh REAP, then E64/E72

**Outputs:** frozen Engram manifest, `stats/reap/<run>/provenance.json`, and the TASKS_PRUNING.md pilot artifacts.

Keep all 384 routed experts while collecting the first new expert statistics, with the frozen Engram mask active. Reuse existing teacher token sequences; regenerating the entire proposed 10,000-task pool is unnecessary. **API transcripts do not provide REAP scores**: observe gates and individual expert outputs through an instrumented weight forward at all required decoder positions.

Run **0.25–1M replayed tokens** to verify implementation and measure cost. Then target **1–8M scored positions** if rank/quality uncertainty warrants it. Scored tokens, causally processed tokens and raw generated tokens are different counts. Late-position scoring must include its actual preceding context; replaying all 200–640M tokens in a hypothetical 10,000-long-trace corpus is not a mandatory first run. Preserve the native quantization caveats in TASKS_PRUNING.md when choosing REAP or the explicitly labeled native-amplitude adaptation.

Export E64 first and evaluate E72 if useful. Every expert-statistics, mask, export and evaluation manifest must identify the frozen Engram mask. If retained rows or miss semantics change, revalidate Engram and collect fresh affected expert statistics. Keep a small joint validation panel to detect interactions. Choose retained experts from measured importance on the target workload; preservation of broad coding capability is not an acceptance requirement.

## Primary sources and limits

- [Exact 1017 config and table geometry](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)
- [1017 Engram hash helper](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/engram.py)
- [1017 model: embeddings, projections and gates](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py)
- [Pinned SM89 native Engram CPU offload and TP gather](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/blob/3f3bbaea916707c851ca027b87b97fb6605998b8/vllm/models/deepseek_v4_1/common/engram.py)
- [Official Engram paper](https://arxiv.org/html/2601.07372v1) and [V4.1 report](https://arxiv.org/html/2609.19969v1)
- [Existing V4.1 REAP infrastructure](https://github.com/Libertai/deepseek-v41-flash-reap)

The compact lookup adapter and this domain-row pruning experiment are proposed work. These sources establish architecture and reusable components, not a demonstrated quality or performance result for the proposed deployment.
