# ARC specialist pruning: terminal-agent work orders

Research date: 2026-10-19. Target source: **DeepSeek-V4.1-Flash-1017**. Target deployment: **4×NVIDIA L4, E64 as the primary expert budget and E72 as a larger candidate, with a validated pruned Engram table on CPU**. This document specifies work to implement and measure; it does not claim that the proposed pruning ratios, runtime, or recovery have been demonstrated.

**Implementation audit:** read [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md) for pinned source findings, exact normalization/aggregation, the native-forward correction, and export caveats. [REAP_HARDWARE.md](REAP_HARDWARE.md) provides the RTX PRO6000 Blackwell calibration plan and the required long-indexer work. [START_HERE.md](START_HERE.md) assigns work that can begin immediately.

## 1. Outcome and boundaries

Make **64 routed experts per layer (E64), top-6 activation, unchanged shared experts** the primary physical export target. Evaluate E72 as a larger candidate if its measured per-rank residency and batch capacity fit. Test E96/E80 as quality controls for the original hypothesis that removing 75–80% of routed experts preserves ARC performance. E48/E40/E32 remain optional experiments from the earlier smaller weight envelope; they are not the current required deployment targets.

Complete the CPU census and validate the Engram row mask described in [TASKS_ENGRAM_FIRST.md](TASKS_ENGRAM_FIRST.md) **before collecting the main REAP statistics**. Freeze its original-ID mapping, retained row bytes and missing-row behavior. Collect expert statistics with this exact Engram variant enabled. Engram rows and MoE experts are separate parameter sets, but changing an Engram contribution can change later hidden states, routes and expert norms. Use the same frozen Engram mask across the initial expert-budget sweep; retain the full-table model as a separate quality control. API-generated transcripts can be reused for this local replay without regenerating all trajectories after row pruning.

Use the **tool-free harness as the required pruning/calibration objective**. The user reports approximately **80% pass@2 without tools and 84% with tools** for unpruned DeepSeek; these are supplied baseline context, not verified results of this pack. No Python-tool control or preservation of general coding capability is required. A tool-assisted specialization remains an optional future branch only if the user chooses it.

The [official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) and [technical report](https://arxiv.org/html/2609.19969v1) describe a 40-layer backbone, 384 routed experts per layer, top-6 routing, hidden width 5120 and expert intermediate width 2304. They also introduce a substantial Engram component outside the backbone. Whole-expert deletion changes stored capacity; retaining top-6 does **not** reduce active expert arithmetic in proportion to the number of experts deleted.

Derived sizing, using `3 × 5120 × 2304 = 35,389,440` parameters per expert per layer:

| Retained experts/layer | Routed experts removed | Routed parameters | Approx. backbone parameters¹ | Estimated GPU weight bytes² |
|---:|---:|---:|---:|---:|
| 24 | 93.75% | 34.0B | 42.4B | 32.6 GB / 30.4 GiB |
| 32 | 91.67% | 45.3B | 53.7B | 39.0 GB / 36.3 GiB |
| 40 | 89.58% | 56.6B | 65.0B | 45.4 GB / 42.3 GiB |
| 48 | 87.50% | 67.9B | 76.4B | 51.7 GB / 48.2 GiB |
| 56 | 85.42% | 79.3B | 87.7B | 58.1 GB / 54.1 GiB |
| **64** | **83.33%** | **90.6B** | **99.0B** | **64.49 GB / 60.1 GiB** |
| **72** | **81.25%** | **101.9B** | **110.3B** | **70.86 GB / 66.0 GiB** |
| 80 | 79.17% | 113.2B | 121.7B | 77.2 GB / 71.9 GiB |
| 96 | 75.00% | 135.9B | 144.3B | 90.0 GB / 83.8 GiB |

¹ Backbone estimates use the rounded published 552B baseline; replace them with the tensor inventory from P00. They exclude Engram and auxiliary heads. The original Engram has approximately 196.6B logical parameters; the retained amount is `256 × retained_original_rows`, determined by the separately validated row mask. Report combined parameter counts from both manifests rather than equating the expert-removal fraction with total model removal.

² Planning assumption: Marlin routed storage of 0.5625 bytes/parameter plus 12.6 GiB of other resident components. These numbers exclude KV, any GPU Engram hot rows, workspace, transient repacks and allocator slack. The runtime agent must replace them with measured per-rank allocations. The earlier 32–56 decimal GB target is no longer a hard acceptance condition; E64/E72 must instead pass the actual concurrency, memory and wall-time gates on 4×L4.

The major methodological references are [REAP](https://github.com/CerebrasResearch/reap), [HOPE](https://github.com/awslabs/HOPE) and the existing [LibertAI V4.1 implementation](https://github.com/Libertai/deepseek-v41-flash-reap). Published pruning results at 25–50% removal do not establish quality at the 81.25–83.33% removal proposed for E72/E64. This plan uses measurement gates, with no invented probability of success.

## 2. Work products and dependency map

All names below, except explicitly identified upstream files, are **proposed project files to implement**. They are not commands or capabilities already provided by those repositories.

| Task | Depends on | Principal output | Gate before advancing |
|---|---|---|---|
| P00 — Pin source and inventory | None | `manifests/source.json`, `manifests/tensors.parquet`, `docs/native_contract.md` | Source revision, layouts, routing and CED behavior understood |
| P01 — Baseline and corpus | P00 tokenizer/encoding | `data/splits.json`, `data/trajectories.jsonl`, `eval/baseline.json` | Root-task isolation and complete harness metrics |
| P02a — Original-model replay and observer scaffolding | P00; small P01 sample | `pruning/native_adapter.py`, `pruning/observers.py`, original-model parity report | Native forward works; supplies Engram E04 comparison without requiring a frozen mask first |
| P02b — Final replay/observer parity | P02a; validated frozen Engram mask from E04 | `eval/replay_parity.json` | Full-position replay agrees with native decoding under the selected mask |
| P03 — Calibration pilot and statistics | P01, P02b; frozen Engram manifest | `stats/<run>/`, `cost/replay_benchmark.json` | Correct counts, stable statistics, measured cost and matching Engram checksum |
| P04 — Candidate selection | P03 | `masks/<method>/E*/keep.json`, solver records | Feasible, auditable original-ID keep sets |
| P05 — Fixed-input falsification | P02b, P04 | `eval/local_damage.parquet`, optional oracle report | Shortlist candidates worth full execution |
| P06 — Selective original-ID loader | P00, P04 | `pruning/subset_loader.py`, `eval/subset_parity.json` | Same model as the full-source mask, without reading all experts |
| P07 — Progressive end-to-end evaluation | P01, P06 | `eval/compression_curve.json`, complete rollout logs | E64 or E72 viable, or a documented stop/revise decision |
| P08 — Optional recovery | P07; backward-path proof | `recovery/`, `eval/recovery_ablation.json` | Useful held-out ARC gain at justified cost |
| P09 — Physical export and runtime handoff | P07 or P08; runtime router support | `exports/E64/`, optional `exports/E72/`, `manifests/export.json`, parity reports | Packed-tensor preservation and runtime equivalence |
| P10 — Frozen Engram audit and handoff | Engram-first validation; monitor alongside P03/P07 | `stats/engram/`, `eval/engram_drift.json` | Original-ID semantics, unchanged mask and measured miss behavior |

Each run must record checkpoint SHA, tokenizer/reference-encoder SHA, code SHAs, data-manifest hash, **frozen Engram manifest checksum**, precision, observer definition, random seeds, hardware, billed hours and evaluation protocol. The Engram manifest must identify the retained IDs and bytes, original hash geometry, lookup map and fallback semantics. Keep source weights immutable. Write large outputs atomically and checkpoint statistics by completed corpus shard.

## P00 — Source inventory and native contract

**Implement:** `pruning/inventory.py`; inspect the pinned native implementation and the actual upstream scripts before adapting them.

Start with the source-audited [LibertAI V4.1 repository](https://github.com/Libertai/deepseek-v41-flash-reap/tree/210ccebac92172fb58655c7b95f6a6776f57a641): `scripts/stream_model.py`, `scripts/reap_calibrate.py`, `scripts/prune_experts.py`, `scripts/prune_convert_mp.py`, `scripts/verify_prune.py`, `scripts/verify_shards.py`, `patches/patch_engram.py` and `patches/patch_kernels.py`. Its released derivatives used the September V4.1 revision. Reuse infrastructure and collect fresh **1017 ARC statistics**, applying the specific forward, batch-validity, single-GPU, exporter and verifier corrections in REAP_IMPLEMENTATION_CONTRACT.md. The audit establishes code behavior; actual-weight numerical parity is still required.

Inventory tensors by layer, original expert ID, logical shape, packed shape, dtype, quantization group, scale tensor, shard offset and byte length. Separate routed experts, shared experts, routers/biases, attention, embeddings/head, Engram and optional auxiliary components. Record actual source and runtime byte counts independently of logical parameter counts.

`docs/native_contract.md` must answer:

- How are unbiased mixture scores calculated, how is selection-only bias applied, and where are coefficients normalized and scaled? Record the native route scale of 1.5 and the fact that gate weights enter before the expert down-projection's activation quantization.
- Which router paths exist by layer/modality? What is the deterministic tie rule? Are there routing-group or expert-placement constraints?
- Which expert outputs are available before or after gating and TP reduction? Which outputs need reduction before their norms are meaningful?
- Which CED positions are skipped by optimized prefill? How can every required prediction position execute both halves of the model?
- What cache state, position offsets, compression state and encoder-output references must persist across replay chunks?
- How are quantized weights/scales linked after loading? Audit meta construction and dtype preservation; do not assume `load_state_dict(assign=True)` preserves every module's intended compute dtype or attached scale attributes.

**Gate:** a short official/native reference forward is finite, uses the intended encoding, and agrees with the proposed calibration path within a measured same-precision tolerance. Record the reference tolerance rather than choosing a loose constant to hide disagreement.

## P01 — Reproducible ARC corpus and evaluation baseline

**Implement:** `data/build_manifest.py`, `eval/run_harness.py`, `eval/summarize_runs.py`.

Split by immutable **root ARC problem ID before augmentation** into calibration, model-selection validation and final audit. All color permutations, rotations, reflections, retries and harness variants of a root stay in one partition. For synthetic families, record generator lineage and add a generator-family holdout where feasible. Do not repeatedly tune on the final audit partition.

The [NVARC repository](https://github.com/1ytic/NVARC) identifies these candidate puzzle sources:

| Dataset ID | Published contents | Use in this project |
|---|---|---|
| `sorokin/nvarc-synthetic-puzzles` | 103k synthetic puzzles | Candidate root problems for fresh 1017 solver trajectories |
| `sorokin/nvarc-augmented-puzzles` | Subsets containing 3.2M augmented puzzles | Use only after recovering root/augmentation lineage |
| `sorokin/nvarc-artifacts-puzzles` | Generated text used to construct puzzles | Construction provenance; do not label it 1017 solver reasoning |

Audit the bundled data-building and pretraining mixtures for any inclusion of public evaluation roots before declaring a holdout clean. Persist a source→root→augmentation mapping and apply exclusions by root. A random split of dataset rows cannot establish independence. This is an audit requirement, not a claim that a particular dataset has contaminated the proposed split.

Trajectory records must contain the root ID and lineage, full prompt, generated tool-free reasoning, final grid, exact token IDs/reference encoding, checkpoint and tokenizer revisions, harness revision, sampling settings, verifier outcome and stop reason. Store role spans and prediction-position alignment. Never concatenate unrelated trajectories without resetting native causal state.

For a proposed **10,000-task coverage pool**, verify root/generator independence and preserve separate calibration, model-selection and final-audit partitions before any census or mask selection. One 20k–64k-token trajectory per root already represents 200M–640M trajectory tokens; additional trajectories multiply that amount. CPU row-address counting can cover this broad corpus without executing the transformer. Use only the calibration partition to fit row counts/keep sets, use validation to choose budgets, and reserve final-audit roots for the frozen candidates.

Create a smaller **GPU calibration coverage pool of 256–1024 root problems × 2–4 trajectories**, drawn from the calibration partition and the tool-free target harness. This is a pool target, not a requirement to generate or replay its entirety before testing the collector. At 20k–64k output tokens per trace, the full pool would contain roughly 10M–262M output tokens. Begin with existing compatible traces and a small sample. The 1B-token allowance remains a reserve for recovery and failure-directed data. Tool-assisted trajectories are not required for this pipeline.

An API transcript is sufficient for CPU row-address reconstruction once tokenization, serialization, normalization and native hashing agree. It does not contain the hidden states, route weights or expert outputs needed by REAP. Reuse saved transcripts for a local forward under the frozen Engram mask; neither regenerating the entire proposed task pool nor teacher-forcing every corpus token is a prerequisite for the first expert-pruning experiment. Missing API-only internal tokens or a mismatched chat template must be recorded as a coverage limitation.

For the first complete calibration, aim for **1–8M scored positions**, stratified across early/middle/late tool-free reasoning, prompt grids and final formatting, including relevant long histories. Include successful runs and productive failed searches in calibration; use a stricter verified-data filter when constructing SFT targets. Record and bound the contribution of unproductive loops. Save strata separately so workload weights can be changed without replay.

**Baseline metrics:** exact-grid solve rate; the actual two-attempt policy if used; solved tasks under final wall-time and token limits; output-token distributions; valid-grid, missing-answer and timeout rates; repetition/termination behavior. Preserve complete tool-free runs, including failures. Record the full-table original model and the full-expert model with the selected Engram mask separately. Measure pruning loss against the same target solve policy; a separate tool-assisted control is not required.

**Gate:** task IDs are disjoint across partitions, replay reproduces stored tokenization, and the unpruned teacher completes representative full harness runs. Report uncertainty using root-level paired resampling; multiple seeds/augmentations from the same root are not independent tasks.

## P02a/P02b — Native full-position replay and sparse observers

P02a begins with original full-expert model behavior and supports the Engram E04 quality comparison. P02b repeats the required final parity checks with the chosen frozen Engram mask before P03. Collector development does not wait for the mask it will help validate.

**Position contract:** implement [CED_CALIBRATION.md](CED_CALIBRATION.md). For P prompt inputs and R completion targets, consume P+R-1 inputs. Encoder prompt/continuation strata have P and R-1 positions; each decoder-generation stratum has R positions beginning at the last prompt input. Extra prompt-decoder computations needed for causal state are not automatically part of the generation score. Preserve warmup diagnostics separately. Treat full causal and bounded-CED initialization as different declared schedules and validate within each before comparing them.

### First prove the CED schedule

The [V4.1 report](https://arxiv.org/html/2609.19969v1) makes prefill and decoding structurally different. A stock fast-prefill replay can collect many encoder observations while skipping decoder observations for most positions. API reasoning text also contains no expert activations. It must be replayed locally through an instrumented path.

The official whole-prefix reference already runs all 40 backbone layers at every supplied position; only its head defaults to last-position logits. Start there for correctness. Its nonzero-start continuation assumes one token, and its unchunked indexer can allocate a 256 GiB score tensor at 64K. The first long-reference implementation tiles query work **inside Indexer.forward** while preserving whole-prefix causal computation, as specified in REAP_HARDWARE.md. Arbitrary multi-token continuation is a separate implementation gate. In a native-runtime adapter, start eager with CED skipping, DSpark and prefix caching disabled so observers see the intended positions.

Define `prediction_position`, `target_token_index` and target role explicitly; verify the one-token alignment against actual decoding. Build a full-position teacher-forced/logprob path, or another demonstrated equivalent schedule, that evaluates every scored position in **all 40 required layers**. Assert counts per layer and role. Do not infer correctness from an encoder-only token count.

On a short transcript, compare actual one-token decoding against full-position replay and chunked replay for logits, selected original expert IDs, gates, output norms and aggregate statistics. Initialize identical full causal state, precision and CED mode for this parity comparison. Approximate bounded decoder replay in a production fast-prefill mode is a separate behavioral change, not an error to hide by loosening the observer tolerance. Repeat across a chunk boundary and target-harness continuation. Include a long-position smoke case before scaling. Preserve official cache/compression/CED state rather than treating this as an ordinary decoder-only blockwise model.

Sparse scoring reduces observer work and storage; a late scored position still requires computation of its causal history. Report both total replayed positions and scored positions, and distinguish encoder/decoder execution if the schedule performs unequal work.

### Observer contract

For a scored token chunk in one layer, obtain `ids[T,6]`, nonnegative native routing weights `w[T,6]` and individual expert contributions before the routed sum. The normalized unbiased scores are multiplied by the configured route scale: native weights sum to **1.5**, not 1. Use `p=w/1.5` for a unit-normalized retained-mass diagnostic, but leave the native forward weights intact.

**Quantization changes the observer contract.** In the [official reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py), `Expert.forward` multiplies `weights` into the SwiGLU intermediate before dtype conversion and `w2`; the quantized linear path then quantizes that intermediate for its GEMM. Consequently the native contribution `q_e(h,w)` is not guaranteed to equal `w × q_e(h,1)`. Dividing a weighted output by its gate does not recover a gate-independent expert function.

Implement and distinguish two amplitude definitions:

- **Native contribution amplitude:** `a_native = norm(q_e(h,w))`. This is cheap if individual contributions are exposed. Name the resulting conditional-mean/pair methods `reap_native_amp` and `hope_native_amp`, because they adapt the paper statistics to the actual quantized contribution.
- **REAP reference amplitude:** choose the explicit gate-independent reference `f_ref(h)=Expert.forward(h, weights=None)`, then `a_ref = w × norm(f_ref(h))`. It requires an unweighted down-projection evaluation; reuse the up/gate projections and pre-weight SwiGLU intermediate where possible. This reproduces the weighted-unweighted-output score definition on a declared reference computation, rather than claiming exact factorization of the native model.

Collect both on the correctness/pilot subset, measure their ranking differences and incremental cost, and select the named definition using held-out pruning results. Do not silently mix their statistics. The sparse accumulator below accepts either definition as `a`; its manifest must state which one was used.

**Audited upstream correction:** LibertAI currently advances its model with `g * expert(h, None)` even when collection is disabled. Preserve `expert(h, g)` as the actual forward; the optional unweighted reference serves scoring only. Do not double-weight a native contribution. The canonical generic REAP observer also reconstructs top-k/softmax and evaluates every expert on every token; use native IDs/gates and sparse output observation instead. See REAP_IMPLEMENTATION_CONTRACT.md for the exact replacement and source pins.

Do not norm unreduced TP partial contributions and then add the norms. For a feature-sharded output, squared norms can be reduced across disjoint features; for partial contributions to the same full output, reduce the vector first or reconstruct it correctly.

The [REAP score](https://arxiv.org/html/2510.13999v3) is a **conditional mean over tokens selecting the expert**, with the amplitude definition stated above:

`reap[e] = sum(a[t,j] where ids[t,j] == e) / selection_count[e]`.

Using total routed mass, frequency or a global-token mean produces a different criterion. Keep counts, conditional first/second moments and maxima so rare or insufficiently observed experts can be detected.

The practical [HOPE implementation](https://github.com/awslabs/HOPE) collects products for **co-selected** experts and divides each pair by its own co-selection count. Preserve this practical definition when reproducing it; the theoretical global-token normalization in the [paper](https://arxiv.org/pdf/2609.18916) is a different normalization.

Pseudocode, intentionally independent of a particular kernel or tensor layout:

```python
# Logical shapes only; implement against the audited native hook.
# Padding/unscored positions have already been removed.
# ids: int64 [T, 6]; a: float [T, 6]; E = 384.
flat_ids = ids.reshape(-1)
expert_sum.index_add_(0, flat_ids, a.reshape(-1))
expert_sum_sq.index_add_(0, flat_ids, (a * a).reshape(-1))
expert_count += bincount(flat_ids, minlength=E)

pair_ids = (ids[:, :, None] * E + ids[:, None, :]).reshape(-1)
pair_values = (a[:, :, None] * a[:, None, :]).reshape(-1)
pair_sum.reshape(-1).index_add_(0, pair_ids, pair_values)
pair_count += bincount(pair_ids, minlength=E * E).reshape(E, E)

# At finalization, not after every chunk:
reap = divide_where_observed(expert_sum, expert_count)
F = divide_where_observed(pair_sum, pair_count)
```

Use bounded chunks, initially 1k–4k scored positions. This creates `T×6×6` pair contributions, not dense `T×384×384` tensors or all-expert hidden states. Across 40 layers, FP64 pair sums plus int64 pair counts require approximately **90 MiB per statistics stratum**. Many strata multiply that amount; flush and merge as needed. Chunk-local FP32 reduction into FP64 is acceptable only after comparison with an accurate reference.

Required invariants: six unique selected IDs; finite nonnegative gates summing to the native scale; summed native contributions reproduce the teacher routed output; correct observer-amplitude definition; symmetric pair statistics; diagonal pair counts equal expert counts; no padding counted; the expected scored-position coverage for each layer and phase/stratum; no duplicate shard incorporation. For each layer/stratum, sum of unweighted selected-expert counts must equal six times its eligible scored positions. For a complete trace with P prompt tokens and R generated target tokens, the combined encoder prompt/continuation strata cover P+R-1 consumed positions per encoder layer, while each decoder-generation stratum covers R positions; decoder warmup remains separate. Executing all 40 layers does not imply identical scoring masks. With observation weights, also check weighted selection denominators against six times the sum of eligible observation weights. Zero-count experts/pairs must be recorded as unobserved. Match upstream zero-count handling for the reproduction, then make any coverage floor, protected rare-expert policy or smoothing a separately named ablation.

**Gate:** the optimized sparse collector agrees with a small, deliberately simple reference collector, and chunked/full replay agrees with actual decoding. This gate precedes large calibration or pruning.

## P03 — Costed pilot and one representative calibration pass

**Implement:** a resumable runner using either the corrected single-GPU packed layer streamer or the instrumented resident SM120 runtime from REAP_HARDWARE.md. Store sufficient statistics, not complete hidden-state histories. The upstream text runner is not automatically resumable and its duplicate final-batch entries must be excluded from observation.

Run the calibration reference with **all original 384 routed experts and the validated, frozen Engram row mask enabled**. In P03–P07, “teacher/reference statistics” mean this full-expert, row-pruned reference unless a run is explicitly labeled as the full-table control. If the Engram mask changes, assign a new manifest checksum and collect the affected expert statistics again; do not merge statistics across different masks as though they described one model. An Engram mask reduces conditional-memory storage but does not make the full-expert backbone fit on 4×L4; retain the measured streaming/rented-calibrator plan.

First process approximately **0.25–1M replayed tokens**, with a few genuinely long traces and smaller cases for diversity. Record source reads, host/GPU peaks, layer-load time, Engram gather time, forward time, observer time, total elapsed/billed time and successful scored positions. Verify a compatible kernel path on the chosen GPU before committing to the complete run.

The [LibertAI reference](https://github.com/Libertai/deepseek-v41-flash-reap) demonstrates packed layer streaming, but its published measurements used the forward shortcut identified in this audit. They do not establish native 1017 parity, 64K memory or RTX throughput. Prefer **one 96GB RTX PRO6000 Blackwell for corrected streamed development**, then a four/eight-card resident implementation if its measured cost is favorable. Enough host RAM and fast local source storage can matter more economically than the lowest hourly GPU rate. No optimizer states are needed for inference-only scoring. The streamed reference and multi-GPU vLLM paths have different kernel requirements; do not launch the existing single-GPU runner under torchrun unchanged.

Measure cost using:

```text
hours = total_replayed_positions / measured_replay_positions_per_second / 3600
cost  = hours × all_worker_hourly_price + storage + transfer
```

A cold full routed-weight sweep is approximately 269 GiB in the native layout. For read bandwidth `B GiB/s`, the sweep alone takes at least `269/B` seconds. This is a scheduling lower bound, not end-to-end throughput. Include causal history, activation limits, repeated masks, teacher scoring and generation when forecasting the experiment.

Grow to the 1–8M scored-position target only after the pilot passes. Save bootstrap keep-set overlap, rank uncertainty near each retention boundary and coverage by root/harness/position. If token-weighting or role-weighting is used, record both weighted numerators and their corresponding weighted conditional denominators; silently replacing the denominator with a global count changes the method.

**Gate:** further calibration is justified by unresolved rank/quality uncertainty. Do not process 1B tokens solely because the allowance exists.

## P04 — Auditable keep sets and solver reproduction

**Implement:** `pruning/select_experts.py` with method-independent output schema:

```json
{
  "source_revision": "pinned-source-sha",
  "stats_manifest": "hash",
  "engram_manifest_sha256": "frozen-validated-engram-manifest-hash",
  "method": "reap-or-hope",
  "amplitude_definition": "explicit-reference-or-native-contribution",
  "experts_per_layer": 64,
  "top_k": 6,
  "ids_are_original": true,
  "layers": {"0": {"keep": ["sorted original integer IDs"]}},
  "solver_and_coverage_metadata": {}
}
```

The strings in this schema are descriptive placeholders, not literal valid keep IDs. Generate REAP keep sets by retaining the highest conditional means. Generate HOPE sets using the observed pair matrix and the budgeted relaxation:

`min pᵀ F p; 0 ≤ p ≤ 1; sum(p) = 384 - E`, then remove the largest entries of `p`.

Reproduce the pinned [HOPE solver](https://github.com/awslabs/HOPE) first. Its paper uses SLSQP; record convergence, objective, equality residual, ties and rounded objective. A conditional pair matrix need not preserve the positive-semidefinite structure of an unconditionally normalized Gram matrix. Do not interpret one numerical solve as a proven global optimum. If additional starts or swaps improve the discrete objective, report them as an explicit implementation variation.

Generate **E192, E128, E96, E80, E72 and E64** candidates from the same full-expert statistics under the frozen Engram mask. Generate E48/E40/E32 only if pursuing the optional earlier smaller-envelope branch. REAP masks can be nested by ranking; independently optimized HOPE masks need not be nested. Keep per-layer budgets uniform for the first experiment. Delay unrestricted cross-layer allocation until there is evidence about layer sensitivity and the runtime can represent variable budgets.

On a small diagnostic subset, include frequency and seeded-random baselines. They are useful for detecting a broken observer, not a substitute for the main methods. Consider refill-aware [RAZOR](https://github.com/nick7nlp/Razor) only where the main methods differ materially; its top-(k+1) instrumentation requires individual outputs and an additional expert evaluation, so cost it separately. A fixed-vector refill formula is not automatically exact for V4.1's weight-dependent activation quantization; compare it with the actual native weighted down-projection before interpreting its damage estimate.

**Gate:** all layers have the requested number of valid unique original IDs; top-6 remains feasible; every mask is reproducible from its saved inputs. No export yet.

## P05 — Cheap fixed-input falsification and optional capacity oracle

**Implement:** `pruning/local_counterfactual.py`.

While an original teacher layer is loaded, retain the teacher input and compute counterfactual mixtures for several masks on sampled positions. Apply the mask to **selection eligibility**, refill top-6 from retained experts, then use the original unbiased scores and native normalization for their mixture. Removing deleted contributions from the teacher's original top-6 without refill evaluates the wrong model.

```python
# Specification, not an implementation of native score details.
# native_mixture passes rescaled gates INTO the native expert computation.
raw_mixture_scores, selection_scores = native_router_scores(teacher_input)
teacher_ids = native_topk(selection_scores, k=6, tie_ids=original_ids)
teacher_mix = native_mixture(teacher_input, teacher_ids, raw_mixture_scores)

for candidate in candidate_masks:
    masked_selection = selection_scores.clone()
    masked_selection[..., ~candidate.keep] = -infinity
    candidate_ids = native_topk(masked_selection, k=6, tie_ids=original_ids)
    candidate_mix = native_mixture(
        teacher_input, candidate_ids, raw_mixture_scores
    )
    accumulate_damage(teacher_mix, candidate_mix, teacher_residual_scale)

# Continue the reference forward with the original teacher output/state.
# Counterfactual outputs never update the teacher's CED or attention caches.
```

Use bounded token chunks and reuse the union of needed expert evaluations where practical. Reuse the up/gate projections and pre-weight SwiGLU intermediate, but **rerun the weighted down-projection/activation-quantization path when candidate weights change**. Rescaling a previously quantized teacher contribution is not exact native refill. A candidate-specific mixture can be accumulated while experts are streamed; never materialize outputs for all 384 experts at every token.

Record per layer, role and position bin:

- Original top-6 normalized gate mass retained, including lower-tail quantiles.
- Number of original selected experts retained, fraction of positions replacing all six, and concentration of the new gates.
- Joint mixture-output RMS error, relative error against teacher mixture RMS, and perturbation relative to the complete block/residual. Guard interpretation when the teacher mixture norm is tiny.
- Error by task success/failure and by late reasoning position, not only the overall mean.

There is no universal retained-mass threshold that proves success or impossibility. High mass can still preserve the wrong computation; low mass can be harmless if substitutes are good. These comparisons are exact local counterfactuals **at teacher inputs**, not end-to-end pruned-model results.

### Optional fixed-vector oracle: a limited routing/capacity diagnostic

For 256–1024 reservoir positions at a few sensitive layers, evaluate the declared unweighted references `f_ref` for all retained experts, then solve:

`min_{w ≥ 0, sum(w)=1.5} ||Σ_e w_e f_ref,e(h_teacher) - teacher_native_routed_output||²`.

Subtract any shared-expert output consistently. Solve from the small `E×E` Gram matrix in bounded chunks. Allowing all E retained vectors is a relaxation of a top-6 mixture **within this fixed linear-mixture diagnostic**. It does not lower-bound every possible native frozen-expert router behavior, because native activation quantization depends on the route weights.

First measure the difference between the native teacher mixture and the linear reference mixture at the actual teacher gates; repeat at candidate gates. If this discrepancy is material, do not interpret the oracle as a sharp capacity test. Low surrogate error with high actual refill error motivates native gate-repair experiments. High surrogate error is evidence about the fixed reference span, not a proof that native router-only changes or a globally adapted network cannot work. Actual native refill, end-to-end drift and full rollouts remain the decision evidence.

**Gate:** select at most two or three budget-relevant masks for the expensive generation stage, while retaining enough larger-budget controls to identify the quality cliff. Do not export eight giant checkpoints to obtain this diagnostic.

## P06 — End-to-end masks through a retained-original-ID loader

**Implement:** `pruning/subset_loader.py`, initially in the correct reference runtime.

A full-source mask still requires the original source weights. Once keep sets exist, read **only retained expert tensor extents and scales** from the immutable source. Keep the original 384-dimensional router temporarily, exclude removed IDs from selection, and map retained original IDs to the loaded expert objects. This enables complete candidate inference before physical reindex/export.

Do not load all 384 experts and discard most afterward. Prove actual source-read reduction with I/O counters. E64 has one-sixth as many routed expert bytes as the source if the storage reader truly filters reads; E72 retains three-sixteenths. Whole-file decompression or a loader that touches full shards can defeat this benefit.

First compare the selective loader against the fully loaded masked source for selected IDs, gates, outputs and logits. Then execute complete teacher-forced sequences so later layers receive the **candidate's own perturbed states**. This is distinct from P05. Preserve the same frozen Engram manifest, missing-row behavior, lookup path and numerical precision in both variants.

**Gate:** numerical parity with the full-source mask on short, chunk-boundary and long-prefix cases; no fallback to loading deleted experts; reduced bytes actually read. A correct slower reference path is acceptable here while the serving agent implements the production kernel path.

## P07 — Progressive quality gates and a decision record

Evaluate the primary ladder **384 → 192 → 128 → 96 → 80 → 72 → 64**, holding the validated Engram mask fixed. E48 → E40 → E32 is an optional extension for the earlier smaller-envelope experiments. First use small paired validation panels to locate the quality cliff; expand full rollout evaluation only for the controls and finalists that resolve the decision. Each expert mask is derived directly from the full-expert reference unless explicitly labeled as an iterative-pruning experiment.

For each stage, record fixed-prefix NLL/KL, reconstruction drift with depth and complete P01 behavioral metrics on tool-free generation at intended reasoning lengths. Short fluent outputs and perplexity alone do not establish quality retention. Use matched roots/seeds/attempt budgets and paired root-level uncertainty. Compare local candidates under the same wall-clock budget and hardware, with equal-token results as a diagnostic. An off-site full-teacher/API reference establishes quality deltas, not comparable local speed. No tool-assisted control is a gate.

Plot tool-free solve rate against retained experts and Engram bytes, showing the unpruned baseline and pruning loss. Compare lookup coverage at equal Engram storage and actual task quality at the fixed solve policy. General coding preservation and separate tool/no-tool calibration are outside the required first pipeline.

Predeclare the minimum acceptable ARC solve-rate/runtime tradeoff in `configs/acceptance.yaml`, using the original tool-free DeepSeek baseline and the actual deployment execution budget. If that product threshold is unavailable, report a Pareto curve and a decision conditional on explicit thresholds; do not invent a success probability or universal quality cutoff.

Interpretation:

- E96/E80 test the initial 75–80% routed-expert-removal hypothesis and provide larger quality controls.
- E64 is the primary physical target; E72 is a larger candidate whose additional weight residency must earn a measured quality/runtime benefit.
- E48/E40/E32 are optional earlier smaller-envelope experiments, not acceptance requirements. E24 is a later stress test only if justified.
- If E64 repeatedly fails complete tasks that E72/E80 solve, combine that evidence with local/oracle damage and error growth before committing to recovery.

Run a small paired **2×2 control** on the finalist budget: full versus pruned experts, each with full versus pruned Engram. This isolates interaction between the two deletions without generating every control across the proposed 10,000-task pool. After joint pruning, log newly generated missing rows and route drift on a small fresh rollout panel. If those observations justify a changed Engram mask or iterative pruning, validate the change and refresh the required statistics; do not automatically replay the entire corpus.

**Gate:** produce `eval/decision.md`: export, try a small recovery, change the memory/runtime approach, or stop this pruning branch. State the supporting measured evidence and the remaining uncertainty.

## P08 — Optional recovery, with a small initial budget

Start only after P07 shows enough useful behavior or the combined P05 diagnostics motivate a measured native repair experiment. The [Router-KD paper](https://arxiv.org/html/2603.02217v1) distills next-token output distributions while updating routers; it is not expert-ID imitation and does not establish rescue after catastrophic collapse.

1. **Prove gradients on a small case.** Frozen experts still participate in the backward path to earlier routers. Inference-only FP4 kernels may lack that path. Compare an autograd-capable small/reference representation before choosing hardware or promising training memory.
2. **Run a 1–5M-position pilot.** Try router projections first, with a modest learning-rate sweep and held-out termination/ARC checks. Keep teacher targets, loss definition and precision recorded. If only text targets exist, label the experiment SFT rather than output-distribution KD.
3. **Handle selection-only bias explicitly.** Because its effect is only on hard top-k indices, ordinary next-token backpropagation does not automatically provide a useful gradient to that bias. Keep it fixed initially, calibrate it separately, or implement a validated surrogate as a separate ablation. Adding it to the optimizer is insufficient evidence of repair.
4. **Use local fitting when full backward is blocked.** Layerwise router fitting on cached teacher inputs/expert outputs can be cheaper; evaluate it separately because local reconstruction is not end-to-end KD.
5. **Escalate only on gains.** Compare router-only recovery with shared-expert/attention-projection LoRA if needed. Full-expert updates and quantized merging are later, more expensive branches.

Do not store dense full-vocabulary teacher distributions at every position of 1B tokens. Stream teacher scoring or cache selected positions. Sparse top-k probabilities plus tail mass support an explicitly approximate/coarsened loss; they are not the full original KL. Track teacher inference, replayed history, student forward/backward and repeated evaluation in the cost ledger.

If iterative pruning is tested, use a labeled sequence such as 384→192→96→64, keep the validated Engram mask fixed, collect fresh statistics after each recovery stage, and compare against a direct E64 prune with a matched total compute/data budget. A changed Engram mask is an additional intervention requiring its own manifest and validation. Do not attribute an improvement solely to staging when it consumed much more training.

**Gate:** useful held-out ARC improvement over the unrepaired candidate without worse termination or effective solve rate. Grow successful recovery to tens/hundreds of millions of tokens, and only then consider the 1B allowance. Stop if gains flatten or the target remains clearly uncompetitive.

## P09 — Physical E64/E72 export and production-runtime handoff

**Implement:** adapt the audited upstream pruning/export tools; create an export manifest and standalone parity checks.

For every layer, sort retained original IDs and record the old→new mapping. Slice every routed expert matrix, scale and quantization metadata; slice corresponding router rows and selection biases, including any modality-specific bias. Preserve shared experts, activation top-k, routed scale, score transformation and normalization. Apply any routing-group constraints discovered in P00. Sorted original IDs preserve relative lowest-ID tie order after compaction.

Preserve source packed tensors/scales bitwise wherever expert deletion permits. Update the model config and safetensors index. Link the export to the exact frozen Engram manifest and verify its retained bytes and fallback behavior. Separate native expert pruning, Engram row compaction and Marlin repacking: each transformation gets its own manifest, hashes, size accounting and parity report. Removing MTP or vision components requires a loader-supported configuration and a workload that does not use them.

The proposed [SM89 fork](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) was audited by the serving workstream at commit `3f3bbae`; its `vllm/models/deepseek_v4_1/nvidia/router.py` dispatch accepted only `(128,3)` and `(384,6)`. **Every proposed top-6 retention count needs runtime validation**, including E32/E64. Do not infer support from a different engine's list of expert counts. Let the runtime agent own whitelist/kernel extension, padding semantics, EP/TP layout and SM89 numerical/performance validation.

**Export gates:**

- Every retained packed tensor and scale equals the intended source or separately documented recovery output.
- Masked source, selective original-ID candidate and physically compacted model agree on routing and logits within the established precision tolerance, with the identical frozen Engram manifest enabled.
- Any padded experts are permanently ineligible and do not perturb normalization or tie behavior.
- The exact 4×L4 runtime loads the model and reports per-rank resident, peak and transient allocations.
- The final engine completes representative long-context ARC harness runs within the real memory/time limits.

## P10 — Frozen Engram telemetry and runtime integration

The row census, keep-set selection, mask validation and compact-table construction belong to [TASKS_ENGRAM_FIRST.md](TASKS_ENGRAM_FIRST.md) and precede the main P03 calibration. This work order audits that the resulting Engram behavior remains consistent throughout MoE pruning and export. Engram is a separate conditional-memory object in the [official model](https://arxiv.org/html/2609.19969v1); deleting experts does not shrink its table automatically.

Verify CPU token-to-row addresses against native gathers, including tokenizer remapping, original hash parameters, per-head offsets, moduli and row IDs. Address counting requires no expert activations; measuring a row's hidden-state-dependent contribution does. A compact table requires an original-ID→packed-row mapping; recomputing a smaller modulo changes the model.

Record per-bank hit rates, whole-token and full-trajectory coverage, missing-slot counts, bytes fetched and lookup latency for the tool-free target harness and by context position. Preserve the frozen missing-row fallback, such as a zero vector for only the absent bank's embedding, throughout calibration and export. Keep compact retained rows on CPU as the baseline runtime design. Any GPU hot cache stores retained rows and counts toward the actual resident memory budget; silently fetching deleted rows from the full source would evaluate a cache instead of the pruned model.

On a fixed token prefix whose required rows are all retained, the lookup values match the original table when retained bytes and computation are unchanged. A missing row can alter later hidden states and routes; this is why P03 uses the frozen mask. Fresh candidate-generated trajectories may address rows absent from the original teacher corpus. Log this shift on a small panel, and revise a keep set only through a new validation cycle rather than allowing the final holdout to train an adaptive table.

**Gate:** the same validated Engram checksum appears in statistics, candidate masks, evaluation and export; original-ID lookup and fallback behavior agree; CPU storage/RAM and measured lookup latency fit the deployment environment. A successful expert-pruning result alone is not a deployable model.

## 3. Final handoff bundle

The terminal agents should deliver the source/encoding manifest, disjoint data splits, replay-parity evidence, per-layer statistics and coverage, all finalist keep sets, solver diagnostics, local/oracle damage, complete validation rollouts, measured cost ledger, recovery ablations if any, export manifests and exact runtime launch configuration. `eval/decision.md` must answer:

1. Does E64, or the larger E72 candidate, meet measured per-rank residency and useful batch capacity on 4×L4 after the actual repack, compact Engram integration and component inventory?
2. What is the measured ARC score and number of solves within the real execution budget relative to the teacher and current Qwen baseline?
3. Is the limiting factor expert capacity, routing, Engram, attention/serving, or a combination supported by the measurements?
4. What additional experiment would change the decision, and what is its measured-cost estimate?

The deliverable is complete when it contains either an exported, validated target or enough measured evidence to stop the branch economically and choose the next one. A mask, a checkpoint file, or a perplexity number by itself does not meet that standard.
