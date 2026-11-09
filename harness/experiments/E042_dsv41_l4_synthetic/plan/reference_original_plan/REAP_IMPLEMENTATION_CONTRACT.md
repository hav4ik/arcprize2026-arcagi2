# REAP implementation contract and source audit

Audited 19 October 2026. This document resolves scoring and integration choices left open in the earlier packet. Read it with TASKS_PRUNING.md before implementing P02/P03. It describes a proposed 1017 calibration adapter; no checkpoint-dependent or GPU parity result is claimed.

## 1. Pinned evidence and reuse boundaries

| Source | Inspected revision | Reuse |
|---|---|---|
| [Canonical REAP](https://github.com/CerebrasResearch/reap/tree/1970473c51ca3caeb98c10392f15b3a08a672974) | `1970473c51ca3caeb98c10392f15b3a08a672974` | Score definition, selection/export ideas; its generic observer is not a native V4.1 adapter |
| [REAP paper](https://arxiv.org/html/2510.13999v3) | v3, sections 3.2 and 4 | Conditional saliency and simplified local motivation; not a guarantee for E64/E72 |
| [Native V4.1 REAP infrastructure](https://github.com/Libertai/deepseek-v41-flash-reap/tree/210ccebac92172fb58655c7b95f6a6776f57a641) | `210ccebac92172fb58655c7b95f6a6776f57a641` | Packed expert streaming, source tensor addressing, export scaffolding; corrections below are required |
| [HOPE](https://github.com/awslabs/HOPE/tree/f335c52d6dd2e97a3d2611c08b950386d43fb529) | `f335c52d6dd2e97a3d2611c08b950386d43fb529` | Co-selection statistics and solver reproduction, optional after REAP |
| [Exact 1017 native model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py) | HF revision `1bae0e1088993b9b82de08f59de5be5a34e42170` | Gate, Expert, Engram, CED and quantization source of truth |

The downloaded exact model.py SHA256 is `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`. Save local source checksums and the complete environment in each run; a repository name or moving branch is insufficient provenance.

## 2. Four different meanings of normalization

### Native routing normalization: preserve it

For text tokens, use the native FP32 router input and its configured temperature:

`s = sqrt(softplus(linear(h, gate.weight) / gate_temp))`

Choose top-6 using `s + gate.bias`. After pruning, exclude deleted IDs from selection and refill from retained IDs. Compute mixture weights from **unbiased** scores of the new selected set:

`w = 1.5 * s[selected] / (sum(s[selected]) + 1e-20)`.

The bias affects selection, not mixture values. Preserve `bias_vl` in physical exports even though tool-free text calibration uses the text path. Do not introduce softmax, use biased scores as mixture weights, or normalize over all 384 original experts. Keep the checkpoint's configured temperature, score function, top-k, norm flag, route scale, clamps, RMSNorm and mHC operations. A deterministic tie convention must match the selected runtime; the scalar fixture uses lowest original ID, whereas generic PyTorch top-k does not promise that tie rule.

### REAP's observation denominator: conditional mean

For original expert ID e, selected-position indicator I[t,e], amplitude a[t,e], and calibration observation weight u[t]:

`A[e] = sum_t u[t] * I[t,e] * a[t,e]`

`D[e] = sum_t u[t] * I[t,e]`

`REAP[e] = A[e] / D[e]` for observed experts.

Default `u[t]=1` for every valid scored position. This is token-uniform over the declared scored-position set. Divide by selections of that expert, not the global token count, sum of router weights, number of sequences, or number of batches. A rare expert with conditional amplitude 10 outranks a frequent expert with conditional amplitude 2, even if the frequent expert's total contribution is larger.

[CED_CALIBRATION.md](CED_CALIBRATION.md) defines the valid position sets and phase strata. Full-layer execution and score eligibility are separate masks: score encoder prompt/continuation naturally, decoder generation by the next target's role, and warmup separately. The 8B/16B active counts do not introduce an additional score normalization.

Also retain `A[e]` as the explicitly named `contribution_total` diagnostic. Comparing its keep set with REAP costs no extra forward. This is a frequency-sensitive alternative, not another implementation of conditional REAP. Keep uniform per-layer retention budgets initially; do not globally rank raw amplitudes across layers or apply per-layer z-scores to invent comparability.

### Expert-output norm: measure before aggregating

Use an L2 norm over the complete output feature dimension for each selected token/expert, computed in FP32 or better. Measure after the down-projection and before routed summation, the shared expert, residual addition or mHC expansion. Do not unit-normalize the output vector, replace it with a weight norm, take the norm of the average output, or multiply mean gate by mean output norm. Average the actual per-observation product.

For a full gate-independent reference `f_ref(h) = Expert(h, weights=None)`, define `a_ref = w * ||f_ref(h)||_2`. The constant 1.5 scales every reference REAP score equally within a layer and does not change its ranking. It remains part of the actual forward. Dividing native gates by 1.5 inside a quantized expert and rescaling afterwards is not an equivalent execution rule.

### Data weighting: a separate explicit choice

Root-uniform weights, role weights, inverse-probability corrections for stratified sampling and caps on repetitive trajectories all change the calibration measure. Record them. Weighted conditional means require weighted denominators as well as weighted numerators. Preserve raw counts independently. If stratified sampling is intended to estimate the original corpus measure, record inclusion probabilities and use the appropriate observation weights; otherwise describe the sampled distribution as the calibration target. Do not silently claim a balanced sample is token-uniform over the entire source pool.

## 3. Preserve the quantized model while observing it

The official Expert multiplies the route weight into its SwiGLU intermediate **before** the cast and down-projection activation quantization. Consequently, with q denoting the native expert computation:

`q(h,w)` need not equal `w * q(h,1)`.

At the pinned LibertAI revision, `scripts/reap_calibrate.py::_patch_moe` calls `expert(x, None)`, observes `g * norm(out)`, and advances the model with `y += g * out`. Its comment assumes linear w2 makes that equivalent to the native forward. The intervening cast/activation quantization invalidates that shortcut in general. The changed forward is used even when collection is disabled.

**Required correction:**

```python
native_out = expert(hidden_for_expert, native_gate_column)
y.index_add_(0, token_indices, native_out.float())

if observe_native:
    amplitude_native = native_out.float().norm(dim=-1)
if observe_reference:
    reference_out = expert(hidden_for_expert, None)
    amplitude_reference = native_gate * reference_out.float().norm(dim=-1)
# Only native_out advances the model. Do not multiply it by the gate again.
```

This is a semantic sketch, not a complete distributed patch. Reuse the up/gate projections and pre-weight SwiGLU intermediate if this is shown equivalent; rerun the weighted down-projection when gates change. The simple two-call reference is acceptable for correctness fixtures.

Use two explicit names: `reap_ref_amp` for conditional means of `a_ref`, and `reap_native_amp` for conditional means of `||q(h,w)||_2`. The cheap production pilot defaults to native amplitude, with both collected on a small correctness/quality subset. Select a final method from the named candidates; never mix their accumulators or describe the native variant as an exact reproduction of gate-independent REAP.

Observer-on and observer-off runs must execute the same native forward and produce the same outputs under the established same-precision tolerance. Reference-amplitude calculations and local ablations must not overwrite teacher activations, Engram rows, attention state or CED caches.

## 4. Exact sufficient statistics and reductions

For each layer and declared stratum, save expert weighted amplitude sum A, weighted denominator D, raw selection count N, weighted second-moment numerator Q, and maximum amplitude. Also save raw/weighted scored-position counts and stable shard IDs. Float64 accumulated sums and int64 raw counts are the baseline. Make chunk-local reductions accurate enough to agree with the CPU contract; reject nonfinite inputs rather than silently sanitizing them into pruning evidence.

Merge numerators and denominators, then divide **once**. Do not average worker/chunk means equally. Example: chunks [1] and [3,5] have pooled mean 3, not 2.5. Checkpoint by immutable completed shard ID; reject duplicate merges.

TP/EP ranks may observe the same tokens. Count each original expert's selected contribution once per data observation. A full expert output owned by one EP rank can be measured there. If ranks hold partial sums of the same output vector, sum the vector before computing its norm. If ranks own disjoint feature slices, sum squared norms then take a square root. Adding local norms is wrong. Data-parallel workers may merge independent sufficient statistics after source/Engram/observer manifests agree.

Unobserved experts receive zero in the exact upstream-style ranking and carry `observed=false`. An observed zero-amplitude expert is a different case. Log coverage and uncertainty around each retention boundary. Any minimum-count protection, smoothing or rare-expert reserve is a separately named alternative; none is silently inserted into the baseline.

The included `tools/reap_stats_reference.py` supplies a CPU sufficient-statistics/router contract and deterministic fixtures. It does not supply GPU hooks, native quantization kernels, a tokenizer, a model loader or an exporter.

## 5. Optional HOPE: pair normalization and solver details

If HOPE is enabled, save actual co-selection sums:

`P[e,f] = sum_t u[t] * I[t,e] * I[t,f] * a[t,e] * a[t,f]`

`C[e,f] = sum_t u[t] * I[t,e] * I[t,f]`.

The pinned implementation uses `F[e,f]=P[e,f]/C[e,f]` where observed and zero otherwise. This is conditional co-selection normalization, unlike a global-token Gram normalization. Apply u **once** to the outer product; putting u into each amplitude incorrectly squares the sample weight. `F[e,e]` is the conditional second moment, not the square of the conditional REAP mean.

Do not reuse canonical REAP's `pairwise_expert_frequency`: at the audited revision it stores `count_e + count_f`, not joint selections. Its `OnlineStatsTracker` does not supply variance/M2 despite its docstring; `max_activations` is a signed coordinate maximum, not an output-norm maximum. Use the explicitly defined accumulators here.

For unweighted observations, FP64 pair sums plus int64 pair counts across 40 layers and 384 experts use 90 MiB per stratum. Separate weighted denominators add another 45 MiB if raw pair counts are also retained. Do not allocate token-by-384-by-384 tensors. Form only the selected 6-by-6 products.

Reproduce the pinned uniform initialization and SLSQP relaxation first. Record solver success/status/message, constraint violation, continuous objective and rounded discrete objective. The upstream result does not enforce successful termination. Conditional normalization need not preserve positive semidefiniteness; neither SLSQP nor a single start certifies a global optimum. Solver variations and global-token normalization are named alternatives, not silent improvements to the reproduction.

## 6. Natural assumptions and limits of the rationale

REAP's local rationale treats an expert as a gate-independent output vector, ranks important contributions on representative inputs, and motivates replacing low-impact experts through top-k refill. It does not prove that errors remain small after deleting most experts across 40 layers or that rare reasoning behaviors survive domain calibration. It also ignores vector direction in the scalar saliency; cancellation, redundancy and replacement quality require actual counterfactuals and generation.

The simplified substitution bound uses promoted gate <= removed gate. With unbiased monotone top-k scores and a common positive scale, that relation survives selected-set renormalization. DeepSeek, however, selects with correction biases and mixes using unbiased scores, so selection order need not imply that relation.

Counterexample with top-2 for illustration: unbiased scores [1,2,3], selection biases [4,2,0]. The initial selection is IDs 0/1, with unit gates 1/3 and 2/3. Delete ID 0; ID 2 is promoted with gate 3/5, larger than the deleted gate 1/3. Native scale 1.5 multiplies both sides and does not repair the inequality. This affects the theoretical transfer, not the ability to compute the scores.

There is also an apparent normalization ambiguity in paper section 3.2 Eq. 7 when its promoted gate is interpreted as the actual post-prune normalized gate. Do not transcribe that expression as a numerical oracle. For a fixed linear expert reference, recompute the two actual normalized mixtures directly. For native V4.1, execute `q(h,w)` at each candidate gate and compare actual routed sums.

For clarity, if p_j < 1 is the removed unit gate (as with positive-score top-k for k >= 2), r is the replacement's score divided by the OLD selected-score sum, Z=1-p_j+r, q=r/Z, and v is the old surviving mixture, the exact fixed-vector difference is:

`delta = p_j*f_j - q*f_i - ((p_j-q)/(1-p_j))*v`.

The fixture raw scores [6,3,2], top-2, delete score 3, with survivor output 1 and both other outputs 0, gives difference -1/12. Native quantization is a separate reason to prefer an actual masked forward over this linear diagnostic. All native normalizations, shared experts and mHC remain intact during pruning.

## 7. Concrete upstream integration fixes

1. **Generic REAP observer:** it rebuilds selection using top-k logits/softmax and evaluates all experts on all tokens. Use native selected IDs/gates and sparse selected-output observation instead. A `[384,T,5120]` activation tensor is outside this plan.
2. **LibertAI native forward:** fix the weighted-down-projection issue in section 3 before using either calibration or its PPL runner as a native control.
3. **Batch validity:** the upstream text calibration pads the final batch by duplicating its last sequence. Exclude artificial replicas and token padding with an explicit observation-validity mask. A modality/image mask is not an observation-validity mask.
4. **Single-GPU scope:** upstream sets CUDA device 0 and its patched MoE loops all experts without distributed reduction. `torchrun` does not make it a valid multi-GPU collector. Implement rank-local expert ownership and reductions, or keep the streamed pilot single-GPU.
5. **Long replay:** the official reference already executes all 40 backbone layers for whole-prefix prefill. It normally projects only the last position at the head; hook pre-head states/project selected positions for diagnostics. Its `start_pos>0` continuation branches assume one token. Tile the indexer's query axis inside whole-prefix execution before attempting 64K reference calibration; see REAP_HARDWARE.md. Arbitrary multi-token continuation is separate work.
6. **HF exporter:** `prune_experts.py` includes every retained w1/w2/w3 weight and scale and slices gate.weight/bias/bias_vl, but copies full Engram and buffers complete output shards. Adapt it to the frozen compact table and bounded memory.
7. **Native MP exporter:** `prune_convert_mp.py` creates `model{rank}-mp{n}.safetensors`, skips Engram expecting external backing, and retains DSpark by default unless configured otherwise. This is not a ready vLLM HF checkpoint with compact Engram.
8. **Verification:** upstream `verify_prune.py` samples w1/gate.weight at four layers and can use different modality-score pooling from export. Use the actual keep manifest and verify every retained packed tensor/scale and all router/bias tensors. Native pruning and Marlin repacking have separate parity gates.

## 8. Dependency correction and stop conditions

Split P02 into **P02a: original-model replay/observer scaffolding**, which can start immediately and supplies the Engram E04 quality comparison, and **P02b: final replay parity with the validated frozen mask**, which precedes main expert calibration P03. This avoids a circular requirement for a validated mask before building the forward needed to validate it.

Do not launch large calibration until native observer-on/off parity, actual per-layer position counts, mask identity, long-context memory behavior and measured pilot throughput pass. Do not export an E64/E72 candidate as a successful result until its complete-task behavior and target runtime pass the existing quality and memory gates. These are experiments to perform, not missing literature that needs to delay the CPU/adapter work.
