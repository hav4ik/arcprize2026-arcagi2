# CED: exact positions for REAP and Engram collection

Updated 19 October 2026. This makes the 8B-prefill/16B-decode distinction operational for P01–P03 and E01–E02. It refines the existing full-position replay requirement by defining which executed positions enter each statistic. The masks below are a proposed calibration policy; native GPU parity remains an execution gate.

## 1. Which halves execute

The backbone has encoder layers 0–19 and decoder layers 20–39. Generated inputs execute both halves. The published 16B decode count includes the roughly 8B encoder work; it is not a separate 16B decoder above the encoder. Optimized prompt ingestion runs the encoder and constructs decoder global KV from its boundary states, then warms the decoder's local attention over a bounded prompt suffix. The report's text example uses 128 tokens and calls bounded replay approximate. Engram modules are at encoder layers 1 and 14.

Sources: [official report, sections 2.2 and 3.2.2](https://arxiv.org/html/2609.19969v1), [exact 1017 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json).

| Workload | Encoder | Decoder | Engram |
|---|---|---|---|
| Optimized prompt ingestion | All new prompt inputs | Bounded warmup suffix and first-output prediction | Prompt inputs at both encoder modules |
| Autoregressive continuation | Each newly consumed generated input | Same input, to predict its successor | Consumed generated inputs at both modules |
| Ordinary optimized prefill of a saved entire trace | Treats the whole trace as prompt | Skips most historical decoder positions | Addresses can be covered, but decoder REAP is incomplete |
| Full causal reference replay | All supplied inputs | All supplied inputs | All eligible supplied inputs |

The official reference `Transformer.forward` loops over all 40 backbone layers for a whole-prefix call. Its prediction head normally selects the last position, but that does not remove intermediate-layer hidden states or MoE events. Use layer observers and project selected positions only for diagnostic logits. See [native model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py).

## 2. Exact input/target alignment

For a single tool-free generation episode, let the saved tokens be `x[0:N]`, prompt length `P >= 1`, and completion-target count `R = N-P >= 1`. The targets to assess are `x[P:N]`. Input position t predicts target `x[t+1]`. The minimal consumed input stream for those predictions is therefore `x[0:N-1]`; its last index is `N-2`.

Use half-open intervals throughout:

| Stratum | Input positions | Expected positions |
|---|---|---:|
| `encoder_prompt` | `[0, P)` | P |
| `encoder_continuation` | `[P, N-1)` | R-1 |
| `decoder_generation` | `[P-1, N-1)` | R |
| Separate `decoder_warmup` for tail W | `[max(0,P-W), P-1)` | min(P,W)-1 |
| Exact-execution Engram census | `[0, N-1)`, with native text eligibility | P+R-1 before eligibility masking |

The **last prompt input** is `encoder_prompt` in the lower layers and `decoder_generation` in the upper layers: it predicts the first completion token. These are different layers and this overlap is intentional. An input-role mask alone is insufficient for decoder scoring. Store input role, prediction-target role, absolute position and physical execution phase separately.

The final completion token is normally not consumed when the episode stops. For example, a generated EOS does not need another model forward. If a continuation later consumes it, include it in that later episode's input accounting. Do not drop a final recorded token mechanically without knowing whether it was consumed. A broad census that includes the entire saved text, even its unconsumed last token, is an allowed separately named conservative coverage variant; do not call it exact execution-frequency matching.

Example: prompt `[p0,p1,p2]`, completion targets `[a0,a1,a2,a3]`. Forward inputs are `[p0,p1,p2,a0,a1,a2]`. Decoder scoring uses inputs `[p2,a0,a1,a2]`, which predict `[a0,a1,a2,a3]`. Both encoder Engram modules process the six consumed inputs once.

For the initial tool-free harness, one episode normally spans the prompt and complete reasoning/answer. If the harness resumes or re-prefills, retain explicit episode boundaries and shared-prefix identity; do not invent a new independent task or count physical cache reconstruction as additional semantic evidence.

## 3. Recommended replay implementation

### A. Full causal reference: first correctness baseline

1. Read the saved exact token sequence and episode/role metadata.
2. Supply all consumed inputs with causal masking to the corrected native reference; execute all 40 layers. Retain the frozen Engram mask for final P02b/P03.
3. Collect encoder prompt and continuation statistics separately. Their natural token-uniform union is the initial encoder-ranking measure.
4. Collect decoder-generation statistics at the target-aligned mask above. Extra decoder prompt computations create causal state but are excluded from the main decoder-generation score.
5. Retain decoder prompt/tail diagnostics separately. Do not accidentally pool every prompt decoder event just because the reference computed it.
6. Compare selected IDs, gates, outputs, cache state and diagnostic logits with token-by-token execution initialized using the same full causal reference semantics. Preserve quantization and same-precision tolerances.

Teacher forcing uses known tokens, so it need not sample every token serially. Parallel causal execution is legitimate, provided the native computation and visibility rules are preserved. Query tiling for the reference indexer is still required before long prefixes; see [REAP_HARDWARE.md](REAP_HARDWARE.md). Never discard preceding causal history merely because only a late position is scored.

Disable decoder-prefill skipping, speculative execution and prefix caching in the initial observer path. This names required behavior, not a claimed universal command-line flag. The native runtime adapter must implement and prove that behavior. The reference's nonzero-start compressor and window-cache paths assume one input token; calling `forward(long_suffix, start_pos=P)` is not an implemented multi-token continuation solution.

### B. Replay matching optimized bounded CED: separate validation path

After A is correct, compare against the actual serving schedule. Warm the decoder on its declared prompt tail and teacher-force completion inputs with decode semantics. The decoder execution interval is `[max(0,P-W), N-1)`; its generation-scoring interval remains `[P-1,N-1)`.

A custom parallel implementation can run the encoder over known inputs, build the required global KV, and process the decoder tail/continuation span. It must preserve absolute positions, causal global-KV visibility, local-attention history limits, candidate/index reuse and both boundary hidden streams and delayed mHC `pre_mix`. That is additional code with a parity gate, not a stock reference API call.

Full causal replay and bounded-tail initialization differ by design. Do not claim exact equality between them or assume the difference necessarily disappears after 128 generated tokens. Compare within each declared schedule first, then measure cross-schedule drift/quality. If deployment uses bounded CED and its calibration drift changes masks materially, collect or validate the finalist's statistics under that schedule. Keep baseline and candidate serving schedules matched during pruning-quality evaluation.

## 4. Statistics and normalization

Store sufficient statistics as `stats[layer_id][stratum][original_expert_id]`. Encoder and decoder expert IDs belong to their own layers; equal numeric IDs are not shared experts across halves. Use the exact native-amplitude/reference-amplitude definitions and gate normalization in [REAP_IMPLEMENTATION_CONTRACT.md](REAP_IMPLEMENTATION_CONTRACT.md).

Default observation weight is one per eligible, sampled semantic position. Pool the encoder's natural prompt/continuation mixture by adding amplitude sums and selection counts, then dividing. The main decoder ranking uses `decoder_generation`. Warmup statistics are a separately retained diagnostic and an optional selection-sensitivity experiment because prompt-tail state can affect generation.

If a later experiment targets phase proportions pi[p], with M[p] eligible sampled positions in phase p, use observation weight `u[t] = pi[p]/M[p]` in both numerator and denominator. Averaging phase-specific conditional expert means does not generally reproduce this mixture. Do not multiply decoder saliency by two, divide encoder saliency by two, or change native routing to compensate for 8B/16B active counts. Keep uniform E64/E72 per-layer budgets for the first candidate sweep; separate encoder/decoder budgets are a later measured experiment.

Maintain semantic observation identity separately from physical execution. A safe key includes checkpoint, Engram mask, replay schedule, trajectory, input position, layer and score definition. Tail rebuilds, duplicated batches, overlapping replay chunks, shared TP views and rejected speculative branches must not create duplicate semantic observations. Physical invocation counts are useful for performance telemetry but are a different dataset.

## 5. Engram census and optional contribution scores

Hash the consumed input-token history ending at x[t], not the next target x[t+1]. Preserve the native canonical token mapping and the preceding three token slots, including native DEAD/image barriers and padding for missing or blocked history, across prompt/continuation and chunk boundaries. Never bridge an image/ineligible barrier by finding older text tokens. Reset only according to native session rules. Count original `(module, order/head, row)` keys separately; the two encoder modules have their own hash geometry.

For exact-execution census, count each eligible input once in each module. There are up to 48 row lookup events per consumed text input across the two modules, independent of whether that input arrived during prefill or generation. Decoder replay is not another Engram lookup. A KV reconstruction path may physically rerun encoder work; its extra reads belong in performance telemetry rather than semantic row frequency. Keep optional prompt/continuation frequency strata so coverage can be checked for grid inputs and long reasoning separately.

Frequency census requires no transformer, decoder or MoE router forward. Context-aware Engram contribution scoring needs encoder hidden states and native module computation. A local module-damage surrogate can be collected there; a score based on final logits or task loss additionally needs downstream computation. In either case, freeze the selected row mask before main expert calibration, because encoder perturbations can affect both halves and decoder global KV.

## 6. Counts, cost and acceptance checks

The included [tools/ced_phase_masks_reference.py](tools/ced_phase_masks_reference.py) checks the scalar position contract. It does not implement model replay or native hashing.

With no padding, sampling or modality exclusions, the sum of raw selected-expert counts over all experts in each scored layer must equal six times that layer's scored-position count. Each encoder layer counts P+R-1 inputs. Each decoder-generation stratum counts R positions. Both Engram modules see P+R-1 eligible inputs, producing up to `48*(P+R-1)` row events. Adapt these invariants explicitly for sampled/weighted observations.

At the expert-work approximation, full replay costs roughly `16B*(P+R)` active-parameter uses. Bounded CED costs roughly `8B*P + 16B*R + 8B*W`, ignoring boundary offsets and other operators. These are workload arithmetic, not throughput forecasts. Long tool-free reasoning makes R dominant, so most relevant calibration work needs both halves. The 8B optimized-prefill speed cannot price full-layer REAP replay. There is no factor-of-two reduction in the full source checkpoint that must be loaded or streamed.

Acceptance fixtures must include: P=1, R=1, P>W, first completion target, final stop input accounting, chunk overlaps, warmup replays, native eligibility masks, and separate semantic/physical counts. GPU acceptance additionally requires short parallel/sequential parity under the same schedule, long-context memory checks, and correct per-layer observer coverage. These GPU/model checks have not been executed in this packet.
