# E043: real-weight pruning of DeepSeek-V4.1-Flash-1017 (Engram rows + MoE experts)

**Goal (user, 2026-10-23):** implement the plan pack's Engram-first pruning on the rented 8x RTX PRO 6000 box:
frequency-prune Engram rows, then REAP expert statistics with all 384 experts, E64 (primary) / E72 candidates.
Calibration data: `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (tool-use rollouts, E031/E036/E038).
Quality: the 120 ARC-AGI-2 eval tasks with the same harness, full vs pruned on the same server.
The plan pack (the user's upload, unzipped) is in `plan/` (START_HERE.md, REAP_IMPLEMENTATION_CONTRACT.md,
TASKS_ENGRAM_FIRST.md, ...). Deployment counterpart: E042 (session dsv4-infer).

## Summary (2026-10-23)

Reports: `reports/pruning_results.html` (ARC evals, routing statistics, NLL screens, probes, throughput) and
`reports/engram_census.html` (Engram rows). The plan pack the work followed is in `plan/`.

| model (ARC-AGI-2 eval, 167 pairs, 1 attempt, same tool harness, self-hosted) | pass@1 | task-weighted |
|---|---:|---:|
| full DeepSeek-V4.1-Flash-1017 (384 experts, full Engram) | 65.9% | 67.4% |
| 384 experts + Engram 64 GiB (34% of rows) | 69.5% | 69.7% |
| E72 w30 + Engram 64 GiB + `--rescue-code` | 29.3% | 28.6% |
| E64 w30 + Engram 64 GiB + `--rescue-code` | 18.0% | 17.5% |
| E48 w30 + Engram 64 GiB + `--rescue-code` | 0% (0/120) | 0% |

- **Engram row pruning is free.** A 64 GiB top-frequency table, from a 252M-token census of the training rollouts,
  scores like the full table on ARC, and its NLL is within run-to-run noise. 93% of the remaining misses come from
  novel n-grams in the model's own reasoning text.
- **Expert pruning to the sizes that fit 4xL4 (E48-E72) costs most of the accuracy.**
  - Routing is spread over 83-196 *effective* experts per layer, so keeping 48-72 covers only 53-65% of the routing traffic.
  - At these sizes the DSML tool-call format collapses first: 0/10 valid first-turn calls at E72, even though
    teacher-forced NLL rose only ~24%.
- **The tool-call format can be fixed; the reasoning damage cannot.** Weighting tool-call-block tokens x30 in the
  expert score, plus a harness safety net, restores tool use.
  - E72 then keeps the puzzle's main idea in most traces but makes execution errors: 45% of its tool calls fail
    (full model: 2.4%) and many answers are near-misses.
  - E64 shows semantic damage, e.g. hallucinated module names.
  - E48 never leaves its first turn.
- **Things that did not help.**
  - Iterative pruning (re-observing under each mask) lost to one-shot statistics at every size.
  - Plain REAP (mean amplitude when an expert is selected) keeps strong but rare experts, covering only 32% of the
    routing traffic at E64.
- **Next options (the user's call).** A healing fine-tune (distillation) of E72 toward the full model; top-8 routing
  for the pruned models (whitelist prepared, untested); or a different model for Kaggle.

## Validation hygiene (user rule, 2026-10-23)

The 120 ARC-AGI-2 eval tasks are never used in the pruning process; they only measure the result.

- **Tool round:** all statistics (Engram census, REAP, keep sets) come from training rollouts only. The choice among
  a few candidates (64 GiB Engram budget, the contribution/decoder rule, w=30) was informed by screens on eval-task
  traces (NLL, probes). That is mild selection leakage. It cannot explain a 29% vs 66% gap, but it is not clean.
- **No-tool round and later:** screening uses a family-disjoint dev split of the training rollouts
  (`scripts/split_dev.py`: sha1(NVARC parent or ARC task id) % 10 == 0, about 10%); the eval set is touched only for
  final metrics.

## Low-bit round: more experts at 2-3 bits (2026-10-24, in progress)

**Goal (user):** keep more experts at lower precision within the 4xL4 budget; attention, router, shared expert and
Engram stay as released. Shapes agreed with E042 (dsv4-infer), both fit 4xL4 with Engram 30%:
**E128** = gate/up uint2 g128 + down uint2 g64; **E96** = gate/up uint2 g128 + down uint3 g64 (asymmetric, bf16
scales, Humming kernels; E96 + DSpark was dropped by E042: the drafter halves the streams and loses 21-28% throughput).

**Pipeline.**
1. `patches/apply_e043_humming.py` + `e043_humming.py`: real-weight Humming experts in the vLLM fork. Each MXFP4
   expert is decoded to exact BF16 at load (checked bit-exact against the fork's DeepGEMM decoder), then Humming
   quantizes it (RTN) or takes precomputed tensors (`E043_GPTQ_DIR`). Fixes for the fork vs humming 0.1.12 (E042's
   dummy-weight runs never loaded real weights): native "humming" schema with weight-only keys
   (`weight_dtype`, `weight_scale_dtype`), no `GROUP_TENSOR`, `weight_scale_2_type`, zero-point loading, per-sublayer
   schema (`DSV41_EXPERT_HUMMING_W2`), router shapes up to (160, 6), GPTQ-only checkpoints (placeholders).
2. Calibration: `replay_reap.py --dump-x` records every layer's MoE input; 1.4M no-tool calibration tokens (61
   rollouts) replayed **under each pruned model's own mask** (E128, E96; plus the full model), 10-20% of positions kept
   (`calx_transpose.py`).
3. `gptq_experts.py`: per-expert GPTQ (gate/up share X^T X of the routed tokens; down uses the quantized gate/up
   outputs with the native clamps, weighted by the routing weight). Experts see ~3.4K rows (median) for 5,120 inputs,
   so low damping overfits: a held-out sweep (20% of rows) picked percdamp 1.0 (layer 30 down: 0.01 is worse than RTN
   held-out, 15.0% vs 13.2%; 1.0 gives 8.8%). Every layer's output decodes bit-exactly through Humming.
4. `make_lowbit_export.py`: GPTQ-only checkpoints (no MXFP4 experts; E042's request): E128 123.1 GB, E96 116.5 GB
   (Engram 30% compact shared).

**Dev screen** (teacher-forced NLL on 64 no-tool dev traces vs the full model 0.5663; answer-writing probe):

| model (keep set x10, Engram 30%) | dev NLL vs full | answer written correctly |
|---|---:|---:|
| E128 native MXFP4 | +8.3% | |
| **E128 uint2 GPTQ** | **+12.1%** | 97.4% |
| E128 uint2 RTN | +22.0% | |
| E96 native MXFP4 | +12.7% | |
| **E96 uint2 + uint3 down, GPTQ** | **+15.2%** | 97.4% |
| E72 native MXFP4 (ARC pass@1 57.5%, pass@2 66.9%) | +22.2% | 99.5% |
| Engram 30% only | +0.10% | |

**ARC** (no-tool, 120 tasks, 2 attempts; full comparison table with tokens and throughput:
`results/lowbit_summary.md`, `scripts/lowbit_summary.py`):

| model | pass@1 | pass@2 task-weighted | out tokens mean | 4xL4 (E042, dummy weights) |
|---|---:|---:|---:|---|
| full (native) | 64.7% | 76.8% | 31.4K | does not fit |
| **E128 2-bit GPTQ + Engram 30%** | **59.6%** | **72.0%** | 41.6K | 16.87 GiB/GPU; util .93: 419 tok/s @48 |
| E72 x10 native | 57.5% | 66.9% | 31.0K | util .93: 311 tok/s @36 |
| **E96 2/3-bit GPTQ + Engram 30%** (final) | 58.7% | **75.3%** (125/167 pairs, = full) | 37.4K | 15.18 GiB/GPU; util .93: 585.9 tok/s @88, 50.6 attempts/h |

(3 of 334 E128 attempts timed out while queued: the client timeout of 3600 s also counted queue time; raised to 6 h
for `--base-url` runs.) E128 2-bit reasons longer (median per-attempt ratio 1.29x vs the full model on the same
attempts) without looping (gzip ratio of the reasoning 0.321 vs 0.326, last 20K chars 0.345 vs 0.342, no loop-like
tails) but with more hedging ("maybe" 1.47x, "hmm" 1.54x per 10K chars; `scripts/reasoning_loops.py`). Attempts per
hour on 4xL4 are then about equal to E72 (~36/h each); the gain is +5 pp pass@2.

**FP4 Engram** (`make_fp4_engram.py`, MXFP4 rows emulated exactly in the MXFP8 container, 1.39% relative MSE vs the
released table): E128 2-bit dev NLL 0.63507 vs 0.63499 with FP8 Engram, answer writing 188/192 vs 187/192. Free; halves
the host-RAM table (56.6 -> ~30 GiB) given a 4-bit Engram lookup kernel.

**Deliverable:** `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E128-2bit` (private, 123.1 GB, GPTQ-only layout,
Engram 30% FP8): parity dev NLL 0.63486 (export) vs 0.63499 (runtime masks). `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E96-2bit3bit` (private, 116.5 GB, w13 uint2 g128 + w2 uint3 g64,
same layout): parity dev NLL 0.65215 (export) vs 0.65213 (runtime masks); ARC pending. Upload rule (user, 2026-10-24):
every model with a known 4xL4 inference path and modest loss vs the full model goes to the user's HF account; quality
probes that don't fit 4xL4 (E192, full model at 2 bits) go to `chankhavu/dsv41-flash1017-e043-gptq-artifacts` as
GPTQ tensors only.

**Expert coverage** (no-tool routing, answer x10 rule): kept share of routing selections E72 74%, E96 81%, E128 87.5%,
E160 92%, E192 95%, E256 98%; a decode batch of 48 tokens touches ~110 distinct experts per layer (max layer 140).
E192 2-bit GPTQ (quality probe; does not fit 4xL4 at 2 bits) and unsloth's experts at E128: evaluating.

**unsloth UD-IQ2_XXS comparison.** Our engine cannot read GGUF IQ codebooks, so `gguf_to_humming.py` decodes each
expert exactly (gguf-py) and re-stores it as uint6 g32 (+0.04% of the weight norm, ~0.5% of IQ2's own error; uint4
would add 12%). Their "UD-IQ2_XXS" experts are a per-layer mix (projection x layer counts): IQ2_S 25, IQ2_XXS 18,
IQ3_XXS 15, IQ1_S 10, IQ2_XS 6, Q2_K 1; IQ2_S gate weight error vs the released FP4 is 7.2% (cosine 0.967) vs ~21-25%
for integer 2-bit RTN. First comparison: their experts on the same E128 keep set (attention/router/Engram as released).
Their Q5_K Engram cannot be represented in our MXFP8 Engram format without snapping back to the originals.

## Tool-use E128 2-bit (2026-10-24, in progress)

User request: E128 with tool use. Everything comes from tool data only:
- **Keep sets:** the tool round's statistics (10M tokens of training tool rollouts, `stats_all.npz` + `toolblock_all.npz`).
- **Engram:** the tool round's 64 GiB table.
- **GPTQ calibration:** its own dump of tool rollouts under the E128 mask (1.4M tokens, 10% of rows).
- **Screens:** a family-disjoint dev split of the 10,001 tool rollouts (`split_dev.py`: 833 dev, 9,168 calibration) and
  first-turn probes on 8 dev prompts (`make_dev_probe_parquet.py`). The 120 eval tasks only score the final model.
  (The tool statistics predate the split and include dev-family rollouts; this biases all candidates equally.)

| native weights, tool dev (64 traces) | dev NLL vs full | tool-block NLL | tool-block top-1 | valid first-turn calls | length cut |
|---|---:|---:|---:|---:|---:|
| full | 0 (0.7130) | 0.118 | 95.7% | - | - |
| full + Engram 64 GiB | +0.04% | 0.118 | 95.7% | 25/25 | 0 |
| E128, tool-block weight 0 | +5.9% | 0.156 | 94.6% | 20/25 | 5 |
| **E128, tool-block weight 10 (chosen)** | +7.9% | 0.135 | 95.2% | 22/25 | 2 |
| E128, tool-block weight 30 | +9.6% | 0.131 | 95.3% | 23/25 | 1 |

All three keep sets keep ~83% of tool-mode routing and overlap the no-tool E128 set by 71-81%. Without the weight the
first-turn format already breaks (the E72 failure mode). Weight 10 recovers most of the tool-block quality at less
general-NLL cost than 30. 2-bit build and the 2-attempt tool-harness eval (`--rescue-code`): running.
GPTQ on tool calibration (layer 0): w13 relative output error 2.6% (RTN 11.9%), w2 5.5% (RTN 13.2%).

**Harness (user, 2026-10-24): sub80s40 + minimal header** (`eval_arc.sh` `PRESET=sub80s40`: 80k total output, 40k
soft budget, per-turn 64k), the best-accuracy API config on the full model (E030, 120 tasks: pass@1 74.5%, pass@2 84.9%
task-weighted, 27.4k mean tokens; s20x60 + header: 69.9% / 80.3% at 23.3k). The user accepts that the keep sets and
calibration came from s20x60 rollouts (experts assumed stable across budgets). Same for the tool-use E96 2/3-bit run
(server B, block weight 10 or 30 picked on the dev first-turn probe).

**4xL4 no-tool attempts per hour (E042, dummy weights, util 0.93, each model's own ARC output shapes, 0
preemptions):**

| model | output mean | streams | decode tok/s | attempts/h | 334-attempt eval |
|---|---:|---:|---:|---:|---:|
| E72 | 31.0K | 52 | 399.5 | 42.4 | 7.9 h |
| E128 2-bit | 41.6K | 48 | 419.2 | 33.8 | 9.9 h |
| E96 2/3-bit | 37.4K | 88 | 585.9 | 50.6 | 6.6 h |

- E96's KV peaked at 0.83, past E042's 80% guard, so E042 would run it at about 80 streams.
- E96 is 1.19x E72 and 1.50x E128 on attempts/h, at no-tool pass@2 75.3% vs 66.9% / 72.0%.
- E128's +5 pp pass@2 over E72 costs about 20% of attempts.
A tool-use E128 needs its own multi-turn replay (prefix-cache reuse) before its L4 throughput is known.

**4xL4 tool-mode throughput (E042 T2, 2026-10-24).** Setup: dummy weights in these schemas, util 0.93, prefix caching,
64 GiB tool Engram, replaying the full model's tool turn shapes from `tools_e030_hdrmin`.

| model x streams | gen tok/s | prefix hits | KV tokens | KV peak / p50 | preemptions |
|---|---:|---:|---:|---:|---:|
| E72 x 32 (util 0.88) | 205 | 87.8% | | | |
| E128 2-bit x 48 | 340.5 | 96.1% | 1.47M | 0.77 / 0.68 | 0 / 58 min |
| E96 2/3-bit x 80 | 453.5 | 96.5% | 2.72M | 0.64 / 0.60 | 0 / 70 min |

- E128: TTFT median 0.84 s, turn median 117 s, turn prompt p50/p90 24.9K/56.3K.
- E96 measured from vLLM metric deltas over a 17-min window (the same method on E128 gives 342.0 vs the replay's 340.5).
- E96 x 96 is queued (T3).
- The pruned models' own turn shapes differ (more, shorter turns; see the interim results below). They go to E042 for
  a replay once the evals finish.

**Final tool-use E128 2-bit (2026-10-24 13:05 UTC; 120 tasks / 167 pairs x 2 attempts, sub80s40 + header,
temperature 1.0, `--rescue-code`; reference = API full model E030 on the same attempts; `scripts/tool_interim.py`):**

| model | pass@1 | pass@2 | pass@2 pairs | correct attempts | mean / median / p90 out tokens | turns | stopped by budget | Python-error calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| API full | 74.5% | 84.9% | 142/167 | 247/334 | 27.4K / 22.8K / 59.8K | 13.9 | 2 | 2.3% |
| **tool E128 2-bit** | **72.8%** | **82.5%** | 136/167 | 240/334 | 35.2K / 29.8K / 72.6K | 17.1 | 27 | 14.2% |
| tool E96 2/3-bit | 64.9% | 75.8% | 126/167 | 213/334 | 30.5K / 26.6K / 59.8K | 23.6 | 2 (+58 at 40 turns) | 37.4% |

pass@1 and pass@2 are task-weighted.

- Within the noise band (2-3 pp) of the full model, at 2 bits and a third of the experts.
- Pairs: 128 solved by both, 14 by the full model only, 8 by E128 only.
- Attempts: 212 solved by both, 35 full only, 28 E128 only.
- The cost is length: +28% output tokens, and 27 attempts hit the 80k budget (10 of them correct for the full
  model).
- 0 rescues: the tool-call format holds with the tool-call-block weight.

**Deliverable:** `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` (private; uploaded 13:51 UTC; all 122 files
verified against the export by name and size).
- GPTQ-only layout, `export/tool_e128_w10_eng64_gptq`, 131,090,957,662 bytes.
- Built by `export_pruned.py --no-routed-experts --engram-from export/e72_w30_eng64`, then `make_lowbit_export.py`.
- The 64 GiB tool Engram table is shared with the tool-round exports.
- Parity on 64 tool dev traces: export 0.79640 vs runtime masks 0.79652 (dev NLL); tool-block NLL 0.1490 vs 0.1494.
- First-turn probe: 22/25 valid calls vs 24/25 (sampling noise at 25).

**Engram size on the 2-bit model.** Question: does a weaker model lean on Engram more? Measured on the tool E128 2-bit
server, dev NLL on 64 tool dev traces:

| Engram | dev NLL |
|---|---:|
| 64 GiB | 0.79679 |
| 112 GiB | 0.79647 |
| full table | 0.79650 |

The spread is 0.04%, the same noise floor as on the full model, so the 64 GiB tool table stays.

**Tool E96 2/3-bit is turn-limited.** It finished at pass@2 75.8%, 9 pp below full.
- Pairs: 117 solved by both, 25 by the full model only, 9 by E96 only.
- 58 attempts stopped at the harness's 40-turn cap, after only 40.4K of their 80K output tokens on average. The full
  model solved 38 of those attempts, and E96's session-answer fallback saved 7.
- Rerun on server B with `--max-turns 64`, everything else unchanged (`e96t_..._sub80_mt64`); interrupted, see below.
- E96's attempts are shorter than E128's (30.5K vs 35.2K tokens). At E042's tool-mode rates (453.5 vs 340.5 tok/s on
  full-model shapes) that is about 53 vs 35 attempts/h on 4xL4.

**The tool errors are typos, from pruning.** DSML format errors are negligible (4 of 4,504 calls for E128). The errors
are Python exceptions in the model's own code, mostly character-level typos:
- `NameError: coms` (for `comps`), `dequeue`, `zzip`, `mmasks`;
- `ModuleNotFoundError: scpy`, `sc`, `scip` (for `scipy`);
- `SyntaxError` from mismatched brackets.

| run | Python-error rate |
|---|---:|
| full, self-hosted, native | 2.4% |
| full + Engram 64 GiB | 2.7% |
| native E72 MXFP4 (no GPTQ) | 44.6% (same typos) |

So expert pruning causes them, and the 2-bit GPTQ does not; more experts means fewer typos. The model recovers from
the error message, but each typo costs a turn (E96: 35 of 250 attempts end at max_turns).

**Truncated sampling does not fix the typos.** Test: tool E128 2-bit on 32 dev-family ARC training tasks (sha1 family
rule), 1 attempt, `eval_arc.sh SAMPLING=...`.

| sampler | solved (attempts done of 36) | tool calls | Python-error calls | typos seen |
|---|---:|---:|---:|---|
| temperature 1.0 | 32/33 | 170 | 10.0% | `dequeue`, `froset`, `froroset`, `dint` |
| + min_p 0.05 | 32/33 | 161 | 9.9% | `coms` x7, `dequeue`, `froroset`, `fset` |
| top_p 0.95 | 31/32 | 166 | 7.2% | `zys`, `ne`, `coms` |

The last 3-4 long attempts of each arm were stopped to free server A.

Typos survive min_p 0.05, so their probability is at least 5% of the top token's. The pruned model genuinely puts mass
on them; they are not the sampler's tail. The fix has to come from the model: the keep set (for example, weighting
identifier-copy tokens), or healing. The dev tasks are easy (about 4.5 turns), so this test cannot rank the samplers
on accuracy.

**Interrupted runs (2026-10-24 14:46 UTC: the Vast account's credit ran out and Vast stopped both boxes).** Two runs
lost their in-flight attempts:
- the self-hosted full-model reference (`full_ref_sub80`, server A, started 13:29);
- the E96 max-turns-64 rerun (server B, started 13:56).

The attempts that finished are the shorter ones, so only same-attempt comparisons mean anything:

| run (finished attempts) | correct | API full | tool E128 2-bit | tool E96, 40 turns | out tokens (vs API) | turns (vs API) |
|---|---:|---:|---:|---:|---:|---:|
| self-hosted full (171/334) | 134 | 144 | 146 | 118 | 20.7K (21.8K) | 12.5 (12.4) |
| tool E96 2/3-bit, 64 turns (110/334) | 93 | 96 | 101 | 86 | 14.9K (16.0K) | 16.7 (10.2) |

- **Self-hosted full vs the API full model**, same 171 attempts: 9 solved only self-hosted, 19 only by the API (sign
  test p = 0.09).
  - Against tool E128 2-bit: 9 vs 20 (p = 0.06).
  - Harness and config are the same (persistent-v6, sub80s40 + header); the self-hosted run adds `--train-check
    --turns-note 5`. Python-error rate 2.6% (API 2.3%).
  - A possible cause is the sampler: self-hosted runs use temperature 1.0 / top_p 1.0, and the API's thinking-mode
    sampler is not known to match.
  - Not conclusive at 171 attempts. If it holds, the self-hosted full model is the fairer baseline for the pruned
    models (same stack and sampler), and tool E128 2-bit is level with it.
- **E96 at 64 turns vs 40 turns**, same 110 attempts: 93 vs 86 (15 vs 8, p = 0.21). The longest attempt used 60 turns.
  Python-error rate 34.5%.
- **What was lost:** the box's `/dev/shm`, which a stop clears. It held the checkpoint (re-download ~25 min), the tool
  E96 GPTQ tensors and calibration dumps. Only their per-layer stats are in `results/box`; the GPTQ itself took ~5 min
  on 4 GPUs.
  - Every uploaded checkpoint is safe on HF.
  - `/workspace` survives a stop. It holds the exports and the server logs after 13:39 UTC, which were not pulled yet.
  - All traces and results are local (`evals/results/`), and the cache-health series is in `results/cache_watch.tsv`.
- **Resume:** run `eval_arc.sh` again with the same TAG. The harness re-runs attempts recorded as errors
  (`run_deepseek_tools.py`: jobs = missing or `"error" in latest`).

## No-tool round (2026-10-23/24)

**Question (user):** without tools the model does not need its coding skills, so maybe it routes over fewer experts
and prunes better. Everything in this round is no-tool only: the statistics come only from no-tool training rollouts
(no mixing with the tool round). Targets: E72 and E64 (inference constraints).

**Answer: yes, much better at E72.** E72 keeps 89% of the full model's no-tool pass@1 and 87% of its pass@2
(tool round: 44% of pass@1).

| model (no-tool harness, 120 tasks / 167 pairs, self-hosted) | pass@1 pairs | pass@1 task-weighted | pass@2 task-weighted (pairs) | tool-round counterpart, pass@1 |
|---|---:|---:|---:|---:|
| full (384 experts, full Engram) | 108/167 = 64.7% | 67.9% | 76.8% (125/167) | tool harness 65.9% |
| **E72, answer block x10, Engram 48 GiB** | 96/167 = 57.5% | 57.4% | **66.9% (110/167)** | E72 w30: 29.3% |
| E64, answer block x10, Engram 48 GiB | 65/167 = 38.9% | 39.4% | - | E64 w30: 18.0% |
| E64, answer block x20, Engram 48 GiB | 54/167 = 32.3% | 32.5% | - | |

- Full-model pass@2 for the same no-tool config on the API: 75.4% (DeepSeek API, E035 `hdrsum`), 74.6% (DeepInfra
  fp8, E044 eval120); self-hosted 76.8%.
- pass@2 pairs: 101 solved by both, 24 by the full model only, 9 by E72 only.
- E72 vs full, same pairs (attempt 1): 82 solved by both, 26 full only, 14 E72 only, 45 neither. E72 attempt 2 alone:
  84/167.
- E64 x10 beats x20 (the dev rule picked x20 for perfect answer writing, but its +7 points of NLL cost more than the
  ~4% of answers it saves); E64 still loses most of the hard pairs (15 runs to the context limit, 33 unparsed answers).
- **E48 dropped (user, 2026-10-24):** pruning is clearly not enough at that size; it would need expert merging.

### Deployable checkpoint (E72 x10 + Engram 48 GiB)

`export_pruned.py --expert-keep reap/keep_nt2/keep_contrib_dec_sp0_w10_e72.npy --engram-keep
engram/masks_nt2_calib/engram_top48gib.npz --text-only` -> `export/notool_e72_x10_eng48` (118.5 GB in 13 shards:
backbone + compact Engram, 107.8M / 87.4M rows kept on layers 1 / 14). Served on the fork with
`apply_compact_engram.py` (TP4+EP4: 16.64 GiB weights per GPU); teacher-forced NLL on the 64 dev traces 0.69195 vs
0.6918 with runtime masks on the full checkpoint (top-1 77.65% vs 77.66%), i.e. parity. Private HF model:
`chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E72` (Engram files included; DSpark and vision tensors removed).

### Data and splits

- **Data:** the `notool_e044_*` configs of `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (E044 = E035
  arm D: effort low + minimal header + task summary, official ARC Prize prompt, one turn), snapshot of 2026-10-23
  21:40 UTC: `arc2_train` 1,070 + `arc2_train_sample2` 1,076 + `nvarc_b0` 1,430 + `nvarc_b1` 1,803 = 5,379 rollouts,
  118.6M tokens (`tokenize_rollouts.py --no-tools`). Prompt tokens match API usage on 5,379/5,379; completion counts
  differ by 0-3 tokens (the usage convention for EOS and `</think>`).
- **Dev split** (`split_dev.py`, family-disjoint, no eval data): calibration 4,931 rollouts / 109.2M tokens, dev
  448 / 9.4M. All screens (NLL, probes) run on dev; the 120 eval tasks are used only for the final ARC scores.
- **Baseline:** `scripts/eval_notool.sh` (E044 config against the box's server), full model 108/167 = 64.7%, mean 31.3k
  output tokens, 1 cut at the context limit (`e043nt_full_a1`).

### Engram (census on the 109M calibration tokens; dev NLL, 64 traces)

| mask | share of rows | NLL gen | vs full |
|---|---:|---:|---:|
| full table | 100% | 0.5663 | - |
| all observed, 73 GiB | 39% | 0.5666 | +0.05% |
| 64 GiB | 34% | 0.5667 | +0.07% |
| 48 GiB (used below) | 25% | 0.5668 | +0.10% |

The answer-writing probe below gives 192/192 with Engram 48 GiB alone.

Lookup hit rates (`results/notool_engram_stats_{dev,eval}.json`, `results/notool_engram_sufficiency.json`; eval = full-model
no-tool traces on the 120 eval tasks, metric only):

| mask | dev hit | eval-trace hit | same budget from half the calibration data: dev / eval |
|---|---:|---:|---:|
| 16 GiB | 91.1% | 88.0% | 90.5% / 87.3% |
| 32 GiB | 93.7% | 91.3% | 93.0% / 90.4% |
| **48 GiB** | **95.3%** | **93.4%** | 94.8% / 92.8% |
| 64 GiB | 96.6% | 95.1% | |
| all observed (73.2 GiB; half data: 51.8 GiB) | 97.5% | 96.3% | 95.3% / 93.5% |
| tool-round 64 GiB mask | 95.0% | 93.7% | |

- Prompt lookups are ~99.9% covered at every budget; the misses are in generated text (eval traces at 48 GiB:
  2-gram 98.9%, 3-gram 93.7%, 4-gram 87.6% hit).
- Doubling the calibration data (55M -> 109M tokens) adds only 0.5-0.6 points at 48 GiB: the long tail grows (all
  observed rows 51.8 -> 73.2 GiB) but carries little traffic.
- Eval traces vs the calibration census: 23.9% of the distinct rows they touch were never seen in calibration, carrying
  3.7% of lookups; 99.8% of those come from generated text (`results/notool_unseen_*`).

### Routing: fewer effective experts without tools (REAP replay, 20M calibration tokens, `compare_routing.py`)

| | tool round (10M tokens) | no-tool (20M tokens) |
|---|---:|---:|
| effective experts per layer, mean (exp of routing entropy) | 121.7 | 108.4 |
| encoder layers 0-19 / decoder layers 20-39 | 127.8 / 115.5 | 111.2 / 105.6 |
| routing share kept by the top-48 / 64 / 72 (contribution ranking) | 59.3 / 66.3 / 69.0% | 64.5 / 70.9 / 73.7% |

No-tool routing is more concentrated in 39 of 40 layers. The two workloads' top-64 sets share 84% of their experts;
the tool-round top-64 set keeps only 66.9% of no-tool routing (the no-tool set: 70.9%).

### Dev screen (teacher-forced NLL on 64 dev traces + answer-writing probe on 48 teacher-correct dev traces)

`answer_copy_probe.py` feeds the teacher's prompt and reasoning up to `</think>` and lets the masked model write the
answer itself (1 greedy + 4 samples at T=1), scored with the harness parser. Answer-block weighting ("answer xW") adds
(W-1) x the contribution on the generated answer block (`</think>` to EOS) to the contribution score
(`special_stats.py select --weight W`, no-tool answer-block statistics).

| keep set (+ Engram 48 GiB) | NLL gen vs full | answer written correctly (T=1) | unparsed |
|---|---:|---:|---:|
| 384 experts | +0.2% | 192/192 | 0 |
| E96 | +11.1% | 183/192 | 8 |
| E80 | +16.1% | | |
| E72 | +20.2% | 157/192 | 32 |
| **E72 answer x10** | +22.2% | 191/192 | 0 |
| E72 answer x30 | +32.0% | 192/192 | 0 |
| tool-round E72 w30 set | +41.6% | 180/192 | 0 |
| E64 | +25.3% | 155/192 | 34 |
| E64 answer x10 | +27.0% | 184/192 | 6 |
| **E64 answer x20** | +34.1% | 192/192 | 0 |
| E64 answer x30 | +39.2% | 192/192 | 0 |
| tool-round E64 w30 set | +47.9% | 178/192 | 0 |
| E56 | +31.4% | | |
| E48 | +39.0% | | |
| E48 answer x30 | +63.1% | 179/192 | 11 |

- Plain pruning loses ~1 in 5 answers *when handed correct reasoning* (mostly unparsable output), so the answer
  block needs protection; a small weight fixes it.
- Rule used for the evals: the smallest answer weight that writes >= 99% of the answers on dev (E72: x10, E64: x20).
  E64 x20 then costs 7 points of NLL over x10, so E64 x10 is evaluated too.
- The no-tool sets beat the tool-round sets by ~9 NLL points at the same size and write answers better.
- Free-generation peek (3 dev prompts, `freegen_peek.py` style): E72 x10 reasons like the full model but often starts
  its reasoning with a stray "." or ":"; on 2 of 164 eval pairs that start derails into a near-empty or garbled trace.

## Side measurement: DSpark drafter acceptance (for E042 / dsv4-infer, 2026-10-24)

The 1017 checkpoint ships a DSpark speculative drafter: `mtp.0-2`, 2,401 tensors, 7.38 GiB (routed experts 6.72 GiB
MXFP4 + scales, dense 0.66 GiB), 128 experts top-3, block size 5, conditioned on target layers 37-39.
`scripts/dspark_probe.sh`: server B restarted with `--speculative-config {"method":"dspark","num_speculative_tokens":5}`
(standard rejection sampling, T=1), four 15-minute windows of real eval traffic (24 tasks, 24 workers), /metrics
counter deltas. The drafter drafts the block in one pass, so the per-position counters give the acceptance length for
any N <= 5. Runtime masks leave the drafter's own experts unmasked. Results: `results/dspark_probe.jsonl`.

| config | mean acceptance length N=1 / 2 / 4 / 5 | per-position acceptance 1..5 |
|---|---|---|
| full, tool harness | 1.66 / 2.11 / 2.64 / 2.81 | 0.66 0.45 0.31 0.22 0.17 |
| full, no-tool harness | 1.69 / 2.19 / 2.83 / 3.05 | 0.69 0.49 0.36 0.28 0.22 |
| E72 x10 (no-tool set) + Engram 48 GiB, no-tool | 1.59 / 1.99 / 2.49 / 2.66 | 0.59 0.39 0.28 0.21 0.17 |
| E72 w30 (tool set) + Engram 64 GiB, tool + rescue | 1.39 / 1.62 / 1.88 / 1.97 | 0.39 0.22 0.15 0.11 0.09 |

## Setup

- **Box:** Vast 57370044, 8x RTX PRO 6000 Blackwell Server (96 GB, SM120), 1.5 TB RAM, 256 vCPU, 1 TB disk
  (RAID1, only ~120 MB/s sustained writes), driver 580.105.08, $13.73/h.
- **Checkpoint:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, 510.31 GB, downloaded into `/dev/shm`
  (RAM; the disk was too slow) at ~400 MB/s. Model load from shm takes ~42 s.
- **Runtime:** vLLM fork `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` + flashinfer vision2 (E042's
  `requirements.box.freeze`), venv `/workspace/dsv4/venv`. MXFP4 experts run on Marlin with native MXFP8
  activations and the router weight applied before the fc2 quantization (`router_weight_before_fc2_quant`).
- **E043 hooks** (`patches/apply_e043_patches.py`, `patches/e043_obs.py`), controlled by a JSON file `$E043_CTRL`
  re-read at each eager forward:
  - expert masks: pruned experts get selection bias -1e9 in the native top-6 kernel, so top-6 refills from kept
    ids and weights renormalize over the selected set, exactly as in a physically pruned model;
  - Engram row masks: dropped row ids are rewritten to -1, which the fork's owner-sharded lookup reads as a zero
    row before Wkv (the plan's declared miss rule);
  - REAP observer on `MarlinExperts.moe_sum`: per token and slot, ids, router weight and the L2 norm of the native
    weighted expert output (`reap_native_amp`), all-reduced over the EP group (needs `--enable-expert-parallel`).
- **Serving:** `scripts/serve.sh` (TP4 or TP8 + EP, fp8 KV, prefix caching, FULL_DECODE_ONLY graphs, Engram CPU
  offload, `--reasoning-parser/--tool-call-parser deepseek_v41`). Two TP4+EP4 servers: A (GPUs 0-3, port 18000)
  for ARC evals, B (GPUs 4-7, port 18001) for replay/NLL with its own control file.

## Data and checks

- `scripts/tokenize_rollouts.py`: rebuilds each rollout with the model's `encoding/encoding.py` (thinking mode,
  tools on the system message so earlier reasoning is kept, numeric effort 50), cut after the last assistant
  message; roles mark model-generated tokens.
  - 10,001 rollouts, 251.8M consumed tokens (106.5M prompt/tool, 145.3M generated).
  - Per-turn check against the API usage (`verify_tokenization.py`): prompt tokens equal the rebuilt prefix on
    **all 78,992 turns**; completion tokens equal the span on 87% of turns and differ by +1 on 13% (the sampled
    stream had one more token than the re-tokenized text, e.g. trimmed whitespace; the next turn's prompt still
    matches), plus a few length-truncated turns.
  - The server's chat template (`/tokenize`, tools + effort 50) gives identical ids on 12/12 sampled rollouts.
- Held-out set: 334 eval-task traces from E030 `s20x60_hdrmin` (same harness, 120 eval tasks), 12.4M tokens.
- Engram hash parity: the fork's GPU hash ids equal the CPU census hashes (`engram_census.py`, a numpy port of the
  official `inference/engram.py`) on 2,188 x 24 ids in both layers.

## Engram row census (report: `reports/engram_census.html`)

- 6.04B lookups per Engram layer touch 223.6M distinct rows (58.2% of 384M); all observed rows = 110 GiB of the
  188.8 GiB table. 22% of touched rows are hit once, 61% at most 4 times; rows hit > 64K times carry 39% of
  lookups. 4-gram banks are 95% touched (saturated by collisions), 3-gram 67%, 2-gram 13%.
- Global top-frequency masks (`engram_census.py select`), held-out eval-trace hit rates:

| kept GiB | calib coverage | eval hit | eval generated | eval 2/3/4-gram | tokens with all 48 hit |
|---:|---:|---:|---:|---|---:|
| 8 | 85.6% | 81.2% | 71.7% | 95.9 / 81.6 / 66.0% | 62.3% |
| 16 | 89.3% | 84.8% | 77.2% | 96.9 / 85.2 / 72.3% | 65.4% |
| 32 | 93.3% | 88.9% | 83.3% | 97.6 / 88.6 / 80.5% | 68.3% |
| 64 | 97.6% | 94.0% | 91.1% | 98.3 / 92.6 / 91.2% | 72.2% |
| 96 | 99.5% | 97.3% | 95.9% | 98.9 / 96.0 / 97.0% | 77.3% |
| 110 (all observed) | 100% | 98.4% | 97.7% | 99.1 / 97.3 / 98.9% | 84.3% |

Kaggle 4xL4 host RAM leaves ~122 GiB for the pinned table plus a 2.9 GiB int32 map (E042), so all observed rows fit.

**Rows the eval uses but training never touched** (`engram_unseen.py`, all 2,004 Flash-1017 eval-task traces of the
6 lean2sub runs, 82.6M tokens): 11.6% of the 164.6M distinct rows the eval touches per layer are absent from the
census, but they carry only 1.7% of eval lookups (two-thirds of them are hit once). By token source
(`engram_unseen_kinds.py`, 21M tokens): task prompt (system + grids) ~0% unseen; tool outputs 0.85% (7% of all unseen
lookups: new digit runs from printed grids, ASCII renderings); model-generated text 2.4% (3-gram heads 4.1%; 93% of
unseen lookups: novel phrasing and code identifiers such as `ates with vertical`, `_obj(m):`, ` grid.tol`). The
census covers the task format completely; the gap is the open-ended tail of the model's own reasoning text.
Hit rates of the masks on the 82.6M-token set: 64 GiB 93.6%, all observed 98.3% (same as on the 12.4M-token set).

### Data sufficiency (held-out eval hit rate, census on half vs all of the calibration rollouts)

| budget | half data | all data |
|---|---:|---:|
| 16 GiB | 84.4% | 84.8% |
| 32 GiB | 88.4% | 88.9% |
| 64 GiB | 93.3% | 94.0% |
| all observed rows | 96.4% (89 GiB) | 98.4% (110 GiB) |

The frequency ranking has converged for a fixed budget (+0.5 pp from doubling the data); more data mostly adds
rare rows, which only matters near the ~122 GiB cap.

### Engram masks: teacher-forced NLL on held-out eval traces

`scripts/nll_candidates.sh` on server B (runtime masks), 32 E030 `s20x60_hdrmin` eval-task traces (<= 40k tokens),
514,113 model-generated target tokens, NLL in nats/token. Two runs of the full model differ by 0.0004
(0.6671 vs 0.6667: request batching changes the reduction order), so that is the noise floor.

| Engram | of table | NLL gen | vs full (mean of 2 runs) | top-1 agreement |
|---|---:|---:|---:|---:|
| full (2 runs) | 100% | 0.66705 / 0.66666 | - | 77.95% / 77.96% |
| all observed, 110 GiB | 58% | 0.66724 | +0.04% | 77.91% |
| 96 GiB | 51% | 0.66740 | +0.06% | 77.90% |
| 64 GiB | 34% | 0.66760 | +0.09% | 77.94% |
| 32 GiB | 17% | 0.66887 | +0.28% | 77.87% |
| 16 GiB | 8.5% | 0.67194 | +0.74% | 77.83% |

**Frozen Engram mask for REAP: `engram_top64gib`** (within noise of the full table, ~46 GiB less Kaggle RAM than
keeping every observed row). 110 GiB is the fallback.

## REAP statistics

- Replay (`scripts/replay_reap.py`): calibration rollouts sent one at a time as token prompts (`max_tokens=1`,
  unique `cache_salt`) to server B with the observer on and the frozen Engram mask; `reap_aggregate.py` checks that
  the recorded `input_ids` equal the request streams and that selections = 6 x positions in every layer.
- Pilot: 41 rollouts, 1.02M tokens, 169 s (6.0k tok/s). All checks pass; the least covered layer saw 367/384 experts.
- At E64 the scoring rules disagree sharply (pilot): plain REAP (conditional mean amplitude) keeps experts that are
  strong when chosen but rarely chosen, covering only 32% of the original routing selections (14% in the worst
  layer); frequency keeps 72% (55%), contribution total (sum of amplitudes) 70% (53%), router-weight mass 72%.
  Keep-set overlap REAP/frequency is 33%. Candidates are compared by NLL rather than assuming REAP.

- Main run: 377 more rollouts, 9.02M tokens (1,501 s). Pilot + main = 418 rollouts, **10.04M tokens**
  (4.28M prompt/tool inputs, 5.76M generated-token inputs); the sparsest layer saw 375/384 experts. Keep sets are
  stable: E64 sets from the 1M pilot and the 10M total overlap 96% for every rule.

### How many experts are (nearly) inactive on ARC traffic

`results/expert_activity.json`, from 10.04M scored positions (60.2M selections per layer; a uniform expert would get
157k):

| category | expert slots (of 15,360) | per layer |
|---|---:|---|
| never selected | 85 (0.55%) | 0-9, mostly the last layers (L39: 9, L38: 7, L35: 6) |
| < 1% of a uniform share (< 1.6k selections) | 1,523 (9.9%) | up to 66 |
| < 10% of a uniform share (< 16k selections) | 4,663 (30.4%) | 55-160 |

- The least-used 192 experts of a layer carry 2.1-11.2% of its routing (mean 4.9%); the least-used 288 carry 12-32% (mean 18%).
- The top 64 carry 56-80% (mean 72%). The busiest expert gets 10-41x its uniform share.
- Pruning to ~E192 removes little traffic, matching the clean E192 probes. The cut to E64-E72 removes experts that
  carry ~25-30% of the traffic.

### E64 scoring rules (one-shot), teacher-forced NLL on held-out eval traces

32 eval-task traces (a different 32-trace sample than the Engram table above: 419,452 generated target tokens),
all with the frozen 64 GiB Engram mask unless noted.

| model | rule | kept routing share (min layer) | NLL gen | top-1 |
|---|---|---:|---:|---:|
| full (384 experts, full Engram) | - | 100% | 0.6640 | 78.0% |
| 384 experts + Engram 64 GiB | - | 100% | 0.6646 | 78.0% |
| E64 | contribution total, decoder layers on generation targets | 66% | **0.857** | 73.6% |
| E64 | contribution total (sum of native amplitudes) | 70% (54%) | 0.871 | 73.3% |
| E64 | sqrt(freq) x mean amplitude | 64% | 0.873 | 73.3% |
| E64 | plain REAP (conditional mean amplitude) | 32% (14%) | 0.922 | 72.1% |
| E64 | router-weight mass | 72% (55%) | 1.006 | 70.5% |
| E64 | frequency | 72% (56%) | 1.064 | 69.4% |

Expert pruning to E64 costs ~+0.19 nats/token (+29%) with the best one-shot rule, while the Engram mask is free.

### Iterative pruning (E042's suggestion) and the E48 / E64 / E72 candidates

`scripts/iterative_prune.sh`: 384 -> 192 -> 128 -> 96 -> 72 -> 64 -> 48 with the same rule, 1.5M fresh calibration
tokens observed under each stage's mask (+ Engram 64 GiB), the next set chosen inside the current one. It is worse
than one-shot at every size, so the candidates are **one-shot "contribution total, decoder layers on generation
targets"** keep sets (`reap/keep_all/keep_contrib_dec_e{48,64,72}.npy`).

| experts | one-shot NLL | top-1 | iterative NLL | iterative top-1 |
|---:|---:|---:|---:|---:|
| 384 (full) | 0.664 | 78.0% | - | - |
| 72 | 0.822 | 74.3% | 0.879 | 73.1% |
| 64 | 0.857 | 73.6% | 0.903 | 72.6% |
| 48 | 0.962 | 71.5% | 1.037 | 69.9% |

### Physical export and parity

`scripts/export_pruned.py` -> `export/e64_oneshot_contribdec_eng64` (129.6 GB: backbone ~58 GB + compact Engram
~71 GB in `model-engram-*` files shared by E48/E64/E72), loaded on the fork with `patches/apply_compact_engram.py`
(TP4: **15.17 GiB weights per GPU**, KV 17.3M tokens on the RTX box). Teacher-forced NLL of the export on the same 32
traces: 0.85694 vs 0.85723 with runtime masks on the full checkpoint (diff 0.0003 < run-to-run noise 0.0004),
identical top-1 (73.63%).

### Free generation breaks below ~E96 (found by the first ARC evals)

- The first pruned ARC runs (E64 physical export on A, E48 runtime masks on B) solved **0/11 and 0/5**: every attempt
  ended after one turn without a valid tool call. Stopped after ~15 min.
- `scripts/decode_consistency.py`: greedy decode agrees with the teacher-forced argmax except for a few near-ties
  (full model too, fp8 KV numerics), so the decode path is fine.
- **Artifact:** switching runtime masks on a server with prefix caching reuses KV computed under the previous mask
  (shared system/tool-schema prefix) and corrupts generation. `gen_probe.py` therefore uses a per-config
  `cache_salt`. (The stopped E48 eval on B was contaminated this way; the E64 export on A was not.)
- Clean first-turn probes (1 simple tool prompt + 3 ARC prompts greedy + 2 T=1 samples each; valid = parsable
  `run_python`/`submit_answer` call):

| model (+ Engram 64 GiB) | valid tool calls | note |
|---|---:|---|
| full / Engram 64 GiB only | 9/10, 4/4 | |
| E192 / E128 / E96 (contrib_dec) | 9/10, 8/10, 10/10 | |
| E72 / E64 / E48 (contrib_dec) | 0/10, 0/10, 3/10 | reasoning fine, DSML block never closed |
| E72 protect 16 / 32 tool-block experts | 8/10, 6/10 | |
| E72 ranked by tool-block contribution only | 9/10 | |
| E64 / E48 protect 32 | 2/10, 0/10 | |

- Failure mode (E72, greedy, special tokens visible): the first `<｜DSML｜ invoke name="run_python">` is correct, but after
  `</｜DSML｜ invoke>` the model opens another invoke (`run_p_python`) instead of `</｜DSML｜ calls>` and runs to
  max tokens. The structural tokens are rare (tool-call blocks are ~6% of generated positions), so all-token NLL and
  contribution totals underweight them.
- Fix under test: expert statistics on tool-call-block targets (`scripts/special_stats.py`, 616k positions from the
  10M-token records) used to protect experts or to up-weight those positions (w = 10, 30) in the contribution score;
  screened with tool-block NLL (`nll_eval.py` now reports `nll_blk` / `top1_blk`).

### Tool-block screen (teacher-forced, 32 eval traces; `nll_blk`/`top1_blk` = targets inside tool-call blocks)

| model (+ Engram 64 GiB) | NLL gen | top-1 | NLL tool block | top-1 tool block | first-turn probe |
|---|---:|---:|---:|---:|---:|
| full | 0.664 | 78.0% | 0.128 | 95.4% | 9/10 |
| E72 contribution (decoder on gen) | 0.822 | 74.4% | 0.305 | 91.0% | 0/10 |
| E72 protect 16 tool-block experts | 0.824 | 74.4% | 0.253 | 92.4% | 8/10 |
| E72 tool-block weight w=10 | 0.878 | 73.2% | 0.205 | 93.5% | 2/10 |
| **E72 w=30** | 0.946 | 71.9% | 0.184 | 94.0% | **10/10** |
| E72 tool-block only | 1.069 | 69.5% | 0.185 | 93.9% | 9/10 |
| **E64 w=30** | 1.013 | 70.6% | 0.199 | 93.5% | **10/10** |
| E64 tool-block only | 1.127 | 68.5% | 0.193 | 93.6% | - |
| E48 w=30 | 1.153 | 68.5% | 0.235 | 92.7% | 0/10 |
| E48 tool-block only | 1.289 | 65.9% | 0.229 | 92.9% | - |

w = weight of tool-call-block positions in the contribution score (`special_stats.py select --weight`).

### Harness safety net `--rescue-code` (evals/run_deepseek_tools.py, opt-in)

A non-final turn without a valid tool call but with code (garbled DSML `code` parameter, JSON `"code"` field,
```python block, or plain code lines) becomes a `run_python` call; near-miss tool names (`run_ppython`) map to the real
tools; a turn with no call, no code and no answer grid gets a short reminder to use the tools (max 3 per attempt).
Evals on runtime masks send a per-run `cache_salt` (`eval_arc.sh` `SALT`).

### What the failures are (Sonnet trace review of the early pruned evals)

Not format-only. E64 shows semantic damage beyond the DSML syntax: hallucinated module names
(`from scipy import nddfficiency`, `scscpy`), ungrammatical reasoning, and code that would not parse even when
extracted. 57% of its tool calls return an error (full model: 1.7%), and garbled-word rates run ~80/1k words vs
~10/1k. E72 is much milder: once it is calling tools, its analysis and code look like the full model's (terser), with
a 19% tool-error rate. Its main problem was first-turn format failures, which the rescue fixes.

### Do the pruned models still see the puzzle's idea? (Sonnet review of the w=30 + rescue evals, partial)

Labels: A = main rule recognized, B = partial, C = wrong or none, D = garbled/no reasoning; judged against the full
model's trace and the task file; partial runs (E72 41 rows, E64 35 rows); counts are a reviewer's calls.

| run | outcome | read | A | B | C | D |
|---|---|---:|---:|---:|---:|---:|
| E72 w30 | correct | 12 | 11 | 1 | 0 | 0 |
| E72 w30 | incorrect | 14 | 7 | 6 | 1 | 0 |
| E64 w30 | correct | 9 | 9 | 0 | 0 | 0 |
| E64 w30 | incorrect | 18 | 2 | 10 | 3 | 3 |

E72's losses are mostly execution: half of its wrong answers state the right rule and miss by a few cells
(e.g. 4/784 cells on 7ed72f31, 3/676 on cbebaa4b), sometimes after degenerate stretches ("Need Need Need").
E64 loses the idea more often, with garbled reasoning ("Weod=w?? Need."), typo loops and turn-limit stops.

## ARC-AGI-2 eval (120 tasks / 167 pairs, 1 attempt, s20x60 + minimal header, self-hosted)

| model | pairs correct | pass@1 pairs | pass@1 task-weighted | notes |
|---|---:|---:|---:|---|
| API Flash-1017 (E030 s20x60_hdrmin), attempt 0 / 1 | 112 / 115 | 67.1% / 68.9% | 68.6% / 71.3% | reference |
| full 384 experts + full Engram (server A) | 110 | 65.9% | 67.4% | `e043_full_a1` |
| E64 + Engram 64 GiB (physical export) | 0 of 11 | | | stopped: no valid tool calls (see above) |
| E48 + Engram 64 GiB (runtime masks) | 0 of 5 | | | stopped; also prefix-cache contaminated |
| E72 w30 + Engram 64 GiB + `--rescue-code` (runtime masks) | 49 | 29.3% | 28.6% | complete |
| E64 w30 + Engram 64 GiB + `--rescue-code` | 30 | 18.0% | 17.5% | complete (44 output_budget, 17 max_turns) |
| E48 w30 + Engram 64 GiB + `--rescue-code` | 0 of 120 | 0% | 0% | stopped after 120: 119 hit the 60k output budget in one turn, 4 tool calls total |
| 384 experts + Engram 64 GiB (control, all experts) | 116 | 69.5% | 69.7% | complete; Engram pruning is lossless on ARC |

## Artifacts (private HF, account chankhavu)

- `chankhavu/dsv41-flash1017-e043-pruning-artifacts` (dataset): Engram counts/masks/geometry, REAP stats and keep
  sets, token streams (calibration + eval traces), raw observer records (20 GB), NLL results, code snapshot.
- `chankhavu/DeepSeek-V4.1-Flash-1017-arc-engram64`: shared compact Engram files + `engram_index.json`.
- `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E72`: the no-tool round's E72 x10 checkpoint with its own compact
  48 GiB Engram files (self-contained).
- `chankhavu/DeepSeek-V4.1-Flash-1017-arc-E72`, `-arc-E64`, `-arc-E48`: backbone checkpoints with the w30 keep sets
  (no Engram files; download the engram64 repo's `model-engram-*.safetensors` into the same folder).
- The artifacts dataset also has `evals/` (all E043 ARC result jsonl and transcripts), `obs/iter_contrib_dec_e*`
  (iterative-stage observer records), `reap/protect` (tool-block keep sets), `results/` (NLL, probes) and `code/`.

## Data inventory for the tech report (2026-10-24, before the box is released)

Local = this folder (also on GitHub); HF = `chankhavu/dsv41-flash1017-e043-pruning-artifacts` (private dataset).

| what | local (`results/` unless noted) | HF |
|---|---|---|
| Tool-round ARC evals (jsonl + transcripts) | `evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_*`, `eval_stats.json` | `evals/` |
| No-tool ARC evals (jsonl + reasoning traces, 2 attempts for full / E72) | `evals/results/e043nt_*`, `notool_eval_stats.json` | `evals/notool/` |
| DSpark acceptance windows + their eval runs | `dspark_probe.jsonl`, `evals/results/*e043_dspark*` | `evals/dspark/`, `box_runs/tp4B_dspark/` |
| Engram census, tool round (counts, masks, report stats, sufficiency, unseen rows) | `engram_stats.json`, `unseen_eval6.json` | `engram/`, `engram_counts/`, `results/` |
| Engram census, no-tool (counts on 109M tokens, masks 16-128 GiB, half-data census, report stats vs dev and eval traces, coverage logs, unseen rows on eval traces) | `notool_unseen_*`, `notool_engram_stats_*.json` | `notool/engram*`, `notool/engram_stats/` |
| REAP sufficient statistics (N, A, Q, W, MX per layer x role x expert), tool / no-tool / pilots | - | `reap/stats_*.npz`, `notool/reap/stats_nt2.npz`, `notool/pilot/reap/` |
| Tool-block / answer-block statistics | - | `reap/toolblock_all.npz`, `reap/special_all.npz`, `notool/reap/answerblock_nt2.npz` |
| Keep sets (bool [40, 384]) | - | `reap/keep_*`, `reap/protect`, `notool/reap/keep*` |
| Expert-routing report stats (activity, coverage curves, per-layer kept shares, overlaps, Jaccard, tool vs no-tool) | `expert_stats.json`, `expert_activity.json`, `expert_stats_notool.json`, `compare_tool_notool*.json` | `notool/reap/` |
| Teacher-forced NLL screens (per sequence) | `nll_*.jsonl`, `notool_nll_*.jsonl` (incl. export parity) | `results/`, `notool/results/` |
| Free-generation / answer-writing probes | `gen_probe*.jsonl`, `notool_copy_probe_*.jsonl` | `results/`, `notool/results/` |
| Raw observer records (per token: ids, positions, 40 x 6 experts, router weights, amplitudes) | - | `obs/` (tool), `notool/obs/` (20M tokens), `notool/pilot/obs/` |
| Token streams + roles (calibration, dev, eval traces) | - | `tok/`, `notool/tok/` |
| Rollout snapshot actually used (no-tool) | `notool_rollout_snapshot.json` (sha256) | `notool/rollouts_snapshot/` |
| Server time series (10 s stats: tok/s, running, KV %, prefix hit %; DSpark lines; mask switches) | `server_series.json`, `throughput_series.json` | `results/`, `box_runs/` |
| Box logs (every job), environment locks, plan pack | `locks/`, `plan/` | `box_logs/`, `e043_experiment/` |
| Checkpoints | - | models `...-arc-engram64`, `-arc-E72`, `-arc-E64`, `-arc-E48` (tool), `-arc-notool-E72` |

Known gaps:
- The tool round's early `runs/tp4B/server.log` was overwritten when B restarted at 20:52 UTC. Its throughput series had been
  extracted into `throughput_series.json`, and an earlier copy is in HF `box_runs/tp4B`.
- Not run: a no-tool ARC control with all 384 experts + Engram 48 GiB. Engram 48 GiB was only evaluated together with E72; the
  tool round's 64 GiB control was lossless.
- The combined tool + no-tool Engram counts (pilot, `counts_comb*`) were not kept (the user wanted no-tool-only statistics;
  they are the sum of the two censuses).
- Pruned-drafter DSpark (16-32 drafter experts) was not tested (E042 put it on hold).

## Reproduce

Box: Vast 8x RTX PRO 6000 (SM120), >= 1 TB RAM (two servers each pin the full ~203 GB Engram table), 1 TB disk.
Environment locks are in `locks/`: `venv_vllm.freeze` (the vLLM fork stack from E042's freeze), `tvenv.freeze`
(numpy/pyarrow/tokenizers tools), and `nvidia-smi.txt`, `topology.txt`, `system.txt`.

1. **Environment.** `uv venv --python 3.12 venv && uv pip install -r locks/venv_vllm.freeze --extra-index-url
   https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`. Then run
   `python patches/apply_e043_patches.py venv/lib/python3.12/site-packages/vllm` and `python patches/apply_compact_engram.py <same>`.
2. **Checkpoint.** `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
   --local-dir /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017`. Use RAM: the box's disk wrote only ~120 MB/s.
   `scripts/dl_shm2.sh` is the watchdog.
3. **Tokens.** `scripts/tokenize_rollouts.py --model-dir <ckpt> --parquet <e031/e036/e038 parquets> --out data/tok/shard$i
   --shard $i/32`. Run the same on the eval-trace parquets from `scripts/make_eval_parquet.py`.
   Check with `scripts/verify_tokenization.py --tok data/tok/shard*`.
4. **Engram.**
   - `scripts/engram_census.py count --tok data/tok/shard* --out data/engram/counts_calib.npz`.
   - `select --counts ... --budget-gib 8 16 32 48 64 96 112 --out-dir data/engram/masks`.
   - `coverage --masks ... --tok data/tok_eval`.
   - Optional stats: `scripts/engram_unseen.py`, `engram_unseen_kinds.py`, `engram_report_stats.py`.
5. **Serve.** `GPUS=0,1,2,3 TP=4 PORT=18000 CTRL=/dev/shm/e043/ctrlA.json bash scripts/serve.sh tp4A`, and the same on
   GPUs 4-7 with port 18001. Stop with `scripts/stop.sh <run> <gpus>`.
   - The JSON control file sets `expert_keep` / `engram_keep` / `observe`.
   - Runtime masks need a fresh `cache_salt` per configuration.
6. **NLL.** `scripts/nll_candidates.sh <out.jsonl> tag:<engram mask|none>:<expert keep|none> ...`, with `TOK`, `PERTASK`
   and `NSEQ` as env vars.
7. **REAP.**
   - `scripts/replay_reap.py --tok data/tok/shard* --tokens 1000000 --seed 1 --obs-dir /dev/shm/e043/obs/pilot
     --engram-keep <engram64>`, then `--tokens 9000000 --seed 2 --exclude <pilot manifest>`.
   - `scripts/reap_aggregate.py aggregate --obs <dirs> --out data/reap/stats_all.npz`, then `select --score
     contrib --decoder-gen-only --experts 48 64 72`.
8. **Tool-block rule.**
   - `PYTHONPATH=scripts python scripts/special_stats.py aggregate --region --obs <dirs> --out data/reap/toolblock_all.npz`.
   - `select --stats data/reap/stats_all.npz --special data/reap/toolblock_all.npz --score contrib --decoder-gen-only
     --protect 0 --weight 30 --experts 48 64 72 --out-dir data/reap/protect`.
   - Iterative variant: `scripts/iterative_prune.sh contrib 1`.
9. **Probes.** `scripts/gen_probe.py --samples 2 --out <jsonl> tag:<engram>:<keep> ...` and
   `scripts/decode_consistency.py`.
10. **Export.**
    - `scripts/export_pruned.py --src <ckpt> --expert-keep <keep.npy> --engram-keep <engram64.npz> --out export/<name>
      --text-only`. For more sizes, pass `--engram-from <first export>`, which hardlinks the shared Engram files.
    - Upload with `scripts/upload_model.sh`.
11. **ARC evals.** On the local machine, open an SSH tunnel to the ports and start a sandbox: `NAME=arc-sandbox-e043
    evals/sandbox_docker/start.sh`. Then run `SALT=<cfg> WORKERS=167 BASE=http://127.0.0.1:1800x/v1 scripts/eval_arc.sh
    <tag> --rescue-code`. Set the server's control file (masks) first.
12. **Report.**
    - `scripts/eval_report_stats.py evals/results results/eval_stats.json`.
    - `scripts/expert_report_stats.py` (on the box).
    - `scripts/make_results_report.py results reports/pruning_results.html` and
      `scripts/make_engram_report.py results/engram_stats.json reports/engram_census.html`.

## Cost and time

Box rental 12:23-~20:30 UTC on 2026-10-23, ~8 h at $13.73/h, about $110 (the whole 8-GPU node, including idle and
setup time). No API spend.
- Checkpoint download into RAM: ~25 min.
- Engram census: ~5 min on CPU.
- REAP replay of 10M tokens: ~28 min on 4 GPUs.
- ARC eval, 167 attempts: 1-1.5 h per model per 4-GPU server.

## Incidents

- 2026-10-24 14:46 UTC: the Vast account's credit ran out (balance -$0.49), and Vast stopped the 8x box and the 4xL4
  box mid-eval. `/dev/shm` was lost; the partial results are under "Interrupted runs".

- The image's auto-started SGLang service held GPU memory; stopped with `supervisorctl stop sglang model-ui`.
- The first download went to the RAID1 disk (~120 MB/s flush) and stalled; switched to `/dev/shm`. A watchdog that
  judged progress by file size killed healthy xet downloads (xet buffers in RAM); fixed to watch NIC bytes.
- `pkill -f` patterns matched my own shells twice (remote and local); the local AGENTS.md rule forbids killing by
  name on this host: use PIDs only.
- Observer bug, fixed before any statistics: non-local rows of the EP `moe_sum` input are uninitialized; masking by
  multiplication let NaN through (`where` now). Amplitudes are stored in fp32 (fp16 overflowed in deep layers).
- The full-workspace GitHub sync is blocked by the secret scan on `.SECRETS.yaml.swp` (an open vim swap file);
  E043 was pushed with only its own files.
