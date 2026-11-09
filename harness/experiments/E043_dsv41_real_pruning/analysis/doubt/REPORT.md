# E043: why pruned DeepSeek-V4.1-Flash-1017 models reason longer, and how much of it is "self-doubt"

Offline, read-only analysis of existing traces (2026-10-26). CPU only: no API calls, no GPU, no spend.
Scripts and JSON outputs are in this directory; see "Reproduce" at the end.

## Summary

- **The extra length is more steps, not longer or more doubtful steps.**
  - E128 writes +7.8K output tokens per attempt (+28%) vs API full on the same 167 pairs.
  - Thinking per turn is about the same: 1,772 vs 1,707 tokens. E128 uses +3.1 turns.
  - The per-word rate of core doubt markers ("Wait", "Hmm", "Actually", re-check, reconsider) is **not** higher: 9.7 vs 10.5 per 1K thinking words.
- **What does change is hedging and register.** Per 1K thinking words, full → E128:
  - question marks 24 → 35;
  - "maybe/perhaps" 12.3 → 16.9;
  - "could/might" 4.1 → 5.8;
  - "likely/probably" 0.75 → 1.20;
  - "Wait" 2.75 → 3.24 (+18%);
  - "Actually" 2.89 → 2.25;
  - explicit "Let me verify / re-check / reconsider" 0.86 → 0.02. The whole "Let me ..." construction nearly vanishes; "Need ..." replaces it.
  - The hedge rates grow monotonically with pruning depth (E128 < E112 < E96).
- **Decomposition of E128's +7.8K vs API full** (167 pairs; pair-averaged):
  - (a) error recovery +1.2K (15%), or 27% counting the failed calls' code;
  - (b) after the first all-pass output +1.0K (12%);
  - (c) exploration before the first pass, in attempts that reach one: +2.6K (33%);
  - (c') attempts that never reach a passing check: +3.1K (39%).
  - E112 and E96 are dominated by (a) and (c').
- **Doubt is mostly productive.** A blind audit of 220 markers found 59-68% lead to a new hypothesis or a correction. E128's markers are somewhat more often idle (38% vs 28%; not significant).
  - In the full model, "Wait" marks the *correct* attempt of a discordant pair (30/37, p < 0.001; pre-pass turns only 25/37, p = 0.047).
  - "maybe" marks the *wrong* attempt (27/37, p = 0.008).
  - In E128 neither association holds.
- **Levers:** the length is mostly a capability effect: more hypotheses, more failures, typos. No serving-time lever found here gives a large saving without accuracy risk.
  - Ranked: (1) the typo / identifier-copy fix (parallel agent); (2) more experts or healing; (3) an "all training pairs pass" harness note (a small saving); (4) decision-token weighting (speculative, small).
  - Not recommended: a logit bias against "Wait", and hard caps.

## Data and definitions

| model (tool harness, sub80s40 + minimal header, T=1.0) | run(s) | attempts used |
|---|---|---:|
| API full (E030) | `deepseek-flash_low_e030_hdrmin` | 334 |
| self-hosted full | `dsv41_low_e043_full_ref_sub80`: 172 finished, 162 errored records dropped (the shorter attempts finished) | 172 |
| E128 | `dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80` | 334 |
| E112 | main run attempt 0 + `_r2` attempt 0 (main attempt 1 and `_r3` excluded) | 334 |
| E96 | `dsv41_low_e043_e96t_w10_u2u3_gptq_eng64_sub80` | 334 |
| dev split, E128 | `devs_e128t_t1` / `_minp05` / `_topp95` (33 non-eval pairs) | 33 / 33 / 32 |
| full-model training rollouts | E031 `arc2_train`, `nvarc_b0`; E036 `nvarc_b1`, `nvarc_b1x` (s20x60 + header) | 8,754 |

- **Tokens.**
  - Per-turn completion tokens come from the trace events. Their sum equals `usage.completion` exactly in every run.
  - Thinking, code, content and submit tokens come from the V4.1 tokenizer applied to `reasoning_content`, the `run_python` code, `content` and the `submit_answer` argument.
  - The rest is DSML / format overhead.
- **Python error:** a tool output matching `Traceback|\w+Error:|\w+Exception:`, the same rule as `scripts/tool_paired.py`.
- **"All training pairs pass"** (`common.verdict`). A tool output counts only when all of these hold:
  - the code compares results with the training outputs (`['output']` plus a comparison);
  - the output shows at least n_train positive verdicts (True / OK / correct / `diff 0`), or an `all(...)` line;
  - no line about the training pairs shows a negative verdict (False / mismatch / nonzero diff). Lines about the test input are ignored.
- **Detector validation against the harness's own `train_pass`** (re-runs the submitted answer program):

  | run | recall (train_pass = True) | flagged pass where train_pass = False |
  |---|---:|---:|
  | E128 | 209/218 (96%) | 31/82 |
  | E96 | 180/186 (97%) | 15/105 |
  | self-hosted full | 123/126 (98%) | 11/46 |

  - `train_pass = False` is itself noisy: 33 of the 82 E128 attempts with `train_pass = False` were correct.
  - A spot check of 14 such flags found ~3 false positives (exploratory outputs with True in them, partial checks).
  - The API run has no `train_pass`, but the detector's recall is the same on both stacks.
- **Turn contexts.**
  - `first`: turn 0.
  - `after_error`: the previous output was a Python error.
  - `post_pass`: any turn after the first all-pass output.
  - `after_fail_check`: the previous output was a failed train check.
  - `explore`: everything else.
- **Doubt markers.**
  - Core markers (`common.MARKERS`): Wait, Hmm, Actually, double-check / re-check / verify again, reconsider / re-examine / re-read, hold on / oops, not sure.
  - Hedges and verification phrases are listed in `phrases.py`.
  - Rates are per 1K thinking words (whitespace words).
- **Earlier claims, re-checked.**
  - E128 tokens after the first all-pass output: 9.2% of all output (the earlier session reported 8.1%; my detector is slightly more permissive).
  - E128 output: 85.8% thinking, 12.3% code, 1.9% DSML / format overhead and answer. The earlier 89/11 split probably folded the overhead in differently.

## 1. Where the extra tokens go (tool harness)

All attempts per model (means per attempt):

| | API full | SH full (172) | E128 | E112 | E96 |
|---|---:|---:|---:|---:|---:|
| pass@1 (attempt accuracy) | 74.0% | 78.5% | 71.9% | 68.9% | 63.8% |
| output tokens | 27.4K | 20.9K | 35.2K | 35.7K | 30.5K |
| thinking / code / overhead+answer | 23.8 / 3.1 / 0.5K | 17.8 / 2.6 / 0.5K | 30.2 / 4.3 / 0.7K | 29.8 / 5.1 / 0.8K | 24.3 / 5.2 / 0.9K |
| thinking share | 86.7% | 85.2% | 85.8% | 83.4% | 79.8% |
| turns | 13.9 | 12.7 | 17.1 | 21.0 | 23.6 |
| thinking tokens per turn | 1,707 | 1,406 | 1,772 | 1,417 | 1,030 |
| attempts reaching an all-pass output | 75.4% | 77.9% | 72.8% | 66.5% | 59.6% |
| tokens before / after the first all-pass output | 25.4 / 2.1K | 19.4 / 1.5K | 32.0 / 3.2K | 33.3 / 2.4K | 28.6 / 1.9K |
| share after the first pass (all output) | 7.6% | 7.2% | 9.2% | 6.7% | 6.2% |
| post-pass turns (attempts that pass) | 3.33 | 2.91 | 3.72 | 3.70 | 4.42 |
| submits in the turn right after the first pass | 18.7% | 18.7% | 14.8% | 11.7% | 12.6% |
| 40-turn stops / 80K budget stops | 8 / 2 | 0 / 0 | 11 / 27 | 33 / 23 | 58 / 2 |
| Python-error share of calls | 2.5% | 2.9% | 14.8% | 30.9% | 38.7% |

- **Before the first pass (attempts that reach one):** full uses 8.7 non-recovery turns at 2,100 tokens each; E128 uses 9.7 turns at 2,301 (+10% per turn).
  - Failed train checks before the first pass are the same: 1.08 vs 1.14.
  - So E128 does not test more wrong rules. It spends more inspection turns, and more words, before it writes a check that passes.
- **The soft budget** (40K "aim to finish" note):
  - Tokens written beyond 40K: API full 4.2K per attempt (15% of all output, 25% of attempts cross 40K); E128 7.8K (22%, 39% cross); E112 7.9K; E96 4.5K.
  - That is +3.6K of E128's +7.8K. This is a time cut that overlaps the phase decomposition below.
  - Once past 40K, E128 writes 20.2K more on average (full: 16.6K), and 27 of its 129 crossing attempts run into the 80K hard limit (full: 2 of 85).
  - Of the crossing attempts, 56 E128 attempts are still correct (full 34). In 40 E128 attempts the first pass comes only after 40K.
- **Post-pass tail.** Tokens after the first pass, per attempt that reaches one:
  - median 690 (full) vs 1,244 (E128); p90 6.9K vs 12.2K.
  - 64% of E128's post-pass tokens come from 32 attempts (13%) with more than 10K post-pass tokens. Their accuracy is 62%. 17 of them later found a new passing rule and 10 had a later failing check.
  - In full, the same tail holds 50% of post-pass tokens (17 attempts, 59% accuracy).
  - So the bulk of post-pass length is rework when the first passing rule does not carry over to the test input, not hesitation.
  - Short hesitation is cheap: the 1-3K bucket holds 69 E128 attempts vs 50 for full, about 0.3K per attempt in total.

## 2. Doubt markers (tool harness)

### Rates per 1K thinking words

Full table: `results/phrases.json`.

| marker | API full | SH full | E128 | E112 | E96 |
|---|---:|---:|---:|---:|---:|
| Wait | 2.75 | 2.89 | 3.24 | 3.18 | 2.98 |
| Hmm / Hm | 3.75 | 3.76 | 4.02 | 3.83 | 3.75 |
| Actually | 2.89 | 3.15 | 2.25 | 2.25 | 2.01 |
| But wait / But let's | 0.33 | 0.43 | 0.20 | 0.17 | 0.10 |
| double-check / re-check | 0.25 | 0.30 | 0.08 | 0.06 | 0.01 |
| "Let me/let's verify / check / re-examine ..." | 0.86 | 1.04 | 0.02 | 0.02 | 0.01 |
| "Need (to) verify / ensure / check ..." | 0.18 | 0.23 | 0.39 | 0.32 | 0.42 |
| Alternatively | 0.29 | 0.29 | 0.12 | 0.11 | 0.06 |
| **maybe / perhaps / possibly** | **12.3** | 11.1 | **16.9** | 17.3 | 18.1 |
| **likely / probably** | **0.75** | 0.72 | **1.20** | 1.35 | 1.79 |
| **could / might** | **4.1** | 3.4 | **5.8** | 5.8 | 6.3 |
| **question marks** | **24.0** | 22.8 | **35.2** | 36.0 | 42.8 |
| **core doubt** (Wait, Hmm, Actually, re-check, reconsider, hold on, unsure) | **10.5** | 11.1 | **9.7** | 8.8 | 8.7 |
| thinking words per attempt | 13,048 | 9,513 | 16,920 | 16,778 | 13,764 |

- **Lexical shift** (`lexicon.py`, sentence-initial words per 1K sentences, API full → E128):
  - down: "Let" 29.3 → 2.2 ("Let me" is gone; "Let's" stays, 73 → 86), "Contradiction" 0.55 → 0.04, "Consistent" 0.35 → 0.08;
  - up: "Need" 8.4 → 22.4, "It" 6.8 → 16.1;
  - turn openings: "We need" 11 → 49 per 1K turns; "I keep (typo ...)" 0 → 9.3 (E96: 18.8, plus "I typo" 9.9).
  - Pruning changed the register (terser "Need ..." notes) and removed the full model's explicit verification phrasing, while verification itself still happens.
- **Where markers occur.** Core markers per 1K words, by context (API full / E128):

  | context | full | E128 |
  |---|---:|---:|
  | first turn | 15.9 | 12.7 |
  | explore | 10.5 | 9.9 |
  | after a failed check | 9.5 | 9.5 |
  | after an error | 8.9 | 9.9 |
  | first post-pass turn | 8.9 | 6.3 |
  | later post-pass turns | 10.5 | 8.2 |

  - 82% of markers are in exploration turns for both models.
  - E128's post-pass turns are longer (607 vs 336 tokens for the first post-pass turn; 1,477 vs 1,053 for later ones) and denser in hedges: per 1K words "maybe" 17.2 vs 12.1, "Need verify" 1.24 vs 0.68, "?" 32.6 vs 23.3. They contain fewer "Wait"-type markers.
  - The marker share by position in a turn is the same for both: 11% in the first 10% of the turn.
- **Re-verification of things that already passed** is rare in both models, and not higher in E128:
  - post-pass turns that re-run a passing train check: full 0.20 per attempt, E128 0.18;
  - near-identical code reruns: 6 vs 5 turns in total;
  - E128 has more post-pass Python errors (79 vs 15).
  - The re-checking that does happen is in prose: re-reading printed values, re-asking "is the output 29 rows?", and so on.
- **Repetition and loops:** none in either model.
  - Within-turn 6-gram repetition: 0.25% (full) vs 0.18% (E128) of 6-grams; no turn above 30%.
  - Sentences repeated verbatim from earlier turns: ≤ 0.08% of words.

## 3. Decomposition of the extra length

**vs API full.** Pair-averaged over the 167 pairs; Δ in tokens per attempt, share of the total Δ in brackets.

| model | Δ total | (a) error-recovery turns | (b) post-pass, non-recovery | (c) pre-pass exploration, attempts that pass | (c') attempts that never pass |
|---|---:|---:|---:|---:|---:|
| E128 | +7,780 | +1,184 (15%) | +966 (12%) | +2,579 (33%) | +3,051 (39%) |
| E112 | +8,259 | +3,064 (37%) | +163 (2%) | −380 (−5%) | +5,412 (66%) |
| E96 | +3,036 | +3,178 (105%) | −396 (−13%) | −3,816 (−126%) | +4,071 (134%) |

- (a) counts the turn that reacts to a Python error. Adding the failed calls' own code gives E128 +2.1K (27%), E112 +5.0K (61%), E96 +5.3K.
- E96 is short only because it hits the 40-turn cap (58 stops), which ends its attempts early.
- **Robustness checks for E128:**

  | reference / stratum | Δ total | (a) | (b) | (c) | (c') |
  |---|---:|---:|---:|---:|---:|
  | self-hosted full, 172 same attempts | +8,711 | 14% | 15% | 45% | 26% |
  | pairs where full passes in both attempts (109) | +7,433 | 13% | 6% | 16% | 65% |
  | pairs where both models are correct in all attempts (89) | +6,152 | 16% | 10% | 34% | 40% |

  - (d) Residual (usage vs the per-turn sum): 0.
- **Approximate answer for E128.**
  - About one sixth is error recovery: one quarter including the wasted code.
  - About one ninth is after success.
  - About three quarters is exploration: more inspection turns and longer prose before a passing rule, and more attempts that never find one and run to the budget.
  - Of the c' share, about 40% is the rate effect (+2.7 pp of attempts never pass, × ~46K tokens) and about 60% is the length effect (never-pass attempts run 52.8K vs 46.2K).
  - "Self-doubt" proper does not appear as a separate bucket. It is spread through (b) and (c) as hedged prose.
- **API vs self-hosted full** (same 172 attempts): the API full is 1.06K longer, mostly post-pass (+0.57K). The stack or sampler matters at this scale, so the self-hosted full model is the fairer reference.

## 4. Is the doubt useful?

**Blind audit.**
- Sample: 220 Wait / Hmm / Actually occurrences (110 per model; 40 pre-pass and 70 post-pass, excluding after-error turns), shuffled and model-blind.
- Labeler: a separate Sonnet agent. Files: `blind/excerpts.txt`, `blind/key.json`, `blind/labels.json`.
- Labels: P = new hypothesis, C = correction or new fact, D = doubt without change, F = filler, T = typo / mechanics.

| sample | P | C | D | F | T | D share | P+C share |
|---|---:|---:|---:|---:|---:|---:|---:|
| full, before the first pass (40) | 20 | 7 | 11 | 2 | 0 | 28% | 68% |
| E128, before the first pass (40) | 18 | 6 | 16 | 0 | 0 | 40% | 60% |
| full, after the first pass (70) | 28 | 17 | 20 | 3 | 2 | 29% | 64% |
| E128, after the first pass (70) | 30 | 11 | 26 | 1 | 2 | 37% | 59% |

- Fisher tests on D: p = 0.34 (pre) and 0.37 (post); pooled 31/110 vs 42/110.
- An earlier unblinded audit of mine (40 post-pass markers each) gave D 7/40 vs 16/40 (p ≈ 0.05). Same direction, single rater.
- **Verdict: weak evidence.** E128's markers are ~1.3× as often idle. E128 also has 1.19× more core markers per attempt, which gives ~1.6× the idle-doubt episodes per attempt.
- Rough cost: (164 × 0.38 − 138 × 0.28) ≈ +24 idle episodes per attempt × ~100-150 tokens (the median paragraph after a marker is 86 tokens for full and 100 for E128; the mean is 121 vs 150) ≈ +2.4-3.6K as an upper bound.
  - The whole paragraph is not waste, so the real saving available is much smaller. This is a hypothesis, not a measurement.

**Same-pair outcome tests.**
- Sample: discordant pairs (one attempt correct, one wrong, same model); sign tests on per-word density.

  | model | "Wait": right attempt higher | "maybe": wrong attempt higher |
  |---|---|---|
  | API full | 30 of 37 (p < 0.001); pre-pass turns only 25 vs 12 (p = 0.047) | 27 of 37 (p = 0.008); pre-pass only 26 vs 11 (p = 0.02) |
  | E128 | 18 vs 14 (p = 0.6); pre-pass 20 vs 12 (p = 0.2) | 17 vs 15 (p = 0.86) |
  | E112 | n.s. | n.s. |
  | E96 | n.s. | 26 vs 13 (p = 0.05) |

  - In the full model, "Wait" goes with self-correction that works, and "maybe" goes with unresolved hypotheses.
  - In the pruned models "Wait" has lost that association.
  - Core-marker density overall does not predict the outcome in any model (full 14 vs 23; E128 14 vs 18).
  - Cross-model: E128's extra marker density on a pair does not correlate with E128 losing that pair to full (r = 0.03).
- **Action after a turn with a core marker** (turns ≥ 1): E128 turns with markers re-run near-identical code more often (50 of 3,992 vs 4 of 3,377 in full). Turns without markers re-run much more often (157 vs 8), mostly retries after typos.
- **Answer change after the first pass.**
  - The final answer grid was already printed at or before the first all-pass output in 61 of 252 full attempts and 67 of 243 E128 attempts.
  - A new, different passing rule appeared after the first pass in 35 attempts in each model. Accuracy in those: full 31/35, E128 26/35.
  - Post-pass rework is productive in the full model, and less so in E128 (n = 35 each; suggestive only).

## 5. No-tool harness (reported separately)

Pairs of the same (task, pair, attempt) with the full model.
- The no-tool models use no-tool keep sets: E128 = `e128_u2_gptq_eng56`, a different keep set from the tool E128.

| per 1K reasoning words | full | E128 | E96 | E72 |
|---|---:|---:|---:|---:|
| attempts / accuracy | 334 / 64.7% | 331 / 59.2% | 334 / 62.0% | 334 / 53.9% |
| completion tokens (median ratio vs full) | 31.4K | 41.6K (1.32×) | 37.4K (1.18×) | 31.0K (1.00×) |
| Wait | 3.33 | 3.53 | 4.10 | 3.96 |
| Hmm | 0.87 | 1.46 | 2.00 | 2.23 |
| Actually | 2.16 | 1.80 | 2.05 | 2.05 |
| maybe / perhaps | 6.6 | 9.6 | 10.8 | 11.8 |
| likely / probably | 1.23 | 1.30 | 1.48 | 1.74 |
| question marks | 32.8 | 36.8 | 42.7 | 42.9 |
| "Need" sentence starts | 2.1 | 2.3 | 3.1 | 2.8 |
| "Let me" | 0.006 | 0.009 | 0.004 | 0.015 |
| 6-gram repetition | 2.0% | 1.9% | 1.9% | 1.8% |

- The same hedge signature appears without tools: "maybe" +45%, "Hmm" +68%, "?" +12% for E128, growing with pruning depth. "Wait" barely moves (+6%), and there are no loops (as `reasoning_loops.json` found).
- The "Let me" register shift is a tool-mode effect: the full model does not write "Let me" in no-tool mode either.
- **Reasoning after the final answer grid is first written in full** (correct attempts where the grid could be located; low coverage, n = 22-56):
  - E128 vs full on 29 pairs where both are correct: 10.8% vs 12.2% of the reasoning, 2.3K vs 1.9K tokens.
  - No sign of more post-answer re-verification.

## 6. Proposals

### Token classes in the full-model training rollouts

- Data: E031 + E036, 8,754 rollouts, 126.5M generated tokens (`train_classes.py`).
- Every class is defined from the turn structure and text, with no task ids.

| class (generic definition) | tokens | share of generated |
|---|---:|---:|
| tool-call blocks (outside thinking / content; already ×10 in tool E128) | 13.8M | 10.9% |
| **decision turn:** the whole assistant turn right after the first all-pass tool output | 2.08M | 1.65% |
| ... in 46% of rollouts that turn is `submit_answer` itself | | |
| all turns after the first all-pass output | 7.92M | 6.3% |
| first 32 tokens of every turn that follows a tool output | 1.92M | 1.52% |
| ... after a pass / failed check / Python error / other output | 0.24 / 0.18 / 0.04 / 1.47M | 0.19 / 0.14 / 0.03 / 1.16% |
| turn that calls `submit_answer` | 1.47M | 1.17% |
| commit sentences ("All ... match/pass", Good / Great / Perfect, "submit") | 0.53M | 0.42% |
| 16 tokens after each Wait / Hmm / Actually (where the doubt is resolved) | 9.44M | 7.5% |
| turn after a Python error (rare in full: 1,155 turns) | 1.03M | 0.81% |

- With the current objective (all tokens + 9 × tool blocks), adding w = 10 on the decision turn would give that class ~8% of the weighted mass.
- **Dev-split check** (33 non-eval pairs; full = its E031 rollouts at s20x60, E128 = sub80s40, so the budgets differ):

  | | full | E128 |
  |---|---:|---:|
  | generated tokens per attempt | 5.87K | 6.26K |
  | Wait / Hmm / Actually per 1K words | 7.5 | 11.5 |
  | submits right after the first pass | 77% | 61% |
  | error-recovery share of tokens | 0.2% | 3.6% |
  | attempts reaching a pass | 94% | 94% |

  - Consistent with the eval findings, but small and confounded by budget.

### Logit-bias facts (V4.1 tokenizer)

- **Token ids:**
  - "Wait" = 39059, " Wait" = 37157, " wait" = 6361;
  - "Actually" = 73498, " Actually" = 48837;
  - "Maybe" = 32064, " Maybe" = 17519, " maybe" = 10796.
  - "Hmm" is two tokens, "H" (42) + "mm" (6002), and " Hmm" = 437 + 6002. "mm" also occurs in code (0.11 per 1K code tokens), so Hmm cannot be biased safely.
- **Rates** (120-attempt samples): Wait tokens per 1K thinking tokens 1.47 (full) vs 1.79 (E128). " maybe" also appears in code comments (0.32 per 1K code tokens in E128).
- **Upper bound on the saving:** removing every excess E128 Wait episode with no substitution saves about 0.32 per 1K × 30K thinking tokens × 150 tokens ≈ 1.4K per attempt. That needs every such episode to be waste, but 60%+ of markers are productive and full's "Wait" predicts success.

### Ranking

| # | lever | when | expected token effect (E128) | accuracy risk | evidence |
|---|---|---|---|---|---|
| 1 | Fix typos / identifier copying: keep-set weighting of identifier-copy tokens, or healing. A parallel agent owns this. | needs box | removes most of (a): −1.2K direct, up to −2.1K with the wasted code (15-27%); E112 / E96 −3K to −5K | low, and positive for accuracy (fewer wasted turns) | strong (measured) |
| 2 | More experts (E128 → E144/E160) or short healing SFT on full-model traces | needs box | hedging and (c)/(c') track pruning depth monotonically; each step E96 → E112 → E128 cut hedges and errors. Largest lever on the 72% exploration share; size unknown | accuracy up; memory cost on 4×L4 | moderate (monotone trends; no causal test) |
| 3 | Harness note once a train check passes: "[All training pairs pass. Check test-input assumptions with code, then submit.]" Use `common.verdict` or `--helpers check(solve)` as the trigger. | serving / harness now | targets (b) only (+1.0K, 12%) and full's own 2.1K; realistic −0.3K to −1K per attempt (1-3%) | low-moderate: 13% of attempts do long, productive post-pass rework (new passing rule in 17 of 32) and the note could cut it short. A/B on the dev split first. | moderate |
| 4 | Up-weight decision-point tokens in the expert score: the decision turn after the first all-pass output (1.65%), plus the first 32 tokens after a passing output (0.19%) and commit sentences (0.42%), w = 5-10 | needs box (new routing stats through the full model) | bounded by (b) ≈ 12%; any effect on global hedging is speculative | low (small mass), but every keep-set change shifts the general NLL | weak (hypothesis) |
| 5 | Stronger header wording | serving now | unknown, probably small: the header already forbids re-verification, and hedging is a calibration effect that wording rarely fixes (E030: prompt suffixes did not shorten 1017) | low | weak |
| 6 | Logit bias −2 to −5 on Wait ids 39059 / 37157 / 6361 | serving now | ≤ 1.4K upper bound; realistic much less (substitution) | **moderate-high:** "Wait" marks successful self-correction in full (30/37) | evidence against |
| 7 | Hard output cap (simulated, attempts over the cap counted wrong) | harness now | 60K: −7% tokens, −2.7 pp pass@1, −1.8 pp pass@2; 70K: −3% tokens, −0.9 / −1.2 pp | high per token saved | measured (conservative) |

**Measured vs hypothesis.**
- Measured: everything in sections 1-5 and the class frequencies.
- Hypotheses:
  - that hedging is a calibration effect of pruning (supported by the monotone trend and the earlier +12% dev NLL, but untested);
  - that decision-token weighting would preserve decisive experts;
  - the size of the harness-note saving.

## Excerpts

- **First turn after the first all-pass output, same pair (1818057f, pair 0):**
  - full (38 tokens): empty thinking, then `submit_answer`.
  - E128 (275 tokens): "Need verify test assumption: all changed are 4s and output expected. ... Need maybe rule could be mark full plus shapes and if a 4 belongs to several pluses? ... Need maybe ambiguity with overlapping pluses ... Need ensure rule matches training all. It does. likely correct. But let's double-check whether any plus shape partly consists of cells already changed etc."
- **E128, re-reading printed values with hedges (7b80bb43):** "Wait H=28 W=17? Output printed has 29 rows? Let's see test input rows count: from prompt 28? Actually list has 28 rows? Let's check shape. Our solve output printed 29 rows? It printed maybe 29? Let's check."
- **E128, typo awareness after errors:**
  - "I keep typo dbool; fix bool." (8b7bacbf)
  - "I keep typo; correct attribute `astype`? Actually it's `astype`? ... Wait I wrote `.ast` causing errors." (3a25b0d8)
- **Blind label P, full post-pass (cbebaa4b):** "<<Wait>>—but maybe the connector rule requires sharing at least 2 twos for ANY connection, so small3 cannot connect and remains UNMOVED ..."
- **Blind label D, E128 post-pass (cbebaa4b):** "<<Actually>> markers are part of piece, not separate; cannot occupy colored. Let's re-examine training outputs for pieces ..."
- **Blind label C, E128 post-pass (e87109e9):** "<<Wait>> unique 8 count=9, so initial marker is 3x3 in test, not 2x2. This is important." Productive: a test-assumption catch.

## Reproduce

```bash
D=experiments/E043_dsv41_real_pruning/analysis/doubt; C=<scratch cache dir>; PY=.venv/bin/python
$PY $D/extract.py --cache $C                    # per-turn features of all tool runs (~4 min, 20 procs)
$PY $D/analyze_tool.py --cache $C --out $D/results/tool_stats.json
$PY $D/phrases.py --cache $C --out $D/results/phrases.json
$PY $D/lexicon.py --out $D/results/lexicon.json --models full_api E128 E96
$PY $D/analyze_notool.py --out $D/results/notool_stats.json
$PY $D/train_classes.py --out $D/results/train_classes.json      # training rollouts + dev check (~4.5 min)
$PY $D/logit_bias_stats.py --out $D/results/logit_bias.json
```

- `common.py` holds the run list, the trace parser, the marker regexes and the all-pass detector.
- Ad-hoc outputs, from inline snippets recorded here:
  - `results/soft_budget.json`
  - `results/post_pass_tail.json`
  - `results/cap_simulation.json`
  - `results/within_pair_hedge.json`
  - `results/blind_audit.json`
- The eval tasks were used only to diagnose mechanisms. The weighting classes are generic and were measured on the training rollouts and the dev split.