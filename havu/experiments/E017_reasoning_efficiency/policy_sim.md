# E017 phase 1A: train-check verifier and compute-allocation policies (offline, $0)

**Scope (reduced on request):** a stratified sample, not the full set. 32 of the 167 pairs, with every sub80s40 (3) and s20x60 (2) attempt on those pairs labelled: 160 attempts, 147 with an answer. No paid API calls. These are rough estimates with error bars, not final numbers.

Files:
- `label_train.py`: replay + train-check labeller (it reuses E016's replay code).
- `label_sample.json`: the stratified sample.
- `train_labels.jsonl`: one row per attempt. It also holds 7 extra rows, heavy sub80s40/sub96n attempts from an aborted full pass, which the simulation ignores.
- `policy_sim.py`: produces the tables below; `policy_sim.json` holds the same results.

## Method
- **Strata.** Each of the 167 pairs is placed in a stratum using its 4–5 recorded attempts:
  - **A**: all correct (64 pairs);
  - **M**: mixed (71 pairs);
  - **W**: none correct (32 pairs).
  - The sample takes 6 A, 18 M and 8 W pairs (seed 17), over-sampling M because policy differences only arise there.
  - Population estimates weight each sampled pair by N_s/n_s.
- **Replay.** Each attempt's trace is replayed into a fresh sandbox session, as in E016: 0 state resets, and 1 of 1,884 replayed cells differed from its original output. The label is then taken from the first matching path, in this order:
  - **expr**: the submitted expression uses `test_input` (e.g. `solve(test_input)`); it is re-evaluated with `test_input` = each training input.
  - **cell**: the last executed cell that assigns the answer variable (`answer`/`ans`) and reads `test_input` is re-executed in a copy of the globals, with `test_input` rebound to each training input. It must reproduce the submitted grid on the real test input.
  - **fn**: a 1-arg session function that reproduces the submitted grid.
  - **transcript**: the whole transcript is replayed with `test_input` := each training input, and the answer is read the same way.
  - If several paths apply, the best train score is used. `train_pass` means all training pairs match exactly.
  - Hashes only leave the sandbox, and the test output never enters it.
- **Policies.** Each policy's result is the exact mean over all orderings of a pair's attempts, which are treated as i.i.d. draws. sub80s40 and s20x60 are pooled, so draws mix the two configs.
  - pass@1-slot: attempt_1 is correct.
  - pass@2: either slot is correct.
  - Train-check selection groups attempts by answer and ranks the groups by: any train_pass, then the number of passing attempts, then the best train fraction, then votes, then order.
  - Tokens are output tokens per pair. $ is the recorded `est_cost_usd` (input + output, off-peak) for 167 pairs.
- **CIs.** 95% bootstrap over the sampled pairs within each stratum (2,000 resamples), on the difference from the 2-attempt baseline over the same pairs.

## What the answers look like (common patterns)
- **Traceability is ~100%.**
  - 131/147 answers were submitted as `answer` and 7 as `ans`; 12 were `solve(test_input)`.
  - The answer is almost always set by one top-level cell that reads `test_input` directly. So the **cell** path traced 125/147, **expr** 12, **fn** 2 and **transcript** 13; 0 were untraceable.
  - A Kaggle-time verifier can therefore use the same trick: re-run the answer cell with `test_input` rebound.
- **False positives (train_pass but wrong, 18/84 raw).** They cluster in a few tasks: d35bdbdc (6), b0039139 (5) and 71e489b6 (3), plus 221dfab4, 8698868d, 8b9c3697, 9385bd28 and 2d0172a1.
  - These are rules that fit every training pair but miss a feature that only appears in the test input. Different attempts often land on the same wrong rule.
- **False negatives (correct but train_fail, 16/63 raw).** These are mostly answers hand-built for the test input: hard-coded coordinates or colours, giving train_frac 0 (e.g. 269e22fb, 6ffbe589, a25697e4, 4e34c42c, d8e07eb2).
  - A few are general rules that fail one training pair (581f7754: 2/3, five attempts; 2d0172a1: 3/4).
- **Attempts with no answer (13/160)** are all wrong, as in E016.

### Raw counts (sampled attempts, sub80s40 + s20x60)

| stratum | attempts | answered | traceable | train_pass: correct / n | train_fail: correct / n | untraceable: correct / n | no answer |
|---|---|---|---|---|---|---|---|
| A | 30 | 30 | 30 | 24 / 24 | 6 / 6 | 0 / 0 | 0 |
| M | 90 | 82 | 82 | 42 / 54 | 10 / 28 | 0 / 0 | 8 |
| W | 40 | 35 | 35 | 0 / 6 | 0 / 29 | 0 / 0 | 5 |
| all (raw) | 160 | 147 | 147 | 66 / 84 | 16 / 63 | 0 / 0 | 13 |

### Population-weighted verifier quality

| quantity (population-weighted) | estimate | 95% CI |
|---|---|---|
| P(correct \| train_pass) | 86% | 76–93% |
| P(correct \| train_fail) | 36% | 13–54% |
| P(correct \| untraceable) | nan% | nan–nan% |
| recall: P(train_pass \| correct) | 80% | 59–95% |

### Test outputs with a train-passing attempt (all 4–5 attempts of the pair)

| stratum (pop. size) | sampled pairs | ≥1 train_pass attempt | ≥1 train_pass & correct | train_pass only on wrong answers |
|---|---|---|---|---|
| A (64) | 6 | 5 | 5 | 0 |
| M (71) | 18 | 16 | 16 | 0 |
| W (32) | 8 | 2 | 0 | 2 |

Untraceable answers do not occur in this sample, so that row is empty (`nan`). P(correct | no answer) = 0.

Over all 4–5 attempts of a pair:
- No pair in A or M had train_pass only on wrong answers.
- In W, 2 of 8 sampled pairs had a train-passing (wrong) answer.

## Policies
Baseline = 2 attempts per pair, drawn from the pooled sub80s40 + s20x60 attempts. Its 74.2% pair-weighted pass@2 sits between the recorded s20x60 (75.5%) and sub80s40 (77.5%) values.

| Policy | pass@1-slot pair-w / task-w | pass@2 pair-w / task-w | Δ pass@2 (pp) [95% CI] | out tok / pair | Δ tok / pair | $ / full set (167 pairs) |
|---|---|---|---|---|---|---|
| (a) 1 attempt (pass@1) | 62.9 / 62.3 | 62.9 / 62.3 | -11.3 [-12.3, -10.4] | 40.6k | -40.6k | $4.44 |
| (a) 2 attempts (pass@2 baseline) | 65.1 / 65.8 | 74.2 / 74.5 | +0.0 [+0.0, +0.0] | 81.3k | +0.0k | $8.88 |
| (b) gated: 2nd attempt only if 1st fails train check | 67.6 / 68.4 | 70.4 / 71.2 | -3.8 [-5.9, -1.7] | 63.5k | -17.8k | $6.93 |
| (b) gated, up to 3 attempts | 70.3 / 71.6 | 72.5 / 73.6 | -1.8 [-4.8, +1.1] | 82.2k | +0.9k | $8.97 |
| (b) gated, up to 4 attempts | 71.8 / 73.3 | 73.5 / 74.7 | -0.7 [-4.4, +2.9] | 98.8k | +17.5k | $10.78 |
| (f) 2 attempts, +3rd only if neither passes train | 70.3 / 71.6 | 76.2 / 76.9 | +2.0 [+1.0, +3.2] | 100.0k | +18.7k | $10.92 |
| (f) 2 attempts, +up to 2 more while none passes train | 71.8 / 73.3 | 77.3 / 78.0 | +3.1 [+1.6, +4.7] | 116.6k | +35.3k | $12.72 |
| (c) best-of-3, train-check select | 72.8 / 73.8 | 76.8 / 77.6 | +2.6 [+1.4, +3.8] | 121.9k | +40.6k | $13.32 |
| (c) best-of-3, majority only | 70.7 / 71.7 | 75.7 / 76.6 | +1.5 [+0.5, +2.7] | 121.9k | +40.6k | $13.32 |
| (c) best-of-4, train-check select | 75.4 / 76.4 | 78.6 / 79.5 | +4.4 [+2.6, +6.5] | 162.6k | +81.3k | $17.75 |
| (c) best-of-4, majority only | 75.2 / 76.2 | 78.0 / 79.0 | +3.8 [+2.4, +5.9] | 162.6k | +81.3k | $17.75 |
| (c) sub80s40 only: best-of-3, train-check | 75.7 / 76.6 | 78.9 / 80.1 | +4.6 [+0.9, +8.3] | 131.3k | +50.0k | $14.39 |
| (c) sub80s40 only: 2 attempts (pass@2) | 67.8 / 68.5 | 76.9 / 77.3 | +2.7 [-0.4, +5.7] | 87.5k | +6.2k | $9.60 |
| (d) @25k: 1 attempt | 29.1 / 27.1 | 29.1 / 27.1 | -45.1 [-58.1, -33.0] | 22.2k | -59.1k | $2.43 |
| (d) @25k: 2 attempts | 36.4 / 35.9 | 37.6 / 36.8 | -36.6 [-50.0, -24.1] | 44.4k | -36.8k | $4.86 |
| (d) @25k: best-of-3, train-check | 41.3 / 41.7 | 42.5 / 42.6 | -31.8 [-45.5, -19.8] | 66.7k | -14.6k | $7.29 |
| (d) @25k: best-of-4, train-check | 45.1 / 46.2 | 46.2 / 47.0 | -28.1 [-43.5, -15.8] | 88.9k | +7.6k | $9.72 |
| (d) @25k: gated, up to 4 | 44.4 / 45.7 | 44.4 / 45.7 | -29.9 [-44.2, -17.5] | 66.7k | -14.5k | $7.30 |
| (d) @30k: 1 attempt | 32.6 / 30.7 | 32.6 / 30.7 | -41.6 [-54.2, -29.7] | 25.5k | -55.8k | $2.79 |
| (d) @30k: 2 attempts | 39.5 / 39.3 | 40.7 / 40.2 | -33.5 [-48.2, -22.5] | 51.0k | -30.3k | $5.58 |
| (d) @30k: best-of-3, train-check | 44.2 / 44.7 | 45.3 / 45.6 | -28.9 [-43.7, -17.3] | 76.5k | -4.8k | $8.37 |
| (d) @30k: best-of-4, train-check | 46.9 / 48.0 | 47.9 / 48.8 | -26.3 [-42.5, -13.9] | 102.0k | +20.7k | $11.15 |
| (d) @30k: gated, up to 4 | 46.1 / 47.5 | 46.1 / 47.5 | -28.1 [-43.7, -15.9] | 74.9k | -6.4k | $8.20 |
| (e) low → sub80s40h 2nd attempt only if check fails | 64.3 / 64.5 | 68.6 / 68.7 | -5.7 [-8.5, -2.4] | 67.5k | -13.8k | $7.36 |
| (e) low + sub80s40h always (mixed pass@2) | 64.3 / 64.5 | 73.3 / 72.4 | -0.9 [-4.3, +2.9] | 89.3k | +8.0k | $9.78 |

Policy definitions:
- **(b) gated:** run the next attempt only while no attempt run so far passes the train check. attempt_1 is the first train_pass answer, otherwise attempt 1; attempt_2 is the other answer, if one was run.
- **(f) two-plus:** always run 2 attempts, then add attempts only while neither passes.
- **(d)** keeps an attempt only if it submitted within B output tokens. Otherwise it becomes "no answer" at B tokens.
- **(e)** uses the recorded sub80s40h (high-effort) attempts on the same pair as attempt 2. They are unlabelled, so attempt_1 stays the low-effort answer.

## Caveats
1. **Small sample.**
   - Only 18 mixed pairs carry the accuracy differences. The CIs cover pair sampling only, not run-to-run variance; E011 shows ±3–5 pp per sample. Treat the gains as ±2 pp-level signals.
   - A and W pairs are scored exactly by construction (all correct / all wrong). However, a stratum is defined by just 4–5 attempts, so a fresh run would re-sort some pairs.
2. **The draws are recycled.**
   - Best-of-k with k=4 uses nearly all of a pair's attempts, so the variety of k-sample outcomes is limited.
   - Mixing two configs inflates diversity a little compared with k samples of one config.
3. **(d) is strongly pessimistic.** Truncating finished long runs at B discards every long correct rollout. The median correct rollout is ~25k tokens, so that alone halves pass@1. A real run with a 25–30k soft budget would push earlier submissions; for reference, E010 s20x60 scored 61% pass@1 with a 20k soft budget. Do not read (d) as a measurement of short-attempt policies; it only shows that **finished-early attempts are not enough by themselves**.
4. **The verifier is weaker than it looks on hard tasks.** False positives come from systematically wrong rules that repeat across attempts, so extra samples do not dilute them. The W stratum (no attempt correct) gains nothing from any policy.
5. **Contamination.** The public eval may be in pretraining data (see E004), and the absolute levels may be inflated. Differences between policies on the same attempts are what matters here.
6. **The train check is done offline by re-executing code.** In a live harness it costs about 0 tokens (one sandbox call). The `--verify-submit` feedback in E011 is a different intervention: the model sees the check result.

## Takeaways
- **Gating the second attempt saves tokens but loses pass@2.**
  - With (b), attempt_1 accuracy rises (65.1 → 67.6%) and tokens drop 22%.
  - But on Kaggle the empty second slot costs 3.8 pp of pass@2, because about 14% of train-passing answers are wrong and those pairs then never get a second try.
- **Spend extra compute only where the check fails.** Policy (f), 2 attempts plus a 3rd only if neither passes, gives **+2.0 pp pass@2 [+1.0, +3.2] for +23% tokens**. That is about $10.9 per full set against $8.9.
- **Train-check selection beats majority vote.** Best-of-3 is +2.6 pp versus +1.5 pp for majority vote; best-of-4 is +4.4 pp versus +3.8 pp.
- **Escalating failed pairs to high effort does not help.** Using sub80s40h for the second attempt did not beat a second low-effort attempt here; compare (e) with (a)/(b).
