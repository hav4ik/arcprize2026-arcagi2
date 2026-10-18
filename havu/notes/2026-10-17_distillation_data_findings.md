# 2026-10-17: DeepSeek → Qwen distillation data, findings so far

This note covers E025 (data collection) and E026 (trace analysis). Details are in `experiments/E025_train_rollouts/README.md` and `experiments/E026_trace_strategy/{deepseek_strategy,qwen_strategy}.md`.

## Goal
Train Qwen3.8-27B to imitate DeepSeek Flash's tool-use reasoning on ARC-AGI-2. Qwen is the Kaggle model; we only care about the DeepSeek harness as a teacher.

## Which DeepSeek harness is best (E009–E017)
All configs below are within the ~2–3 pp noise band of each other, so pick by tokens.

| Config | pass@1 | pass@2 | Tokens per attempt | Note |
|---|---|---|---|---|
| V3 (`lean2roles` + `--task-summary` + `--helpers` + `--train-check`) | 69.2% | 77.2% (78.4% with conditional 3rd attempt) | 39.5k | E017 recommendation |
| sub96n | 68.3% | 78.5% | 45.7k | |
| s25x80 | 64.7% | 75.3% | 35.9k | cheapest within 5 pp on both metrics |
| s20x60 | 62.4% | 76.5% | 32.9k | cheapest within 5 pp on pass@2 |

## E025: data collected (2026-10-16, $91.97, 1.24B tokens of which 133M output)
- **Batches:**
  - 1000 Kaggle ARC-2 training tasks (1,076 pairs): 94.0% pass@1, 12.1k tokens.
  - 4000 NVARC `nvarc_training` tasks: 68.2% pass@1, 30.0k tokens.
- **Yield:** 3,740 correct rollouts; 3,665 are correct and pass train.
- **The Kaggle training set is easy:**
  - 767 of its 1000 tasks are ARC-AGI-1, scoring 95.4%;
  - even the 233 ARC-2-only tasks score 89.8%, against 69% on eval.
- **NVARC tasks look hard, but are partly noise:**
  - P(correct | train_pass) is only 77.7%, and 765 train-passing answers are wrong;
  - all NVARC parents are ARC-AGI-1;
  - re-check those answers on the unused pairs in `manifest.jsonl` before treating them as hard negatives.
- **Throughput:** 24 → 64 workers gave about 1.6–1.8× throughput, and per-attempt speed fell 41%, so the DeepSeek API is the bottleneck. 128 workers is not worth a restart.
- **Where it is stored:**
  - task pool, extendable: `experiments/E025_train_rollouts/data/`;
  - rollouts: `evals/results/*_e025_*`;
  - HF, private: `livobereshka/arc-agi2-deepseek-v3-rollouts-e025`;
  - GitHub snapshot: `hav4ik/arcprize2026-arcagi2`.
- **Budget guide:** $0.021 per NVARC attempt and $0.009 per easy attempt. $5k ≈ 240k NVARC attempts, which is more than the 43k unique NVARC puzzles left.

## E026: does the task summary help DeepSeek? Barely.
- **No accuracy gain** (E017 V1 vs sub80s40: +0.6 pp pair-weighted, −1.8 task-weighted, 1 sample).
- **Without the summary, DeepSeek computes the same statistics itself:**
  - with no tools, by hand, about 33% of its tokens; slips there cause most no-tool failures;
  - with tools, in cheap Python before writing `solve()`, about 1.6k tokens: shapes 94%, bboxes 71%, components 56%, colour counts 48%, diff masks 40%.
- **What the summary removes:** −9% tokens, mostly hand-copied grid rows. It does not touch the real cost, prose reasoning over cell coordinates (about 9.4k tokens per attempt, unchanged). The saved tokens move into hypothesis prose.
- **The model re-derives summary items anyway** in 10/10 read traces: `solve()` needs masks, the tables are truncated, and some rules depend on edge or crossing cells.
- **Its framing misled the model in 6 of 10 read traces:** background vs separator colour, 8-connected multi-colour blobs, object views suggesting wrong rules. It helped in 5 of 10, when the rule could be read straight off `holes=` or the recoloured/removed lines. No summary number was wrong.
- **Risk:** it was designed from eval failure analysis, so it may be overfitted to local validation.

## E026: how Qwen differs from DeepSeek (matched harness: summary + helpers + roles)
- **Same train-pass rate** (77% vs 78%), but **P(correct | train_pass) is 72% vs 84%.** Qwen fits train with coincidental, absolute sub-rules where DeepSeek uses relational ones.
- **Qwen is slow to reach code:**
  - first turn 8–26k vs 4.3k tokens;
  - first `solve` at about 38k vs about 20k tokens;
  - the cause is the xhigh tier, since the first turn is 0.5–1.4k at medium.
- **Qwen trusts hand counts over code output** and drops correct rules after miscounting.
- **Tone is already similar** (terse), so SFT must change strategy, not style.
- **At the Kaggle 48k budget,** Qwen's failures are dominated by running out of budget: 49% of K3 attempts never submit.

## Decisions and recommendations for the distillation data
1. **Collect future DeepSeek batches without `--task-summary`.**
   - The no-summary traces teach the habit Qwen lacks: inspect the grids in code, then reason.
   - Qwen does not depend on a host table at inference and does not inherit its framing errors.
   - Cost: about +9% tokens. Keep `--helpers` and `--train-check`.
   - Before scaling up, a paired check (V3 with vs without the summary) on the hard training subset (~150–200 pairs, 2 samples per arm) costs about $15–20 and touches no eval data.
2. **E025 was collected with the summary;** the traces quote it about 23 times per attempt. A Qwen model trained on them must get the summary at inference too. Option: compare a summary-on SFT model (E025) with one trained on a no-summary batch.
3. **Filters:**
   - keep rollouts whose answer comes from a train-passing program (94% of correct V3 attempts have one);
   - prefer a first tool call at ≤5k tokens;
   - down-weight turns of 16k+ tokens;
   - do not imitate hand-built or hand-patched answer grids, "noise repair" edits, or hard-coded coordinates.
4. **Format conversion for Qwen:**
   - `submit_answer(expr)` becomes a self-contained `submit_program(solve)`;
   - API JSON tool calls are re-rendered into Qwen's XML tool format;
   - the budget-note text becomes the Qwen harness text.
5. **Run Qwen at the medium tier (or no tier line) for SFT and inference, not xhigh.** About 75% of correct DeepSeek traces fit a 48k cap in Qwen's tokenizer.
6. **Input format:** both harnesses use JSON grids (ARC Prize prompt) in the first message. Digit rows with dimensions would take 64% fewer prompt tokens (E018); the accuracy effect is untested.
7. **Hard data is scarce.** Over-sample the kind of pair DeepSeek solves with a relational rule, and grow the hard pool (ARC-2-only seeds, generated or composed tasks), since there are only 233 real ARC-2 tasks outside eval.

## Update 2026-10-18: Flash-1017 and the minimal effort header (E029, E030)
- **The teacher changed.** Since 2026-10-17 `deepseek-flash` serves V4.1-Flash-1017; E025 is from the old model. Label the model version in any dataset that mixes the two.
- **1017 is better and shorter:** sub80s40 gets 82.4% pass@2 (+4.9 pp) at 31.7k tokens (−22%).
- **`--effort-header minimal`** is a "Reasoning Effort: Minimal …" system message, in the same style as DeepSeek's own effort prefixes. It shortens traces without hurting hard pairs:
  - sub80s40 + header: 84.9% pass@2 at 27.4k tokens;
  - s20x60 + header: 80.3% pass@2 at 23.3k tokens (−27%), still in the noise band.
- **Share of correct traces under 40k output tokens:** sub80s40 80%, + header 86%, s20x60 91%, s20x60 + header 93%. Shorter teacher traces suit Qwen's 48k Kaggle budget.
- **For SFT:** the header is a system message, so either drop it from the SFT input (Qwen learns the terse style as its default) or keep it at Qwen inference. These runs used `lean2sub` without `--helpers --train-check`; the next batch should combine the header with those flags.
