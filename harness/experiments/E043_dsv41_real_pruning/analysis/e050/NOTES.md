# E050 token buckets: effort 25 vs 50 on E128 (2026-10-26, for dsv4-infer)

`buckets.py` runs the `../doubt` split on E050's traces. The runs are in `experiments/E050_effort_ab/runs/{eff25,eff50}`: 120 tasks, one multi-test rollout per task, V6 pipeline. Correctness comes from E050's `analyze.load_arm`. Output is `buckets_out.txt` and `results.json`.

- Per-turn token sums equal the logged usage exactly.
- The all-train-pass detector agrees with the harness `train_pass` on 86% of attempts in both arms.

## Buckets (mean output tokens per task; diff is 25 − 50, with a 95% bootstrap CI over tasks)

| bucket | effort 25 | effort 50 | diff [95% CI] | share at 50 |
|---|---:|---:|---:|---:|
| total | 43.4K | 42.2K | +1.2K [−2.0, +4.6] | |
| (a) error recovery | 2.0K | 2.4K | −0.4K [−1.1, +0.4] | 5.6% |
| (b) post-pass | 4.1K | 5.9K | −1.9K [−4.3, +0.5] | 14.1% |
| (c) pre-pass, attempts that pass | 19.8K | 18.2K | +1.6K [−2.8, +6.1] | 43% |
| (c') attempts that never pass | 17.5K | 15.7K | +1.8K [−3.2, +7.0] | 39.4% |

- **Effort 25 shifts work between buckets instead of removing it.**
  - Each turn is shorter: turn 1 is 413 vs 543 tokens, later turns 2.30K vs 2.59K (−11%).
  - It takes 1.1 more turns.
  - On the 82 tasks that pass in both arms, it reaches the first pass later: +1.0 turn and +2.9K tokens [−0.7, +6.6]. It then stops sooner: post-pass −2.2K [−4.7, +0.2] and −0.6 turns.
  - The tool loop conserves the work: less thinking per turn is paid back in more turns.
- **No change in style:**
  - thinking share 0.85 vs 0.86;
  - doubt markers 9.2 vs 9.7 per 1K thinking words;
  - giving-up phrases 0.3 per attempt in both;
  - Python errors 3.0 vs 2.9 per attempt;
  - never-pass attempts 29 vs 27, at 75K vs 74K tokens each.
- **Post-pass is concentrated in a tail.** At effort 50, 81% of post-pass tokens come from the 20 attempts with more than 10K post-pass tokens.
  - Those attempts score 0.75, vs 0.90 for attempts with tails of 2K or less.
  - This is the earlier E043 finding: long post-pass work is rework when the first passing rule does not fit the test input, and it is partly productive.

## What this means for the token goal

The effort setting is not a token lever for E128 in this harness, because the minimal-effort header and the tool loop already set the pace. The remaining levers are structural, and each needs an A/B before adoption:

| lever | share of tokens | estimate |
|---|---|---|
| a harness note once the train pairs pass ("check test assumptions with code, then submit"); see `../doubt/REPORT.md`, row 3 | post-pass 14% | about −3 to −5% of total tokens if it removes a third of post-pass; the risk is cutting productive rework in the >10K tail |
| never-pass attempts | 39% | stop rules were neutral to negative in E042's simulator (`goal75/giveup2.py`); the attempts still score 0.38–0.41 because the detector misses some passes |
| auto-repair (`evals/auto_repair.py`) | error recovery 5.6% | E128 only has about 3 errors per attempt |
