# E002: Imbue-style program evolution on DeepSeek, then distill into a small model

Status: **planned**. Start once the E001 baselines are final.

## Hypothesis
1. Imbue's per-task evolution over Python `transform(grid)` programs lifted Gemini 3 Flash from 34% to 61.4% and Kimi K2.5 from 12.1% to 34.0% (public eval, pass@2). It should lift DeepSeek V4.1-Flash well above its plain-prompt baseline (E001a) at low cost.
2. The search traces can be distilled into a small model that fits Kaggle's offline budget (4×L4, 12h). Traces here means verified programs, explanations, and repair steps (parent + feedback → better child).

## Background
- Method summary: `prior_works/imbue_arc_agi_2_evolution.md`. The code is at https://github.com/imbue-ai/darwinian_evolver (AGPL-3.0); reimplement it rather than vendoring it.
- Imbue's evolution cost about 9–10× a single base call. For Flash that suggests roughly $0.2–0.4 per task; this is an extrapolation, not measured.

## Plan
1. **Reimplement the loop.**
   - Population of programs, starting from the identity function.
   - Mutation prompt with cell-level diffs on the train pairs.
   - 3-parent crossover, run 25% of the time.
   - Fitness = train correctness + an LLM-judged transfer score + simplicity.
   - Fix the code-vs-blog discrepancies noted in the summary.
2. **Score on the same 120 problems as E001**, then on a clean holdout. Track cost per task and the score at each iteration.
3. **Log every generation.** Keep verified program/explanation pairs, repair pairs, and transfer-judge labels.
4. **Build distillation data.**
   - Run the loop on training sources only (ARC-AGI-2 train, NVARC `nvarc_training`, ARC-GEN). Never on eval-derived data.
   - Turn the logs into SFT data for a small model (e.g. Qwen3-4B-class): task → program, and program + feedback → better program.
5. **Evaluate the distilled model offline**, both with and without a small local evolution loop.

## Open questions
- Should the evolved programs be fed to the small model, or should it learn transductive answers?
- Is an execution sandbox allowed in the Kaggle notebook? (Plain Python execution should be fine.)
- How does the transfer-judge threshold need recalibrating for DeepSeek? Imbue used 0.95 for Gemini/Kimi and 0.35 for Opus.
