# Greenblatt: Draw More Samples (GPT-4o program sampling)

- **Author:** Ryan Greenblatt (Redwood Research). Leaderboard date 17 Jun 2024.
- **Blog:** https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt
- **Code:** https://github.com/rgreenblatt/arc_draw_more_samples_pub, in `arc_solve/solve.py` and `prompting.py`, read at `dc4b608`.
- **License:** MIT.

Source tags: [B] = blog, [C] = code, [L] = ARC Prize.

## TL;DR
- **Method:** GPT-4o samples about 8k Python `transform` programs per task: about 5k initial and 3k revisions. Programs correct on all train pairs vote, and the top 3 outputs are submitted. [B,C]
- **Results:** 50% on a 100-task public-eval subset with top-3 (48% with top-2). [B] 43% on the ARC-AGI-1 semi-private set, reported at k=2,048. [L]
- **Cost:** the leaderboard lists $40k. The blog gives no figure, so this is likely the total.

## Method
- **Prompt** [C]:
  - *"determine the transformation rule and implement it in code… Start your response by carefully reasoning in <reasoning></reasoning> tags."*
  - About 30k tokens of few-shot examples plus images.
  - Same-size tasks also get spreadsheet-cell, connected-component and diff ASCII views. [B]
- **Sampling** [C]: 4 prompt variants at temperature 1 (1024/1024/1024/2048 samples).
- **Revision** [B,C]:
  - Runs only on tasks with ≤32 train-correct programs.
  - Picks 12 parents ranked by their Hamming distance to the outputs.
  - Gives each successive parent 0.75× the samples of the previous one, about 3k samples in total.
  - Prompt: *"The `transform` function you implemented failed on at least one of the examples… The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule."* It includes a diff and few-shot examples, for one round only.
- **Selection** [B]: a majority vote over train-correct programs. Outputs equal to the test input are rejected.

## Results & ablations [B,C]
| Configuration | Test accuracy |
|---|---|
| V0 at 1024 samples | 25% |
| V2 at 2048 samples | 34% |
| All variants | 37% |
| All variants + revision | **50%** |

- Accuracy rises about 3% per doubling of samples.
- Revision fixes about 20% of the remaining failures.

## Reusable for us
- **Not for DeepSeek Flash:** 8k calls × $0.035 is about $280/task.
- **What carries over:** filtering by execution on the train pairs plus voting, revision focused on the closest parents, and log-sample scaling curves for budgeting.
- **Kaggle:** this regime suits short, non-thinking code samples from a small local model, roughly 500–1,000 samples per task.
- **Distillation:** verified programs make good SFT data.

## Open questions
- What configuration the semi-private run used, including whether revision was on.
- The cost per task.
