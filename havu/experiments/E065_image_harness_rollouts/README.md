# E065: training-pool rollouts with the E058 harness (images + text twin)

2026-10-28. User request: collect the 10K-pool traces again with the best harness (E058 = E046 best + up-front images)
for pruning calibration and SFT of the pruned model, NVARC first (Flash-1017 was trained on the ARC training set:
arc2_train pass@1 ~97% with short reasoning); then a text-only twin of the same size with the Kaggle-tested E046 best
harness, so pruning can be done per harness and for a friend's SFT.

## Setup
- DeepSeek API, `deepseek-flash` (V4.1-Flash-1017), 1 attempt per pair, off-peak only (`--offpeak-only 20`: no attempt
  starts in a weekday peak window or within 20 min of one).
- Harness (both sets): E046 best = lean2sub, answer-tool, final-mode session, budget-notes 0.01, budget-warn 0.75,
  lenient-parse, budget-soft 40000, effort-header minimal, max-tokens 64000, max-total-output 80000, max-turns 40,
  multi-test + multi-budget 0.5, fallback-solve, train-check, turns-note 5. The image set adds
  `--grid-format text+image --image-cell 0` (one DeepSeek-aligned 42-px-tile image per grid in the first message).
- Pairs: the E031 pairs (arc2_train 1,070 pairs / 994 tasks, nvarc_b0 1,430) and the first 3,570 of E036's
  parent-balanced nvarc_b1 build order (`pairs_*.txt`). 6,070 pairs per set; 5,000 NVARC.
- Launchers: `launch.sh BATCH [PAIRS]` (images, 64 workers), `launch_text.sh` (text, 32 workers), `chain.sh`,
  `chain_text.sh`; re-running resumes and retries API errors. Analysis: `scripts/analyze.py`.
- Output: `evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_{e065,e065t}_{arc2_train,nvarc_b0,nvarc_b1}.jsonl`;
  full message traces at each record's `trace_path` (images inline as base64 `image_url` parts, byte-reproducible).
  NVARC parent = `task_id` prefix (manifest in `experiments/E025_train_rollouts/data/`).

## Results (after retrying connection errors: image 0 errors, text 1 error left; paired over pairs all three sets have)

| batch | set | pass@1 | train_pass | P(c given tp) | correct & tp | +/- vs image | out tok / task mean, p50, p90 | $ / task |
|---|---|---|---|---|---|---|---|---|
| arc2_train (1,070) | image | 96.9% | 94.6% | 97.7% | 989 | | 6.3k, 2.9k, 15.3k | 0.0054 |
| | text | 96.4% | 96.1% | 97.2% | 999 | +18/−23 | 6.6k, 2.9k, 15.4k | 0.0046 |
| | old (E031) | 95.8% | 93.5% | 97.3% | 973 | +17/−29 | 5.9k, 2.9k, 14.9k | 0.0046 |
| nvarc_b0 (1,430) | image | 73.1% | 85.4% | 79.4% | 970 | | 18.1k, 10.1k, 49.7k | 0.0137 |
| | text | 71.9% | 86.6% | 78.9% | 978 | +75/−92 | 19.0k, 10.1k, 51.1k | 0.0129 |
| | old (E031) | 70.3% | 81.3% | 80.3% | 934 | +72/−111 | 15.6k, 9.8k, 37.1k | 0.0115 |
| nvarc_b1 (3,568) | image | 71.5% | 85.2% | 78.5% | 2,388 | | 18.5k, 9.7k, 51.6k | 0.0141 |
| | text | 71.5% | 84.9% | 79.1% | 2,398 | +186/−186 | 19.0k, 10.3k, 53.3k | 0.0130 |
| | old (E036) | 70.1% | 80.7% | 80.4% | 2,315 | +167/−218 | 15.9k, 10.2k, 39.1k | 0.0118 |

- On NVARC, images and text tie (5,000 pairs: 71.9% vs 71.6%), unlike the eval set (+3.0-3.4 pass@1, E064). The image
  set uses 3-5% fewer output tokens but costs ~8% more per task (image input tokens).
- Both new sets beat the old E031/E036 traces (s20x60 + header) by ~1.4-2.8 pp pass@1 and +4-5 pp train_pass, with
  longer tails (p90 50-53k vs 37-39k: the larger 80k output budget).
- Volume (dsv4-pruning count, image set): ~96.6M generated tokens, ~180M full-sequence replay tokens, 44.4K images
  (~18M image tokens). Text set: 101.6M generated. Both exceed REAP's 10-20M; pruning uses them as is (keeps failed
  attempts, drops error rows; image and text sets kept separate).

## Cost
Recorded $75.86 (image, incl. the 150-pair pilot and retries) + $69.76 (text) = $145.6, DeepSeek API off-peak.

## On Hugging Face
Private dataset `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e065`, configs `e065_images_{arc2_train,nvarc_b0,nvarc_b1}`
and `e065_text_*` (built by `scripts/hf_build.py`, secret-scanned). Briefly uploaded to the E031-E036 dataset first,
then moved (removed there; the files remain only in that repo's history).
