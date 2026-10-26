# Goal (user, 2026-10-26 ~13:45 UTC): a model/harness projected at >= 75 on Kaggle within 24 h

`grid.py` (multi_sim, E128 run-3 multi records, 240 tasks, 60 reps, KV 1.556M / 0.83):

| config | projected | no attempt 0 | attempt 1 % |
|---|---|---|---|
| V6 now (s 1.04, x36, gate 0.6, thrash 0.88:0.6) | 70.3 | 6.6 | 2.2 |
| ceiling: every job exactly 1 attempt | 71.7 | 0 | 0 |
| no thrash, no gate, x44 | 73.5 | 0.1 | 12.6 |
| tokens x0.85, V6 gate | 73.9 | 0 | 15.9 |
| **no thrash, no gate, x44, tokens x0.85** | **77.2** | 0 | 32.6 |
| no thrash, no gate, x44, throughput +15% | 75.6 | 0 | 24.5 |
| ceiling: every job 2 attempts | 83.8 | 0 | 100 |

Levers and tests: (1) remove thrash -> retention-interval A/B on L4 (fix list S3b), then x44 ungated at Kaggle KV and an
>= 8 h stress test; (2) -15% tokens at equal pass@1 -> reasoning effort 25 vs 50 on both 4x4090 boxes (fix list I2);
(3) fix-list code for safety. KV offloading is not on the critical path (more workers alone: 73.5 -> 73.7 at x56).

## Give-up rule (dsv4-pruning, 2026-10-26): not adopted
Stop a per-pair attempt with no all-train-pass output by T tokens (answer assumed wrong; trace data from
experiments/E043_dsv41_real_pruning/analysis/giving_up/results.json; multi-test records untouched). `giveup2.py`:

| T | correct pairs lost | per-pair tokens saved | V6 now | no thrash x44 | no thrash x44 + tokens x0.85 |
|---|---|---|---|---|---|
| none | - | - | 70.3 | 73.5 | 77.2 |
| 70K | 2 | 2.6% | 70.1 | 73.0 | 76.8 |
| 60K | 7 | 6.5% | 69.9 | 72.8 | 77.3 |
| 50K | 22 | 12.7% | 69.7 | 72.1 | 76.2 |

Freed tokens buy second attempts of low marginal value while the cut attempts sometimes held a right answer: neutral
to slightly negative. Only a token cut at equal accuracy (E050) moves the projection.

## L2 result and re-plan (2026-10-26 16:40 UTC)
L2 (x44 ungated, interval 512, Kaggle KV) thrashed at 2.08 h / KV 0.89 (hits -> 0, 48-62 tok/s; `mt44b` at 128: 2.01 h /
KV 0.82). Thrash is capacity-driven, so "no thrash, no gate" is not reachable by the interval alone. With the
measured thrash severity (x0.3 throughput above KV 0.88), `kvcap.py` / `kvcap2.py` (E128 accuracy):

| effective KV | x36 g0.6 | x40 g0.65 | x44 g0.7 | x44 ungated | x56 ungated |
|---|---|---|---|---|---|
| x1.0 (today) | 69.2 | - | 65.1 | 40.4 | 37.5 |
| x1.14 (util ~0.96) | 71.6 | 71.4 | 70.2 | | |
| x1.24 (util ~0.97) | 72.0 | 72.5 | 71.9 | | |
| x1.24 + tokens x0.9 | 73.8 | 74.4 | 74.3 | | |
| x1.37 (E120-sized pool) | 72.5 | | 73.3 | 65.8 | 46.9 |
| x2 (KV offload) | 73.0 | | 74.1 | 74.1 | 74.6 |
| x3 (KV offload) | 73.0 | | 74.1 | 74.1 | 76.1 |

So >= 75 needs more effective KV (util 0.97, KV offload E051, or a smaller model) AND fewer tokens at equal accuracy.
