Snapshot: October 31, 2026, 6:59 PM UTC-07:00. Kaggle ARC-AGI-2 evaluation.

Pass@1 is task-weighted and averaged across both draws. Token and tool-call means are per logical task attempt; multi-test pair records are counted once.

**Image + tools — finished: 120 tasks × 2 draws per model**

| Model | Pass@1 | Pass@2 | Tokens mean | Tokens p50 | Tokens p90 | Tool calls mean (total) | Tool errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| E128 (~16B) | 72.43% | 83.47% | 38,905 | 30,435 | 80,523 | 17.85 (4,284) | 12.72% |
| E074 (~14.58B) | 70.83% | 81.81% | 39,483 | 31,028 | 85,411 | 17.76 (4,263) | 13.02% |
| E076 (~14.02B) | 71.67% | 83.89% | 40,717 | 30,935 | 85,653 | 18.06 (4,334) | 13.54% |

**Text only + tools — finished: 120 tasks × 2 draws per model**

E128 (~16B): 240/240; E074 (~14.58B): skipped by user; E076 (~14.02B): 240/240.

All rows use the same 120 tasks with both draws complete (240 attempts/model). All120 tasks complete in each available arm; skipped or queued references have no comparison.

| Model | Pass@1 | Pass@2 | Tokens mean | Tokens p50 | Tokens p90 | Tool calls mean (total) | Tool errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| E128 (~16B) | 72.29% | 81.81% | 41,251 | 31,454 | 86,629 | 18.90 (4,535) | 10.17% |
| E074 (~14.58B) — skipped by user | — | — | — | — | — | — | — |
| E076 (~14.02B) | 69.24% | 79.44% | 42,380 | 31,946 | 94,318 | 18.37 (4,408) | 12.34% |

E128 (~16B) matched-subset cap stops: 10; attempts missing any answer: 8.

E076 (~14.02B) matched-subset cap stops: 12; attempts missing any answer: 11.

Tool errors are failed/rejected calls after auto-repair, divided by all tool calls; infrastructure errors are separate. Token quantiles use linear interpolation.

Text generation throughput, matching elapsed minutes [5, 155]: E128: 1,061.21 tok/s; 14.02B: 1,088.87 tok/s. Matching elapsed windows include unfinished-attempt tokens and evolving batches; not isolated-kernel speed. Skipped or queued references have no throughput comparison.

Image equal-window minutes5–150: 14.02B 1,083.09 vs E128 1,003.18 / 14.58B 1,052.72 tok/s (+7.97%/+2.89%).

[Exact snapshot, task IDs and provenance](completed_tools_report.json).
