# ARC-AGI-2 submissions tracker (dashboard)

The live dashboard for Kaggle submissions and their 12 h pod replicas: https://claude.ai/artifact/RvbHR1L9axaNcDi73nSzbM (link-shared on purpose by the user). This folder is its source of truth in git: the page, the data snapshots it renders, and the builder that turns pod data into those snapshots.

## What's here

- `index.html`: the page. It is a Claude "Dashboard" artifact page and runs on that type's runtime (`dash.data`, `dash.draw`, `dash.calc`), so it doesn't render as a plain local file.
- `*.json`: dataset snapshots the page reads.

  | file | what |
  |---|---|
  | `runs.json` | one row per replica |
  | `mem.json` | host RAM, GPU free, cap − peak, cap − reserved over time |
  | `tps.json` | throughput and running requests |
  | `acc.json` | score so far and tasks answered |
  | `outages.json` | OOM hangs, deaths, restarts |
  | `events.json` | the curated timeline |
  | `findings.json` | audit and test findings |
  | `submissions.json` | LB vs replica per submission |
  | `ref_b32.json` | the 10-27 B32 replica as a reference line |

- `tools/`: the builder.
  - `update_dash.py` rebuilds `mem/tps/runs/acc/outages.json` from each pod's synced files: `gpumem.log` (vLLM's per-rank "GPU memory:" and "Capped the PyTorch allocator" lines plus the stats lines), `ram.tsv`, `tps.tsv`, `status.json` and `attempts.jsonl`, under `experiments/E062_manh_vllm_fork/per_layer/*/data/`.
  - Helpers: `gpumem_series.py`, `tps_series.py`, `acc_series.py` (ARC task score over all 240 tasks, best of the finished attempts, unanswered = 0), `build_mem.py`, `pod4.py`.
  - Run it with `DASH_DIR=<folder>` (default: this folder): `python3 tools/update_dash.py`.

## How it was kept current (2026-10-30)

- **Pod agents** synced their data files every 5 min.
- **A refresher agent** ran `update_dash.py` every 5 min, uploaded changed JSON files as artifact assets, and pointed each `datasets/<id>` doc at the new blob.
- **The coordinator** curated `events.json`, `findings.json` and the page.
- **Measurement notes** (also on the page):
  - Kaggle-equivalent RAM = the container's cgroup memory + 7 GiB (pods' `/proc/meminfo` is host-wide).
  - The real GPU margin is cap − reserved.
  - Outages are drawn from per-minute metrics (0 while the server hangs or restarts).

## The story it tells (2026-10-30)

- **B32-M (LB 46.5):** with Kaggle-matched GPU headroom, the replica hit 4 CUDA OOMs, ran out of restarts at 7.26 h and finished at **50.4**. See `notes/2026-10-30_b32m_oom_modes.md`.
- **Manh's V12:** 0 events, a flat 1.08 GiB margin, and 80.6 so far at 9 h.
- **The 10-27 B32 replica** finished at 79.1.
