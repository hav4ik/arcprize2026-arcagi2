# ARC-AGI-2 submissions tracker (dashboard)

## Current local monitor (2026-10-31)

Open **http://localhost:8765/local.html** (LAN: http://10.0.0.185:8765/local.html).
The local page now uses `monitor/server.py`; it does not use the legacy scratchpad builder below.
Reload an already-open legacy page once to load the replacement.

- The backend discovers the account's L4 Runpod instances and SSH endpoints, then polls them read-only every **60 seconds**.
  It detects notebook/solver/server processes, current queue jobs, GPU usage and timestamped telemetry; new E072 and E073 runs
  appear automatically. It never starts, stops or alters an experiment or pod.
- The browser checks a **single atomic snapshot** every **10 seconds**; **Fetch now** requests a background collection.
  Every chart, count and table in a generation uses the same snapshot. Failed requests retain the previous data and show an error.
- Use the table's **Compare** checkboxes to overlay multiple runs on all four charts. The chart run list also includes archived
  experiments; **+ Manh V12** adds the gold replica. **Live runs**, **Only selected run** and **Clear** are shortcuts. Colors match
  the legend across every chart; selections survive automatic refresh and page reload. Inspecting another row keeps an existing
  multi-run comparison. Mixed workloads and archived time origins are labeled explicitly; missing samples are never zero-filled.
- Status requires process evidence. A missing pod or observation older than three minutes is unverified, not running. An alive
  solver with sampler data older than five minutes is marked telemetry-stale. A lingering sampler cannot make a finished run live.
- UTC timestamps come from Unix timestamps; time-of-day strings are never assigned a guessed date. Snapshot time, pod observation
  and telemetry time are shown separately. Missing telemetry stays missing, rather than turning into a measured zero.
- Workloads are checked against the actual Kaggle-version challenge grids. Each row shows its own task/pair count. Unknown or
  mismatched grids suppress the score. Pass@2 uses at most two distinct training-ranked answers, including three-draw research
  screens. Unanswered tasks count as zero; a finished attempt without a grid still counts as a finished attempt.
- The V12 reference on a verified run's matching tasks is an **endpoint reference**, not an assertion that different workload
  sizes have comparable progress curves or pod speed. E073 holdout accuracy is hidden until the validation run finishes.
- `events.json` and `findings.json` remain the canonical LLM-curated comments. They are displayed verbatim and are **never written
  by the collector**. Optional `comments.json` accepts an array of `{id, author, utc, run, text}` comments. Changes appear at the
  next collection. Original HTML and data were preserved in `archive/20261031T072441Z/`, linked from the new page.
- Secrets are read from the root `SECRETS.yaml` at runtime, never placed in commands, served files or logs. Known credential
  values are redacted from observed log tails. The HTTP server only serves its explicit assets/API and the frozen archive;
  it does not expose the workspace or a directory listing.

Start: `bash kaggle/dashboard/submissions_tracker/monitor/start.sh`. The process runs without agent turns;
PID and log are in `monitor/state/`. Snapshot and pod caches survive a server restart. Only run one production instance on 8765.
The previous static HTTP server and scratchpad refresher were stopped by their specific identities when this service replaced them.

API: `GET /api/snapshot`, `GET /api/health`, `POST /api/refresh` (same origin).
Check: `.venv/bin/python -W ignore::ResourceWarning -m unittest discover -s kaggle/dashboard/submissions_tracker/monitor/tests -v`.
Development preview: `.venv/bin/python kaggle/dashboard/submissions_tracker/monitor/server.py --port 8766 --bind 127.0.0.1 --state-dir /tmp/arc-monitor-preview`.
No Python package or frontend CDN dependency was added. Chromium desktop/mobile rendering and refresh are tested against the live API.

Validation on 2026-10-31: **20 tests passed**; Chromium at 1440 px and 390 px rendered all four charts with no page overflow
or console errors. A real Fetch-now cycle advanced the browser snapshot automatically and preserved run selection and the
comment tab. All **22 timeline comments and 16 findings** matched their original JSON values exactly.
The multi-run browser check is `node kaggle/dashboard/submissions_tracker/monitor/tests/compare_browser.mjs`, with a dedicated
Chromium profile exposing its debugging endpoint on 9235. It covers table checkboxes, distinct chart colors, archived V12,
mixed-workload labeling, light/dark mode, mobile layout, persistence and automatic refresh.

## Preserved artifact implementation

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

## Automatic refresh (no agent)

- `tools/auto_dash.sh`: every 5 min (`INTERVAL` s) runs `update_dash.py` into `DASH_DIR` (the live scratchpad copy), copies the JSONs here, and refreshes `publish_queue.json` (datasets whose sha256 differs from the last publish). Start: `setsid nohup kaggle/dashboard/submissions_tracker/tools/auto_dash.sh >/dev/null 2>&1 &`. PID in `auto_dash.pid`; stop with `touch auto_dash.stop`; log in `auto_dash.log`.
- New live runs: `tools/pod4.py:discover()` picks up any `phase_h/<dir>/data/status.json` newer than 3 h that is not a known/frozen dir. Label = status.json `run`/`label`/`name`, else the dir name (pod = status `pod`). To pin a label, add the dir to `CANDS`.
- Publishing (needs Claude's Artifact tools): `python3 tools/publish_list.py` lists changed dataset ids and files; after publishing, `python3 tools/publish_list.py --mark`.
- `local.html`: screenshot page. `cd kaggle/dashboard/submissions_tracker && python3 -m http.server 8765`, open http://localhost:8765/local.html. It loads `index.html` at runtime (style, markup, script) and runs it on a small `dash` shim, so it stays in sync; refreshes every 60 s.
