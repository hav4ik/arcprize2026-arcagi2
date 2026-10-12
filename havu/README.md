# havu/ — Vu's ARC Prize 2026 (ARC-AGI-2) workspace

Snapshot of my working directory: experiments, eval harnesses, Kaggle submission code, notes and run results.
Start with `AGENTS.md` (how the workspace is organised), then `experiments/experiments.tsv` (every experiment, status and
results), `experiments/results_summary.md` (all runs in one table) and `kaggle/README.md` (the offline Kaggle pipeline).

## Layout
- `experiments/` — one `E###_name/README.md` per experiment + the registry `experiments.tsv`.
- `evals/` — eval harnesses (DeepSeek no-tool / tool-use, Qwen generation API, Qwen tool agent) and `evals/results/`
  (raw results + gzipped reasoning traces; useful as SFT data candidates).
- `kaggle/` — the offline Kaggle submission (notebook builder, solver, tests, provenance checks, run outputs, reviews).
- `notes/`, `prior_works/` — research notes and summaries of public ARC methods. `data_tools/` — contamination checks.

## Not included (re-create locally)
- `SECRETS.yaml` (API keys: `kaggle_api_token`, `deepseek_api_key`, `openrouter_api_key`, `vast_ai_api_key`) and
  `ENDPOINTS.yaml` (self-hosted servers: `name`, `token`, `addr`) — create your own; never commit them.
- `data/` (17 GB): the Kaggle competition data (`kagglehub.competition_download('arc-prize-2026-arc-agi-2', output_dir='data')`),
  the ARC-AGI-2 GitHub repo (`data/arc-agi-2-github/`) and the NVARC datasets — see `AGENTS.md` "Data".
- `.venv/` — Python env managed with uv (`uv venv .venv && uv pip install --python .venv ...`).

Paths in the docs are relative to this folder (`havu/`).
