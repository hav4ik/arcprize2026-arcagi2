# AGENTS.md

Guidance for coding agents working in this repository (ARC-AGI-2 / ARC Prize 2026).

## Secrets

- **`SECRETS.yaml`** (project root) is the source of all API keys and tokens. Current keys:
  - `kaggle_api_token`: Kaggle API token
  - `deepseek_api_key`: DeepSeek API key
  - `openrouter_api_key`: OpenRouter API key
  - `vast_ai_api_key`: Vast.ai API key. The `vastai` CLI is in `.venv`. Pass the key per call with `--api-key "$K"` rather than `vastai set api-key`, which would store a copy in `~/.config`.
- **`ENDPOINTS.yaml`** (project root) lists self-hosted model servers (`name`, `token`, `addr`), for example Vast.ai SGLang boxes serving an OpenAI-compatible API at `http://<addr>/v1`. Treat `token` like a secret.
- Read values from this file at runtime, for example by exporting them as environment variables. Never print, log, or commit them, and never copy them into other files.
- `~/.kaggle/kaggle.json` contains an outdated key that Kaggle rejects. Use the token from `SECRETS.yaml` instead:

  ```bash
  export KAGGLE_API_TOKEN="$(sed -n 's/^kaggle_api_token:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
  ```

## Environment

- Python venv is at `.venv`, managed with [uv](https://docs.astral.sh/uv/).
  - Install packages with `uv pip install --python .venv <pkg>`.
  - Run scripts with `.venv/bin/python <script>`.

## Data

- `data/` holds the Kaggle competition `arc-prize-2026-arc-agi-2`, downloaded with `kagglehub.competition_download(..., output_dir='data')`:
  - `arc-agi_training_{challenges,solutions}.json`: 1000 tasks
  - `arc-agi_evaluation_{challenges,solutions}.json`: 120 tasks
  - `arc-agi_test_challenges.json`: 240 tasks
  - `sample_submission.json`
- `data/nvarc/`: NVARC (ARC Prize 2025 winner, https://github.com/1ytic/NVARC) Kaggle datasets:
  - `nvarc-synthetic-puzzles/<subset>/<parent_task_id>/<parent>_<hash>.json`: a list of `{input, output}` pairs (about 30 per puzzle). The parent ID is the ARC-AGI-2 task the puzzle was derived from.
    - `nvarc_training`: 47,337 puzzles from 716 **training** parents.
    - `nvarc_full`: 55,886 puzzles from the 120 **evaluation** parents.
  - `nvarc-augmented-puzzles/<subset>/`: HF `datasets.load_from_disk` folders with columns `puzzle_name` and `messages`. `messages` is a list of user/assistant turns; each grid is digit rows joined by `\n`. Subsets and sample counts:
    - `nvarc_full` 1.79M, `nvarc_training` 1.13M, `arc2_training` 156k, `rearc` 102k, `concept` 41k, `mini` 38k, `arc2_evaluation6` 720.
  - The NVARC artifacts dataset (about 47 GB of intermediate text and code) was deliberately not downloaded.
- **Eval contamination:** `nvarc_full` and `arc2_evaluation6` are derived from the 120 ARC-AGI-2 evaluation tasks. Exclude them when using `arc-agi_evaluation_*` as a local validation set. Dedup any external data against the eval tasks by grid hash.
- `arc-prize-2026-arc-agi-3` requires accepting the competition rules on Kaggle first. Until then, downloads return 401.

## Research and evals

- `experiments/`: experiment registry `experiments.tsv` (id, path, status, results, cost, notes) and one `E###_name/README.md` per experiment. Update the TSV row when status or results change.
- `prior_works/`: links and summaries of public ARC methods and datasets. `README.md` is the index. Add a summary file per method and keep the index tables up to date.
- `notes/`: dated research notes; `notes/prompts/` holds ARC Prize's official eval prompt and DeepSeek's reasoning-effort prefixes.
- `evals/run_deepseek_arc2.py`: DeepSeek API eval on the 120 ARC-AGI-2 eval problems, using NVIDIA's prompts. Results go to `evals/results/*.jsonl`, and re-running resumes.
  - Thinking mode is on by default; set the effort with `reasoning_effort` (low/high/max).
  - Peak pricing (2× off-peak) runs 01:00–04:00 and 06:00–10:00 UTC on weekdays. Chinese public holidays are off-peak all day.
  - **Estimate cost before launching**: calls × output tokens/call × list price. Flash is $0.60/M output tokens off-peak; Pro is $1.98/M. ARC eval calls average ~53–81k output tokens, so ~$0.035–0.05 per Flash call. Never infer the price from balance deltas: the balance lags. Choose concurrency at launch, because restarting re-bills in-flight calls. Get approval before runs over ~$10. Full accounting: `experiments/E001_deepseek_baselines/README.md`.
- Tool-use analysis scripts (E004):
  - `evals/compare_runs.py`: tool runs against E001d on the same pairs;
  - `evals/sample_stats.py`: per-sample pass@1 mean/sd, pass@2 and majority vote;
  - `evals/candidate_pass1.py`: single-answer ensembles;
  - `evals/balance_log.py`: DeepSeek balance against recorded cost estimates.
- **Sandbox cleanup:** `start.sh` enables the server's idle-session reaper (`SANDBOX_SESSION_TIMEOUT`, default 1800 s; the longest recorded attempt took 15.4 min), so finished sessions are stopped after 30 min idle. The container is capped at 40 GiB. Restarting it between big runs is still cheap insurance.
- `evals/report.py`: one-table summary of every run in `evals/results/` (calls, errors, per-call accuracy, tokens, ~$).
- `evals/sandbox_docker/start.sh`: code-execution sandbox built from the AIMO-proof-pilot NeMo-Skills image. It runs on an internal Docker network (no internet) with no host directories mounted, plus a patched `main.py` so timeouts keep session state. It prints the server URL.
- `evals/run_deepseek_tools.py`: tool-use harness (E004) on top of that sandbox, using the ARC Prize prompt and scoring. Modes are `grid` (Python tool, answer as a grid) and `code` (submit a `transform()` checked on the train pairs). Use `--dry-run` for a cost estimate; it refuses to launch above $10 without `--allow-expensive-run`. Tests: `ARC_SANDBOX_URL=<url> .venv/bin/python -m unittest discover -s evals/tests`. Run plan: `experiments/E004_tool_use/README.md`.
- **Token-efficient tool harness (E009–E011, harness `persistent-v4`):**
  - Options of `evals/run_deepseek_tools.py` are opt-in; the defaults reproduce `persistent-v3`.
    - `--answer-tool`: `submit_answer(expr)` reads the grid from the session.
    - `--final-mode session`: no forced final turn; falls back to the session's `answer`.
    - `--budget-notes`, `--budget-soft`, `--soft-wording`: token-budget notes in tool output.
    - `--lenient-parse`, `--verify-submit`, `--think-turns`.
    - `--max-balance-stop` (default $20): the kill-guard.
  - Prompts: `lean2sub`, `lean2subv`.
  - Recommended configs and commands: `experiments/E011_pareto_tool_harness/README.md`.
  - `evals/paired_vs_baseline.py`: runs vs E004-GL on the same pairs, with bootstrap CIs, pass@2 and the long tail.
  - `evals/frontier.py`: accuracy/token frontier and mixed-config pass@2.
  - Other agents may share the sandbox container: check `docker stats` before restarting it.
- **Self-hosted serving for tool use (rule):** agentic, multi-turn tool use re-sends the whole growing transcript on every turn (typically 20–80k tokens, ~15 turns per attempt). So:
  - **Prefix caching is mandatory.** Use vLLM `--enable-prefix-caching` or SGLang's radix cache. Never benchmark, tune or submit a tool-use configuration with it off: throughput or KV numbers measured with it off do not transfer.
  - **Choose the continuous-batching size (concurrency) by cache health, not just KV capacity.**
    - Too many concurrent attempts evict each other's cached prefixes, and every turn then re-prefills from scratch.
    - When sizing concurrency, measure the prefix-cache hit rate and evictions/preemptions on a realistic multi-turn replay (E014).
    - Keep the hit rate healthy, high and stable over the run. If it drops or evictions grow, lower concurrency or add KV offload to host memory.
  - **Log cache metrics during real runs:** the prefix-cache hit rate, KV usage and preemptions/retractions.
  - Current setup and evidence: `experiments/E013_kaggle_submission/README.md` (vLLM 0.23, prefix caching on, KV 1.40M tokens, concurrency 35) and `experiments/E014_sglang_kaggle/README.md`.
- `data_tools/contamination_check.py`: checks training data against ARC-AGI-2 eval grids, invariant to rotation, flip and colour relabelling. Reports go to `data_tools/contamination_report.json`.
