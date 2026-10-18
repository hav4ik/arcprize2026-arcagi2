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

## GitHub snapshot

- The workspace is mirrored into the private repo `hav4ik/arcprize2026-arcagi2` under `havu/`. The local clone is `/home/vu/Workspace/arcprize2026-arcagi2`.
- Sync with `data_tools/sync_github.sh`:
  - with no arguments it mirrors, stages and secret-scans (`data_tools/secret_scan.py`), then shows the diff;
  - with a message (`"havu: ..."`) it also commits and pushes to main.
- What it leaves out: `data/`, `.venv`, `SECRETS.yaml` and `ENDPOINTS.yaml` are never synced, and the repo's `.gitignore` skips `*.log`, `*.pid` and vendor copies.
- Commits are unsigned, as in the history, because GPG cannot prompt from agent shells. Push only when the user asks.
- **Distillation data (DeepSeek → Qwen SFT):**
  - The findings and decisions are in `notes/2026-10-17_distillation_data_findings.md`.
  - Collect new DeepSeek batches without `--task-summary`, but keep `--helpers --train-check` (E026: the summary saves only 9% of tokens and its framing misleads the model).
  - The task pool and launcher are in `experiments/E025_train_rollouts/` (`build_dataset.py --batch nvarc_bN`, `launch.sh`).
- E025 rollouts are also on Hugging Face as the private dataset `livobereshka/arc-agi2-deepseek-v3-rollouts-e025`. The org is `livobereshka`, spelled with an "s"; the token is `huggingface_api_key` in `SECRETS.yaml`.

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
- **E017 options (harness `persistent-v5`, opt-in; recorded in the config):**
  - `--task-summary`: a host-computed perception pack (`evals/task_summary.py`) appended to the first message, about 1.5k tokens on average and capped near 3k. It holds object tables, in→out matching and an "Outside train range" line for the test.
  - `--prompt lean2roles`: `lean2sub` plus a ROLES note, a ban on unproven "noise" labels, and a test-difference audit before submitting.
  - `--helpers`: preloads `evals/arc_helpers.py` in the session (`objects`, `d8`/`match_d8`, `panels`, `stamp`, `ray`, `show`, `check(solve)`) and documents them in the tool description.
  - `--train-check`: after each attempt, re-runs the answer program on the training inputs and records `train_pass` / `train_frac` / `train_trace` (`evals/train_check.py`). It uses no model tokens; the host compares hashes.
  - Not part of the config, so files resume without them:
    - `--extra-attempt-if-no-train-pass` (with `--attempts 2 --train-check`): runs attempt 2 for pairs where neither attempt 0 nor 1 passes the train check.
    - `--auto-peak-price`: prices each turn at 2× inside weekday peak windows.
  - `evals/train_check.py:ranked_answers` ranks attempts for Kaggle by train_pass, then attempt order, and keeps the top-2 distinct answers.
  - Phase-2 comparison against sub80s40: `experiments/E017_reasoning_efficiency/scripts/phase2_compare.py`. Results: `experiments/E017_reasoning_efficiency/README.md`.
  - `evals/frontier.py`: accuracy/token frontier and mixed-config pass@2.
  - Other agents may share the sandbox container: check `docker stats` before restarting it.
- **Qwen tool harness (`evals/run_qwen_tools.py`, `evals/arc_harness/agent_qwen.py`, harness `qwen-tools-4`):** the defaults behave as `qwen-tools-3` (E012).
  - `--answer-mode program` (E019): the only answer tool is `submit_program(code defining solve(grid))`. It is checked host-side in a fresh process on every train pair before the test; no typed or forced answers, and the fallback is the best submitted program or the session's `solve`. Kaggle: `ARC_TOOLS_ANSWER_MODE=program`.
  - Other options: `--record-train-check` (submit mode, bookkeeping only), `--base-url` (unauthenticated server such as local SGLang), `--context`, `--out`.
  - Tests: `evals/tests/test_qwen_tools.py`, `evals/tests/test_qwen_program.py`. Cache-health logger: `evals/sglang_cache_monitor.py`. Results: `experiments/E019_qwen_program_answers/README.md`.
- **Self-hosted serving for tool use (rule):** agentic, multi-turn tool use re-sends the whole growing transcript on every turn (typically 20–80k tokens, ~15 turns per attempt). So:
  - **Prefix caching is mandatory.** Use vLLM `--enable-prefix-caching` or SGLang's radix cache. Never benchmark, tune or submit a tool-use configuration with it off: throughput or KV numbers measured with it off do not transfer.
  - **Hybrid models (Qwen3.5/3.8 DeltaNet layers):** prefix caching depends on vLLM saving DeltaNet states at block boundaries (`mamba_cache_mode` "align"). Both 0.23 and 0.31 enable it by default with prefix caching; check the server log (0.31: `Mamba cache mode is set to 'align'`).
    - On tool traffic, vLLM 0.31 still reached only ~62% prefix hits (vs ~86% on 0.23) with MTP2, cause open (E027).
    - Compare hit rates before switching versions.
  - **Choose the continuous-batching size (concurrency) by cache health, not just KV capacity.**
    - Too many concurrent attempts evict each other's cached prefixes, and every turn then re-prefills from scratch.
    - When sizing concurrency, measure the prefix-cache hit rate and evictions/preemptions on a realistic multi-turn replay (E014).
    - Keep the hit rate healthy, high and stable over the run. If it drops or evictions grow, lower concurrency or add KV offload to host memory.
  - **Log cache metrics during real runs:** the prefix-cache hit rate, KV usage and preemptions/retractions.
  - **Pool sizing (E019/E020):** one model copy per GPU (~600–630 tok/s per RTX PRO 6000 on tool traffic) beats TP4. Size concurrency per box as KV tokens ÷ p90 attempt context (≈10–12 for xhigh-96k attempts, ≈20 for 48k); early-run windows overstate it. Summary: `notes/2026-10-13_qwen_harness_and_pool_findings.md`.
  - Current setup and evidence: `experiments/E013_kaggle_submission/README.md` (vLLM 0.23, prefix caching on, KV 1.40M tokens, concurrency 35) and `experiments/E014_sglang_kaggle/README.md`.
- **E020 options of `evals/run_qwen_tools.py` (harness `qwen-tools-5`, opt-in; defaults reproduce `qwen-tools-4`):**
  - `--task-summary`, `--helpers` (also preloaded inside `submit_program` processes), `--prompt lean2progroles`: E017 ports.
  - `--turn-think-cap N --turn-think-grace G`: after N thinking tokens in a turn, a "test it with code now" note; after G more, the thinking is closed into a `run_python` call.
  - `--feedback rich`: colour confusions, small-grid diffs and shape hints for rejected programs.
  - `--confirm-test`: shows the first train-passing test output once; the model re-submits to answer (`first_tp_pred` recorded).
  - `--settle N --pool-in F`: adaptive k; skip a pair's next attempt once N train-passing attempts agree on the test output.
  - `--augment d8|colour|d8colour`: attempts ≥1 see a transformed task; the answer is mapped back.
  - `--record-alternatives` (free attempt_2 bookkeeping), `--think-call-rescue` (execute a tool call written inside open thinking after a nudge).
  - Pool operation: `--worker-slots auto:N` (harness-side per-box slots, direct worker URLs; the gateway ignores `x-smg-target-worker`), `--request-timeout 1200`, `--session-mem-mb 2048`; see `experiments/E019_qwen_program_answers/router/README.md`.
  - Never put "sglang" in eval command lines (use `--base-url`): other agents' `pkill -f sglang` killed every run once.
  - Offline selection vs k: `experiments/E020_qwen_harness_80/select_sim.py`. Launcher: `experiments/E020_qwen_harness_80/launch.sh`.
- **E027 throughput findings for 4×L4 (`experiments/E027_l4_throughput/README.md`):**
  - **Kaggle hardware:** 4×L4 is only available to notebooks attached to the ARC-AGI-2 competition (otherwise the push silently gets 2×T4). The links are PCIe Gen3, and NCCL's default SHM all-reduce runs at ~4.9 GB/s.
  - **Never set `NCCL_P2P_LEVEL=SYS` on Kaggle:** P2P there is ~12× slower than SHM. It helps only on the Vast box.
  - **Rejected:** 4-bit weights (INT4/AWQ) lose tool accuracy; pipeline parallelism is slower on decode; MTP2 + TurboQuant KV produces broken text on vLLM 0.31.
  - **Also rejected (2026-10-17):** MTP2 on vLLM 0.31 (accuracy tie, slower on real tool traffic: 0.31 reuses less prefix cache) and tuned FP8 GEMM configs (faster on the decode bench, slower on a real-traffic probe, unresolved; v4d not pushed).
  - **Transcript compaction (opt-in, off by default):** `AgentConfig.compact_at/compact_keep/compact_tool_chars`, solver `ARC_COMPACT_AT/ARC_COMPACT_KEEP/ARC_COMPACT_TOOL_CHARS`. It empties earlier turns' thinking and cuts old tool outputs once per +50% transcript growth. Tests: `evals/tests/test_qwen_tools.py` `CompactTest`.
- **Never kill processes by name on this host.** `pkill -f` patterns such as `sglang` or `vllm` also match other agents'
  eval runs and the SGLang gateway (E021 incident, 2026-10-12: a test of the Kaggle emergency writer killed E020's runs).
  In `kaggle/`, the emergency writer, the notebook guard cell and `kill_server` use name patterns **only on Kaggle**
  (`/kaggle/input` exists and no `ARC_EXTERNAL_SERVER`); everywhere else they kill only the solver's own descendants
  (`solver.descendants`). `kaggle/tests/fake_v3_test.py` checks this with a decoy process.
- `data_tools/contamination_check.py`: checks training data against ARC-AGI-2 eval grids, invariant to rotation, flip and colour relabelling. Reports go to `data_tools/contamination_report.json`.
