# E021: Kaggle submission notebook v3 (v2 + today's harness findings, as toggles)

Started 2026-10-12. Code: `kaggle/src/solver.py`, `kaggle/build_notebook.py --variant v3`, notebook `kaggle/nb_v3/`
(same slug as v2, `chankhavu/arc2-qwen38-tools-submit`, so a push becomes version 3). Flags: `kaggle/README.md`
("v3"). Nothing was pushed to Kaggle and no Kaggle run was started.

## Goal
Same compute as v2 (12 h rerun, 240 tasks / 259 test pairs, 4×L4, about 44k tokens per pair at ~295 tok/s), but a
better score, by porting today's findings (E018, E019, E020) as switchable options. The defaults stay v2 until
E020's K arms decide.

## Changes vs v2

| Area | v2 | v3 | Default |
|---|---|---|---|
| Harness code | qwen-tools-3 snapshot of `agent_qwen`/`executor` | **The current eval modules**, written by the notebook with the repo layout: `arc_harness/{grids,arcprize,prompts,executor,audit,selection,agent_qwen}.py`, `worked_example.txt`, `arc_helpers.py`, `task_summary.py`. The solver only maps settings to `AgentConfig` fields (`agent_options`). | same as evals |
| attempt_2 | copy of the test input | `ARC_FREE_ATTEMPT2=1`: the agent's `record_alternatives` + `record_train_check` (E020 fields; no tokens) feed `selection.select_two`. Order: train-passing grids, then the attempts' own answers, then the rest by train fraction. The test-input copy is used only when nothing else exists. | off |
| Second attempts | `legacy`: queued for every pair, launched only with ≥ 32k share (in practice never) | `ARC_ATTEMPT2=gated` (below) | legacy |
| Thinking | none | `ARC_TURN_THINK_CAP` / `_GRACE` (E020 per-turn cap, "test it with code now" nudge) | off |
| Answer mode etc. | submit_answer | `ARC_TOOLS_ANSWER_MODE=program`, `ARC_TOOLS_PROMPT=lean2progroles`, `ARC_TASK_SUMMARY`, `ARC_HELPERS`, `ARC_FEEDBACK=rich` | off |
| Memory | executor RLIMIT_AS 4 GB | + per-session cap at VmSize + 2 GB (`executor.SESSION_MEM_CAP`, the E020 `--session-mem-mb` text), submitted programs 2 GB | **on** |
| Concurrency | KV / 40k = 35 | KV / p90 live context (`ARC_KV_PER_SEQ` 48000) = **29** on vLLM 0.23's 1.40M KV | **on** |
| Cache metrics | vLLM `/metrics`, SGLang `/v1/loads` | + SGLang `/metrics` fallback (older SGLang without `/v1/loads`) | |
| Feature self-test | executor only | Every enabled feature is exercised on CPU before GPU work. A broken one is switched off (v2 behaviour) and logged; `phases_s.v3_selftest` and `phases_s.agent_config` go into the summary. | |
| Kill paths | emergency writer and guard: `pkill -f '[s]glang'` etc. | **descendants of the solver only** (from /proc); name patterns only on Kaggle (`/kaggle/input` exists, no external server). See "Incident". | |
| Summary | | `v3` block: gated counters, attempts by index, think nudges, train passes, alternatives, slot-2 sources. Local mode adds `score_attempt1_only` and per-pair `alt_correct`. | |

### Gated second attempt (`ARC_ATTEMPT2=gated`)
- **Eligibility** (`selection.retry_category`): no grid of the pair passes train (neither the answer nor an
  alternative). Ranked, in this order:
  1. the model's submitted answer fails the train check;
  2. a submitted answer that cannot be checked (`ARC_ATTEMPT2_UNKNOWN=1`, the default);
  3. no answer, or only a fallback answer (forced, session, salvage, best failing program...).
  - Ties go to the higher best train fraction, then fewer first-attempt tokens.
- **Why this order: E012 retry values** (full set, v2 config, 167 pairs × 2 attempts; attempt 1 solving a pair that
  attempt 0 got wrong):

  | attempt-0 outcome | pairs | a0 wrong | a1 solves | per retry | a1 tokens |
  |---|---|---|---|---|---|
  | submitted answer | 108 | 66 | 10 | 15.2% of wrong ones, 9.3% of all | 34.0k |
  | forced at the cap | 31 | 31 | 0 | 0% | 43.4k |
  | no answer | 28 | 28 | 2 | 7.1% | 43.5k |

  - Retrying only "no answer or fallback" pairs (the original spec) is worth 2/59 retries (3.4%) at 43k each.
  - Retrying uncertain submitted answers is worth about 3× more per token.
  - With K0's train check, category 1 splits into train-fail (category 0, the best retries) and train-pass (no retry).
- **Scheduling** (`Scheduler.next_job`, `gated_budget`): a second attempt starts only when all three hold:
  1. no first attempt is pending: all have started; a first attempt requeued after a server crash goes first;
  2. budget ≤ capacity left = measured rate × time to force − remaining budgets of everything in flight;
  3. budget ≤ 0.85 × per-attempt speed × (time to force − 2 min). Per-attempt speed is the median tokens / wall s
     of finished first attempts, which includes tool time.
  - Budget = min(`ARC_B_MAX2`, `ARC_TOOLS_CAP`, (2), (3)); it starts only if ≥ `ARC_B_MIN2` (32k).
  - When (3) fails the queue is dropped, since time only shrinks. When (2) fails the worker waits: capacity comes
    back as attempts finish under budget.
  - Workers do not exit while a first attempt can still queue a retry.
  - Deadline forcing (force 11:00, hard stop 11:30) applies to second attempts exactly as to first ones.

## Tests (no GPU)
- `kaggle/tests/fake_v3_test.py`, all pass:
  - **No name-based kill off Kaggle:**
    - a static check of every `pkill` site;
    - in a child process: `kill_children` / `kill_server` issue no pkill and do kill their own descendant;
    - a decoy process whose argv has `sglang`, `vllm.entrypoints` and `VLLM::` must survive the emergency path.
  - **Selection:**
    - v2 rule without metadata;
    - a train-passing alternative goes first and the answer is kept;
    - a failing answer competes by train fraction;
    - a second attempt's answer beats first-attempt alternatives;
    - invalid grids are ignored;
    - malformed metadata falls back to the v2 rule.
  - **Retry gating:** categories and priority order.
  - **Gated scheduler unit:**
    - priority order;
    - budget bounds;
    - first attempts always first;
    - capacity wait keeps the job;
    - no-time drop;
    - workers wait while first attempts are in flight;
    - `off` / `legacy`.
  - **Memory caps:**
    - a 3 GB allocation in a session gives MemoryError, state is kept, 500 MB still works;
    - a 2.5 GB submitted program fails;
    - the self-test turns off a mismatched prompt.
  - **End to end against a scripted fake server**, on synthetic flip tasks so train checks are real. Scores out of
    10 pairs:
    - v2 defaults: 2;
    - free attempt_2: 4, because "bad" pairs are rescued by the train-passing alternative;
    - gated: 10. Second attempts run only for eligible pairs and only after every first attempt has started;
      with `ARC_ATTEMPT2_UNKNOWN=0`: 8;
    - gated + deadline: everything ends by the hard stop;
    - program mode + helpers in the program namespace + task summary + think cap (nudges and forced acts
      recorded): 10.
  - **submission.json failure paths:**
    - emergency writer: a solver subprocess whose server answers `/v1/models` and never completes; the file is
      corrupted mid-run; the timer fires at 25 s and leaves a valid file; exit 0; no processes left;
    - the notebook's final cell repairs 11 broken slots.
- Unchanged and passing: `fake_server_test.py` (v1 plain path, legacy attempt 2), `fake_tools_test.py`, and
  `evals/tests/test_qwen_{tools,program,e020}.py` (58 tests).
- `kaggle/tests/check_notebook.py --variant v3 --extract-to /tmp/e021_nb_v3`:
  - nbformat and nbconvert pass;
  - every embedded file equals the repo;
  - all 10 modules import;
  - all three fake suites pass against the extracted modules: NOTEBOOK CHECK OK.

## Functional runs on the local 2×3090 SGLang server (AWQ-INT4, `qwen38-sglang-radix`, 13 slots)
- **Driver:** `scripts/local_run.py`. It runs the extracted notebook modules, the notebook's guard cell and final
  cell (paths remapped), and an input dir whose test file holds public-eval tasks.
  - Those ids differ from the public test file, so the solver takes the **competition-rerun path** (mode submit,
    deadlines from T0).
  - Scoring happens afterwards against the eval solutions.
- **Pairs:** 1818057f, 2ba387bc, dd6b8c4b ×2, 1ae2feb7 ×3, 136b0064, 58f5dbd5 (6 tasks / 9 pairs). Concurrency from
  KV: 343,619 / 48k → 8.
- Outputs: `runs/<run>/` (run.log, result.json, working/summary_submit.json, attempt records, traces).

| run | config | deadlines (s from T0) | what triggered | result |
|---|---|---|---|---|
| A | submit, think cap 3000/1024, free attempt_2, gated (+unknown), 8k caps | force 900, hard 915 | 9 first + 9 gated attempts; all done at 7.1 min | 2/6 tasks; 116.6k tokens, 275 tok/s, cache hit ~92%, KV ≤ 49%, 0 retractions; 20 think nudges; valid file, exit 0 |
| A2 | as A, 16k caps | force 240, hard 250 | **force**: 7 attempts cut at 4:00 (`forced=True/deadline`); the gated queue dropped for lack of time | 2/6; valid file, exit 0 |
| A3 | as A | hard 150 (< force) | **hard stop**: 7 in-flight attempts cancelled at 2:30, the 2 answers kept | 2/6; valid file, exit 0 |
| B | program + lean2progroles + summary + helpers + rich feedback + think cap + free + gated | emergency 300 | **emergency writer** at 5:00 mid-run | 3/6 (58f5dbd5 solved in program mode); valid file, exit 0 |
| C | as A | guard 100 | **notebook guard** at 1:40: solver group plus its descendants killed (exit −9) | final cell: valid file (fallback), 0 slots repaired; no orphaned executors |

- Before and after each run, `ps -u vu` was compared. E020's 6 `run_qwen_tools` processes, the gateway and the
  cache logger were untouched by every run after the fix.
- **Findings from the runs:**
  - In A2 the cut attempts had no answer. With a 10 s gap between force and hard stop, the forced
    `submit_answer` call's deadline (`max(hard − 180, min(now + 30, hard − 10))`) equals the force time.
  - On Kaggle the gap is 30 min, so the forced turn gets its 30 s. That is the designed behaviour, not a bug, but
    never set `ARC_HARD_STOP_S` within ~1 min of `ARC_FORCE_AT_S`.
  - Run B (AWQ, program mode): 2 of the 2 attempts that received a think-cap nudge wrote a `<tool_call>` *inside*
    the still-open thinking and then hit EOS. The call is never executed, and the turn ends `unclosed_think` after
    the EOS continues.
    - In program mode that ends the attempt (here at ~4k of 16k tokens). In submit mode it would go to a forced
      submit.
    - E020's NVFP4 files show **0 such cases in 137 nudged attempts** (K1, x96pc8 shards), so this is likely
      local/AWQ behaviour.
    - Suggested harness fix, for E020 to decide: if the thinking is still open but the text after the nudge holds
      a complete `<tool_call>…</tool_call>`, insert `</think>\n\n` before it and execute the call.

## Budget and deadline analysis (Kaggle rerun, 259 pairs)
- **Timeline** (v2 commit run): server ready at 562 s, first token at 570 s, force at 39,600 s. That leaves a
  ~39,000 s (10.8 h) generation window.
- **Token supply:**
  - 220 tok/s (v2's prior) gives 8.6M tokens, **33k per pair**.
  - 295 tok/s (E013 end-to-end, vLLM 0.19, 16 seqs, MTP 3) gives 11.5M, **44k per pair**.
  - vLLM 0.23 at concurrency 29 under real load is **not measured** (E014 replay never got a 4×L4).
- **First attempts:** the v2 config (K0) averages 35.6k tokens per attempt at the 48k cap (E012; median 40k).
  - The adaptive share binds when supply is below ~36k per pair: at 220 tok/s, first attempts get ~33k caps and
    nothing is left over.
  - Program mode (K2) uses about +8% tokens (E019: 39.5k vs 36.7k on dev-56).
  - K1's think-cap token effect: pending (E020).
- **Leftover for gated attempts** = supply − Σ first-attempt tokens:
  - at 295 tok/s: 11.5M − 259 × 35.6k = **~2.3M tokens, about 55 retries of 42k**;
  - at 250 tok/s: ~0.5M, about 12 retries;
  - at ≤ 235 tok/s: none.
  - Gating is therefore safe on a slow run: no retry starts when there is no leftover.
- **When retries can run:**
  - Per-attempt speed ≈ total rate / 29 ≈ 10 tok/s plus tool time, so a 42k retry takes ~70–80 min.
  - The first-attempt queue empties after about (259 − 29)/259 of the first-attempt work, ~7.7 h of generation:
    about **T0 + 8 h**.
  - Retries can start until about force − 80/0.85 min ≈ **T0 + 9.4 h**.
  - Slots free up as first attempts finish, so the window holds roughly 1–2 retries per slot. That is consistent
    with the ~55 that the token supply allows.
- **Expected value of gated retries** (E012 rates, best-EV categories first):
  - ~55 retries × 9–15% ≈ 5–8 pairs solved, ≈ **+2–3 points** pair-weighted, only if throughput is near 295 tok/s.
  - E020's offline estimate (K0 train checks) should replace this.
- **Free attempt_2:** no tokens. Its value is E020's offline estimate (pending). In E019 dev-56 the alternatives
  were rarely different and correct, so expect +0–2 points.
- **Expected tokens per pair:** ≈ supply (33–44k). First attempts take ~35.6k (K0); gated retries absorb any
  leftover before 11:00.

## Push commands (do not run without the user's go-ahead: a push starts a 4×L4 commit run, ~25 GPU-min)

```bash
cd /home/vu/Workspace/ARC-AGI-2
export KAGGLE_API_TOKEN="$(sed -n 's/^kaggle_api_token:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
# 1. set the K-arm decision in experiments/E021_kaggle_v3/v3_settings.json, then build + check
.venv/bin/python kaggle/build_notebook.py --variant v3 --env "$(cat experiments/E021_kaggle_v3/v3_settings.json)"
.venv/bin/python kaggle/tests/check_notebook.py --variant v3 --extract-to /tmp/nb_v3_check     # must print NOTEBOOK CHECK OK
# 2. commit run (local self-scored test set, gate in the last cell); creates version 3 of the kernel
.venv/bin/kaggle kernels push -p kaggle/nb_v3
.venv/bin/kaggle kernels status chankhavu/arc2-qwen38-tools-submit
.venv/bin/kaggle kernels output chankhavu/arc2-qwen38-tools-submit -p kaggle/runs/v3-v3        # check summary_local.json, COMMIT GATE: PASS
# 3. competition submission (uses a daily slot; user approval)
.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-2 -k chankhavu/arc2-qwen38-tools-submit -v 3 \
    -f submission.json -m "v3: qwen-tools-5 (<K arm>), free attempt_2, gated 2nd attempt, conc 29"
.venv/bin/python kaggle/monitor_submissions.py --once
```

`kaggle/nb_v3/kernel-metadata.json` is the same as v2's:
- internet off, `NvidiaL4`;
- the user's `chankhavu/vllm-0-23-0-py312-cu130` kernel source;
- `chankhavu/arc2-local-testcases`;
- the verified model mirror.

No new Kaggle dataset is needed: every module is embedded in the notebook.

## Risks
1. **Throughput at concurrency 29 on vLLM 0.23 is unmeasured.**
   - Below ~240 tok/s there is no leftover, so the gated attempt does nothing; harmless.
   - 29 instead of 35 may lower total decode throughput a little. The trade is a healthy prefix cache in the long
     tail: E014 working-set p99.9 is 1.17M at 28 vs 1.42M at 35.
2. **Gated retries share decode with the tail of first attempts.**
   - They start only after every first attempt has started, and within conservative capacity.
   - Still, late first attempts decode a little slower and are more often cut at 11:00.
3. **Snapshot drift.** The notebook embeds the eval modules at build time, and E020 is still changing
   `agent_qwen.py`. Rebuild and run `check_notebook.py --variant v3` right before any push.
4. **`record_alternatives` adds executor work at the end of each attempt:** program runs on the test input, the
   session `solve`, salvage. Each call is capped (~65 s), so worst case it takes minutes near the deadline, inside
   the 30-min force-to-hard-stop gap.
5. **Program mode on Kaggle** spawns one Python process per grid per submission. That is fine on 48 CPUs, but it
   was never measured at 29 concurrent attempts on the Kaggle image (the CPU check covered LocalExecutor only).
6. **Model differences.** Kaggle runs FP8, the evals NVFP4, local smoke AWQ. Run B's "tool call inside open
   thinking" after a nudge shows quantization can change agent behaviour.
7. **The 2 GB session cap** may turn a legitimate large allocation into a MemoryError. The model sees the error and
   the session survives; this is rare for ARC grids.
8. **v2 rerun 57088016 is still PENDING** (21 h after submission at 21:05 UTC). v3 has the same deadlines; if Kaggle
   queues reruns, our elapsed-time watcher is not the notebook clock.
9. **Selection can place an alternative in attempt_1.** Scoring is unaffected (pass@2), but keep this in mind when
   reading slot sources.

## Incident (2026-10-12, ~21:00–21:19 UTC), caused by this experiment
1. **K0 crash.** A transient edit of `evals/arc_harness/agent_qwen.py` (20:58:39–20:59:17 and 20:59:38–20:59:57)
   left an undefined name, and E020's K0 crashed on import.
   - The file was restored byte-identical to E020's version. v3 now uses E020's `record_alternatives` instead.
2. **Killed processes.** `fake_v3_test`'s emergency-path test ran the solver's emergency writer on this host. Its
   `pkill -9 -f '[s]glang'` (unchanged since v2) killed vu processes whose command line mentions sglang: E020's
   `run_qwen_tools.py --endpoint local-sglang-router-…` and the `smg` gateway. This happened around 21:09, 21:10,
   21:12 and 21:19 UTC.
   - Fixed:
     - the solver, the notebook guard cell and `kill_server` now kill only their own descendants (from /proc);
     - name patterns run only on Kaggle;
     - a test with a decoy process guards this;
     - AGENTS.md has the rule.
   - Rule kept from then on: files under `evals/arc_harness/` are not edited while evals run, and shared changes
     go to the coordinator as diffs.

## Status
- **Code ready.** Defaults = v2 + robustness (memory cap, concurrency 29), in `v3_settings.json`.
- **Waiting for the K-arm decisions** (E020: K0 v2 config, K1 + think cap 8192, K2 + program/summary/helpers,
  K3 = K2 xhigh) and the free-attempt_2 / gated estimates. Then set `v3_settings.json`, rebuild, check, and hand the
  push commands to the user.

## v4 (2026-10-12 late): the v3 code with E020's K3 settings
E020 chose K3 from 2 full-set samples per arm: 37.4% mean pass@1, ~38.3% gated pass@2 at a 44k mean, against
~27–28% for v2/v3.

- **Settings** `v4_settings.json` (builds into `kaggle/nb_v4/`, same slug):
  - `ARC_TOOLS_TIER=xhigh`, `ARC_TOOLS_CAP=48000`, `ARC_TOOLS_SOFT=24000`;
  - `ARC_TOOLS_ANSWER_MODE=program`, `ARC_TOOLS_PROMPT=lean2progroles`, `ARC_TASK_SUMMARY=1`, `ARC_HELPERS=1`,
    `ARC_FEEDBACK=rich`;
  - `ARC_TURN_THINK_CAP=8192`, `ARC_TURN_THINK_GRACE=2048`, `ARC_THINK_CALL_RESCUE=1` (new setting: E020's
    `think_call_rescue`);
  - `ARC_FREE_ATTEMPT2=1`, `ARC_ATTEMPT2=gated`, `ARC_ATTEMPT2_UNKNOWN=0`;
  - `ARC_SESSION_MEM_MB=2048`, `ARC_KV_PER_SEQ=48000` (concurrency 29), `ARC_RATE_PRIOR=220`.
  - The rest (vLLM 0.23, MTP 0, local commit test) is as v3.
- **K3 config diff** (`scripts/k3_config_diff.py v4_settings.json evals/results/qwen_tools_e020_K3_all.jsonl`, which
  builds the notebook's own `build_agent_config` and compares it with K3's recorded config): **28 of 30 fields
  equal**. The two differences:
  - `routing_key_header`: gateway-only bookkeeping, absent on Kaggle;
  - `think_call_rescue`: on in v4, off in K3.
- **Differences outside the config:**
  - executor: K3 ran the Docker sandbox, Kaggle runs `LocalExecutor`;
  - both use a 2 GB session cap and 2 GB programs (K3: `--session-mem-mb 2048`);
  - model: NVFP4 on SGLang (K3) vs FP8 on vLLM 0.23 (Kaggle);
  - the hard cap on Kaggle is min(48k, adaptive share), so it can be below 48k when throughput is low.
- **Retry gating in program mode** (`selection.retry_category`, unchanged):
  - `program_pass` → no retry;
  - `program_best_fail` / `session_solve` failing train / no answer → category 2;
  - a `session_solve` that reproduces every train pair → no retry.
  - Tests were added for these cases.
  - In K3, 83 of 167 attempt-0 pairs are eligible, and attempt 1 solves 9 of them (10.8%) at 46.6k tokens.
- **CPU checks:** `check_notebook.py --variant v4 --extract-to /tmp/e021_nb_v4`: NOTEBOOK CHECK OK (fake_server,
  fake_tools and fake_v3 against the extracted modules). This host has no `/kaggle`, so no name-based kill can run.
- **Local functional run** `runs/local_v4_K3`:
  - setup: AWQ 2×3090, 6 tasks / 8 pairs, rerun path, local overrides (24k cap, 12k soft, SGLang, 98k context),
    force at 15 min;
  - all features passed the self-test: session cap, helpers, the program checker with a 2.5 GB program refused
    under the 2 GB cap, task summary, think cap, rescue, xhigh;
  - results: 8 attempts, 136k tokens, 218 tok/s, cache hit ~90%, KV ≤ 49%, 0 retractions;
  - answer sources: program_pass 4 (all correct), session_solve 1 (correct), none 3;
  - **4/6 tasks**, 11 think nudges, 0 forced acts, 0 rescues;
  - the 4 eligible retries were dropped for lack of time, as designed for a 15-min window;
  - valid file, exit 0, no leftover processes.
  - Not exercised in this run: alternatives (none arose), rescue (no in-thinking call), the forced deadline.
    These are covered by the fake suites and the v3 runs.
- **Budget (K3: 38.1k mean tokens per attempt, median 48k, final-context p90 67k):**
  - Supply over the 10.8 h window: 220 tok/s → 8.6M (33k/pair); 250 → 9.8M (37.7k/pair); 295 → 11.5M (44.4k/pair).
  - **At ~295 tok/s:** first attempts 259 × 38.1k = 9.9M, leftover ~1.6M, about 35 gated retries at 46.6k ×
    10.8% ≈ 4 pairs (≈ +1.5 points). Retries start around T0 + 8 h and only while they can finish before 11:00.
  - **At ~250 tok/s:** no leftover, so no retries. The adaptive share binds at ~38k, so most first attempts run
    with caps below 48k.
  - K3's accuracy is steep in the cap: pass@1 if attempts had been capped (lower bound, correct within C tokens)
    is 36.5% at 48k, 33.2% at 44k, 27.8% at 38k, 24.6% at 33k. **Throughput matters more than the retry for v4.**
  - `ARC_RATE_PRIOR=220` (from v2) sets the first wave's caps (~29 pairs) to ~33k until the rate is measured after
    15 min. Raising it to ~280 gives early pairs ~42k; later shares adjust either way.
  - Concurrency 29 holds K3's larger contexts (E014 working set scaled ×1.1: p99.9 ≈ 1.29M < 1.40M), but this is
    unmeasured on L4.
