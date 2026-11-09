# E020: a Qwen3.8-27B harness toward 80% pass@2 (no fine-tuning)

Goal: get Qwen3.8-27B (NVFP4, self-hosted SGLang gateway, no fine-tuning) as close as possible to 80% Kaggle pass@2
on the 120-task / 167-pair ARC-AGI-2 public evaluation. 80% is a stretch goal; numbers below are reported as measured.

Reference points (Kaggle pass@2 / pass@1, pair-weighted):
- no tools, xhigh, temp 1.0, 240k budget (E008): 47.9% / 35.9% at 73.6k tokens per attempt;
- tools, medium, 48k hard cap, submit_answer (E012): 33.5% / 24.6% at 35.6k;
- ARC Prize semi-private for this model: 42.4%.

## Summary (final state of this agent, 2026-10-13 05:50 UTC)
- **Ceiling (no Kaggle budget):** x96pc8 (xhigh, 96k, program answers + summary + helpers + roles + rich feedback +
  per-turn think cap) scores **56.0% pass@1** (58.1% task-weighted) at 55.9k tokens/attempt, +21.1 [+12.1, +29.8]
  over no-tool xhigh-240k. Pooling it with x96p and the two E008 no-tool samples gives **69.5% pass@2** (70.7%
  task-weighted, oracle 71.3%) at 261k tokens/pair; with every candidate logged so far (median 5 per pair, incl. the
  partial x96pal / x96pex / x96pc8 2nd sample) **71.3% / 72.4%**, oracle 74.3%. 80% was not reached; coverage
  (oracle) is the limit, not selection.
- **Kaggle budget (~44k tokens/pair):** K3 (xhigh, 48k / 24k, program + summary + helpers + roles + rich feedback +
  think cap) is the best arm: 2-sample pass@1 37.4% vs 26.9% for the submitted config (K0), +10.2 [+3.0, +17.4] on
  attempt 0; gated pass@2 at a 44k mean 38.3% / 39.9% task-weighted (K2: 37.1%, a tie); at 64k 43.1%.
- **What mattered:** program answers with a host-side train check plus the E017 perception pack and helpers (+8.4 pp
  at medium-48k, K2nc − K0), xhigh tier (+3.6 to +5.4 pp at 48k), a bigger budget (48k → 96k: +19 pp at xhigh), and the
  per-turn think cap at xhigh-96k (+7.2 [+0.6, +14.4]). Not helpful: think cap at medium-48k (+0.6, −1.8), the
  test-audit/lint review turn (x96pal, partial: +0.8 [−7.1, +8.7]; it changed the answer in 1 of 91 reviews), free
  attempt_2 (+0 to +1.2 pp).

## Recommended configs
- **Kaggle (E021 port):** K3 + gated 2nd attempt:
  `--answer-mode program --tier xhigh --hard 48000 --soft 24000 --prompt lean2progroles --task-summary --helpers
  --feedback rich --turn-think-cap 8192 --record-alternatives` (+ `--think-call-rescue` on vLLM: 1 of 51 medium-tier
  nudges wrote its tool call inside open thinking), attempt_1/2 by `selection.select_two`, retries by
  `selection.retry_priority` (in program mode only pairs without a train-passing grid qualify, ~50%).
- **Best unconstrained harness:** x96pc8 x k samples + no-tool xhigh samples, top-2 by `vote_tp` (`select_sim.py`):
  ```bash
  SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh x96pc8 all <k> 20      # program attempts
  .venv/bin/python experiments/E020_qwen_harness_80/select_sim.py c8='<x96pc8 files>' nt=evals/results/qwen_nvfp4_gen_xhigh_240k_t1.jsonl --k <n> --ordered --policies vote_tp
  ```

## Open questions
- x96pal (122/167), x96pex (8/167) and the 2nd x96pc8 sample (23/167) are partial; finish them to settle items 1-3 at
  xhigh-96k and to measure pass@2 vs k for x96pc8 alone.
- Cross-attempt memory (`--memory`) and `--augment` / `--confirm-test` / `--settle` are implemented and unit-tested
  but were not run on the gateway.
- A real gated-retry run (instead of the 2-sample simulation) and the retry order (0,2,1) vs (0,1,2) in submit mode
  (the order is irrelevant in program mode).

## Harness changes (`qwen-tools-5`, all opt-in; defaults = `qwen-tools-4`)
Code: `evals/arc_harness/agent_qwen.py`, `evals/arc_harness/executor.py`, `evals/run_qwen_tools.py`.
Tests: `.venv/bin/python -m unittest evals/tests/test_qwen_e020.py` (7 tests), plus the 42 tests of
`test_qwen_tools.py` / `test_qwen_program.py`, unchanged and passing. With every new option at its default the config
dict, prompt and harness version are byte-identical to qwen-tools-4 (checked against the pre-E020 module), so
E019 files resume.

| Flag | What it does |
|---|---|
| `--task-summary` | E017 P1: the host-computed perception pack (`evals/task_summary.py`) appended to the first message. |
| `--helpers` | E017 P3: `evals/arc_helpers.py` preloaded in the session (re-run after a reset) **and** executed in the namespace of every `submit_program` process, so programs may call `objects`, `crop`, `d8`, ... The tool docs say so; prompt line "Use check(solve) after every change to solve." |
| `--prompt lean2progroles` | `lean2prog` + the E017 ROLES / no-"noise" / test-difference-audit text (verbatim). |
| `--turn-think-cap N` / `--turn-think-grace G` | Per-turn thinking cap (H3). After N thinking tokens in one turn the harness appends, inside the thinking, "I have been reasoning for a long time without running anything. I should stop here and test my current hypothesis with code now (and use code, not my head, for any counting or cell reading)." If the thinking is still open G tokens later, it is closed and a `run_python` call is opened (the model writes the code). Recorded: `think_nudges`, `think_forced`, per-turn `think_nudge` / `think_forced`. |
| `--feedback rich` | H5: rejected programs also get colour confusions (expected->got counts), a small-grid diff of the first failing pair (expected grid and the program's output with '.' for correct cells, grids up to 20x20), and shape hints (transposed, k-times too large/small) with both grids when small. |
| `--confirm-test` | On the first train-passing program, the model sees its test output (digit rows) once and must call `submit_program` again (same or fixed code) to answer. `first_tp_pred` is recorded. |
| `--test-audit` | As `--confirm-test`, plus host-side flags (`evals/arc_harness/audit.py`, no tokens, never the test output): colours in no training output and not in test_input; a broken size rule (same as input, constant, fixed ratio); output identical to the input or near-empty; the task summary's "Outside train range" line; a test run >5x slower than train. Recorded in `audit`. |
| `--lint` | AST lint of submitted programs: literal tables (>=6 entries), literal cell indices with a value >=2, literal copies of training grids, comparisons with literal grids, branches on an exact size >3. Shown after rejection feedback and in the review turn (with lint alone, a clean passing program is accepted directly). |
| `--worked-example` | About 500 tokens: a workflow demo on ARC **training** task 3aa6fb7a (inspect with code, one-line hypothesis, draft `solve`, `check`, test-assumption check, submit), with the real outputs of its code. Needs `--helpers`. |
| `--memory N` | Cross-attempt memory: attempt i's first message lists up to N programs of earlier attempts of the pair that failed the train check (truncated to 1,500 characters each), plus a count of attempts that wrote no program. Recorded in the config and in `memory_from`. Not for use with `--augment`. |
| `--augment d8\|colour\|d8colour` | Attempts >=1 see the task under a D8 transform and/or a colour permutation of 1-9; the answer grid is mapped back (attempt 0 is never transformed). |
| `--record-alternatives` | Bookkeeping, no tokens: other test grids of the attempt (`alternatives`: earlier submissions run on the test input and the session's `solve` in program mode; `answer`, `solve(test_input)` and the salvage function in submit mode) for a free attempt_2. |
| `--think-call-rescue` | After a think nudge, a complete `<tool_call>` written inside still-open thinking is executed (`</think>` inserted before it); count in `think_call_rescued`. |
| `--worker-slots auto:N\|url=n,...` + `--slot-state F` | Harness-side per-box attempt slots shared by all processes (`evals/arc_harness/slots.py`); each attempt talks directly to its worker's URL. Not in the config. |
| `--request-timeout S` | Client timeout per completion request (default 3600; launch.sh 1200) against half-open tunnel connections. Not in the config. |
| `--session-mem-mb N` | Safety, not in the config: an RLIMIT_AS cap of the session kernel at its VmSize + N MB, and submitted programs at min(3072, N) MB. A runaway cell raises MemoryError. |
| `--settle N` + `--pool-in F` | Adaptive k: with `--attempts K`, an attempt is skipped at dispatch once N finished attempts of its pair (this file and the `--pool-in` files) passed train with the same test output. Uses only predictions and train checks, never labels. |

Offline tools: `select_sim.py` (selection policies vs k, both metrics, oracle, tokens/pair, paired CI), `launch.sh`
(arms x shards), `shard0.txt` / `shard1.txt` (alternate pairs of the 167, so a second process can be added later).

## Runs (final for this round: all Vast boxes were destroyed at ~06:00 UTC, 2026-10-13; the last three runs were stopped with SIGTERM and are resumable)
All on the full public eval (167 pairs), program mode unless noted, temp 1.0 / top_p 0.95 / top_k 20, through the
worker-slot scheduler since 21:24 (10 per box, 5 boxes). Files `evals/results/qwen_tools_e020_<run>.jsonl` (+ `_traces/`,
logs `log_qwen_tools_e020_<run>.txt`); launched by `launch.sh <arm> <shard|all> <attempts> <conc>`.

| run (file) | config | done | status |
|---|---|---|---|
| `x96p_shard0/1` | xhigh, 96k hard / 48k soft, `lean2progroles`, summary, helpers, rich feedback | 167 x 1 | done |
| `x96pc8_shard0/1` | x96p + `--turn-think-cap 8000 --turn-think-grace 2048` | 167 x 1 | done |
| `x96pc8_all` | x96pc8, attempt index 1 (2nd sample) | 23 / 167 | stopped (resumable) |
| `x96pal_all` | x96p + `--test-audit --lint` (items 1+2) | 122 / 167 | stopped (resumable) |
| `x96pex_all` | x96p + `--worked-example` (item 3) | 8 / 167 | stopped (resumable) |
| `K0_all` | the submitted Kaggle config: submit mode, medium, 48k / 24k, context 131072 (+ `--record-train-check --record-alternatives`, bookkeeping only) | 167 x 1 | done |
| `K1_all` | K0 + `--turn-think-cap 8192` | 167 x 1 | done |
| `K2_all` | K1 + program + `lean2progroles` + summary + helpers + rich feedback (medium, 48k / 24k) | 167 x 2 | done |
| `K2nc_all` | K2 without the think cap | 167 x 1 | done |
| `K3_all` | K2 at xhigh (48k / 24k) | 167 x 2 | done |
| `smoke_local`, `slots_smoke_local` | local AWQ server smoke tests | 4, 3 | done |

**Resume** (needs the boxes, gateway/tunnels and `arc-sandbox-e019` back; each command continues its file; finished attempts are skipped, errored ones re-run). Slot caps live in
`experiments/E019_qwen_program_answers/router/slots.json` (static 10 per box once nothing else runs; a cap
controller, `cap_controller.py --state ... --other-state ...`, keeps two slot files under 10 per box together):
```bash
cd /home/vu/Workspace/ARC-AGI-2
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh x96pal all 1 20
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh x96pex all 1 15
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh x96pc8 all 2 20 --only-attempt 1
# any K arm, e.g. a 3rd K3 sample:  SLOTS=auto:10 .../launch.sh K3 all 3 25 --only-attempt 2
```
`launch.sh` passes `--base-url` (gateway URL, unused with slots), `--session-mem-mb 2048`, `--request-timeout 1200`
and `--worker-slots`; none of these is part of the results config.

### Operations log (pool, incidents)
- 16:49-17:30: arms at 8+10 per arm through the gateway; E019 resumed 48 attempts → JP KV 0.94-0.99, hit 0.66, 1
  retraction on AU. E019 cut back; then only E020.
- 18:58-19:42: two KV/thrash episodes (AU hit 0.04-0.46, KV 0.97-0.99, 1 retraction) caused by min_load piling
  long-context attempts on one box and by >44 xhigh-96k attempts in total. Stopped x96pex, then x96pal.
  AU's `/health` (a generation) timed out under load and the gateway marked it unhealthy twice (keys reassigned).
- 19:40: `x-smg-target-worker` found to be ignored by the gateway (manual policy) → harness-side worker slots
  (`evals/arc_harness/slots.py`, design in `../E019_qwen_program_answers/router/README.md`) and `cap_controller.py`.
- 20:15-21:00: two new boxes (AU2 :30104, CA :30105); caps of 10 per box (user: no KV offload, headroom first).
- 21:09-21:19: every run_qwen_tools process (command line contained "sglang") and the gateway were killed by another
  agent's `pkill -9 -f '[s]glang'` test. All runs resumed at 21:24 from their files, every process through the slot
  scheduler on direct worker URLs; `launch.sh` now passes `--base-url` (no "sglang" in command lines).
- 22:24-22:46: the AU/AU2 SSH tunnels hung twice; 20 attempts stalled ~15 min, retried by the client, 0 errors.
- 01:00: `evals/sandbox_client.py` closed sessions with a DELETE without `X-Session-ID`; nginx hashes on that header, so
  the DELETE hit a random worker (404, ignored) and every finished kernel lived until the 30-min idle reaper (~90
  leaked kernels, 14 GiB). Fixed (header added; verified) for processes started after 01:05.
- Sandbox: kernels of 8-10 GB from shard attempts started before `--session-mem-mb` existed; container raised to 32 GB by
  the coordinator; every later launch has the 2 GB session cap.

Local smoke (`qwen_tools_e020_smoke_local.jsonl`, AWQ server, xhigh 32k, all options, cap 4000): 4 attempts, the
nudge fired 4-5 times per attempt and the model closed its thinking and ran code immediately each time.

## Offline checks
- **Audit / lint flags on logged train-passing programs** (E019 dev program samples plus the first E020 records: 66
  programs, 41 correct and 25 wrong on test). After de-noising (a colour counts as new only if it is in no training
  output *and* not in test_input; cell indices need a value >= 2; neighbour-offset lists are exempt), the output flags
  fire on 0/41 correct and 0/25 wrong programs, lint on 4/41 and 1/25, and the "Outside train range" line on all 66.
  So the flags do not separate right from wrong; any gain from `--test-audit` must come from the review turn itself
  (the model sees its own test output once). Before de-noising, the colour flag fired on 17/41 vs 12/25.
- **H1, selection vs k on E019's samples** (dev-56, medium 48k; s0 56, s1 13, s2 6, s3 2 records): P(correct | train
  pass) 49-53% (DeepSeek: 84%), P(correct | no train pass) 0-5%. With train-pass ranking, pass@1 = pass@2 = oracle at
  k = 2-4, so selection is not the bottleneck; coverage is. Pooling program + baseline samples: pass@2 26.8% → 32.1%
  from k = 2 to 4 (+5.4 [0, +11.7]).

## Results

### Ceiling arms: xhigh, 96k hard / 48k soft, program mode (full set; 166 common pairs)
| arm | pass@1 pair-w | pass@1 task-w | tokens/attempt mean | cap stops | train-pass rate | P(correct \| tp) | Δ vs no-tool xhigh-240k a0 [95% CI] |
|---|---|---|---|---|---|---|---|
| no-tool xhigh 240k (E008, attempt 0) | 34.9% | 36.1% | 74.3k | 0% | — | — | — |
| x96p (program + roles + summary + helpers + rich feedback) | 48.8% | 49.4% | 57.9k | 26% | 74% | 63% | +13.9 [+4.9, +22.8] |
| **x96pc8** (x96p + think cap 8000 / grace 2048) | **56.0%** | **58.1%** | 55.9k | 22% | 77% | 72% | **+21.1 [+12.1, +29.8]** |

x96pc8 − x96p: +7.2 [+0.6, +14.4] pair-weighted, +8.7 [+1.4, +16.1] task-weighted (wins 23 / 11): at xhigh-96k the
per-turn thinking cap helps (4.0 nudges per attempt, 0.02 forced).

### Cross-mode pooling (pass@2 with selection; `select_sim.py --ordered`, full set)
| candidates per pair | policy | pass@2 pair-w | pass@2 task-w | oracle | tokens/pair |
|---|---|---|---|---|---|
| no-tool xhigh ×2 | first2 | 47.9% | 48.2% | 47.9% | 147k |
| x96pc8 + x96p | tp | 62.3% | 63.6% | 62.3% | 114k |
| x96pc8 + no-tool a0 | tp | 64.1% | 65.8% | 64.1% | 130k |
| **x96pc8 + x96p + no-tool ×2** | **vote_tp** | **69.5%** | **70.7%** | 71.3% | 261k |
| + K3 + K2 (7 candidates) | vote_order | 69.5% | 70.7% | 72.5% | 335k |
| all logged x96 candidates (x96pc8 ×1-2, x96p, x96pal, x96pex; partial) + no-tool ×2 (median 5, max 7) | vote_tp | **71.3%** | **72.4%** | 74.3% | 312k |

Partial items 1-3 at xhigh-96k: x96pal − x96p on 122 pairs +0.8 [−7.1, +8.7] (task-w +2.3); the review turn was
shown in 91 attempts (91 had flags — the "outside train range" line always fires — 10 had lint flags) and changed
the answer once. x96pex: 8 pairs, no conclusion.

vote_tp = most votes over all candidates, ties by the number of train-passing programs, then fewer tokens. With
2 x96 samples, selection already reaches the oracle; mixing modes raises the oracle (different modes fail on
different pairs). Weak modes (48k) add little coverage and some noise votes.

### Kaggle-budget arms (1 attempt per pair, full set, 167 pairs; `compare_arms.py`, `budget_sim.py`)
| arm | pass@1 pair-w | pass@1 task-w | tokens/pair mean / median / p90 | cap stops | Δ pass@1 vs K0 [95% CI] (wins/losses) |
|---|---|---|---|---|---|
| K0 = submitted Kaggle config (medium, 48k / 24k, submit) | 26.9% | 28.3% | 36.7k / 44.0k / 48.2k | 38% | — |
| K1 = K0 + think cap 8192 | 27.5% | 29.2% | 34.9k / 41.3k / 48.2k | 25% | +0.6 [−6.2, +7.4] (16/15) |
| K2 = K1 + program + roles + summary + helpers + rich feedback (medium, 48k) | 33.5% | 35.6% | 34.3k / 42.6k / 48.0k | 46% | +6.6 [−0.6, +14.3] (27/16) |
| K2nc = K2 without the think cap | 35.3% | 35.4% | 34.6k / 41.2k / 48.0k | 47% | +8.4 [+1.2, +15.9] (29/15) |
| **K3 = K2 at xhigh (48k / 24k)** | **37.1%** | **39.0%** | 38.5k / 48.0k / 48.0k | 51% | **+10.2 [+3.0, +17.4] (28/11)** |
| E012 attempt 0 (same config, older run) | 25.7% | 27.5% | 35.4k | 35% | −1.2 [−6.9, +4.6] |
| no-tool xhigh 240k attempt 0 (reference) | 34.7% | 35.8% | 74.2k | — | +7.8 [−0.6, +15.9] |

- **K2 with two real samples** (`two_sample.py`; pair-w / task-w): 2-sample mean pass@1 32.0% / 33.7% (samples 33.5%
  and 30.5%); free pass@2 34.1% / 36.0% at 34.3k; **gated pass@2 at a 44k mean 37.1% / 39.3%** (37 of 81 eligible
  pairs retried; 43.6k tokens/pair, p90 95k; +3.0 [+0.6, +5.9]); at 64k 40.7% / 42.2% (all 81; 55.5k); always 2
  attempts 43.1% / 44.3% at 69.6k. In program mode the gate is real: only pairs without any train-passing grid (49%)
  are eligible.
- **K3 with two real samples:** 2-sample mean pass@1 37.4% / 38.6% (38.1k tokens per attempt); free pass@2 37.1% /
  39.0%; gated pass@2 at a 44k mean 38.3% / 39.9% (19 of 83 eligible retried; 43.9k/pair); at 64k 43.1% / 44.0%
  (61.7k); always 2 attempts 45.5% / 46.1% (76.1k). K3 − K2 2-sample pass@1 **+5.4 [+0.9, +10.2]** (task-w +4.9
  [0, +10.2]); at a 44k mean the gated pass@2 is a tie (K3 38.3% vs K2 37.1%), at 64k K3 leads by 2.4 pp.
- **Think cap at medium:** K2 − K2nc = −1.8 [−10.1, +6.2]; the cap helps at xhigh-96k (+7.2) but not at medium-48k
  (K1 − K0 +0.6, K2 − K2nc −1.8). The gain of K2 comes from program mode + summary + helpers (K2nc − K0 +8.4).
- **Free attempt_2** (`--record-alternatives`; the session's `answer`, `solve(test_input)`, the salvage function;
  v3 `selection.select_two`): K0 26.9% → 28.1% pass@2 (+1.2 [0, +3.0]); K1 +0. Only 12/113 attempts had a distinct
  alternative. `select_two` never dropped a correct submitted answer.
- **Gated 2nd attempt:** 89% of K0 pairs qualify (81/167 answers cannot be train-checked), so "gated" ≈ "always":
  +9.0 pp pass@2 [+4.7, +13.6] at 68.9k tokens/pair with K1 as the retry sample (+6.6 with E012 a1).
  Budget-matched (v3 `retry_priority`, retries budgeted at 40k each): at a 44k mean (~30 retries) +0 to +0.6 pp with
  the v3 order (0 = fails train, 1 = unknown, 2 = none/fallback) and +1.2 to +2.4 pp with (0, 2, 1); at a 64k mean
  +4.8 to +6.0 pp [+1.8, +10.4]. Retry yields on K0 a0-wrong pairs: category 1 11%, category 2 16% (K1 retry) / 8%
  (E012 a1 retry).
- **Think-cap nudge parsing:** 1 of 51 K1 nudges (medium) produced a `<tool_call>` inside open thinking that ended
  the attempt; 0 of 501 at xhigh. `--think-call-rescue` (opt-in) executes such a call.
