# E019: program-only answers for the Qwen tool harness

## Goal
Every answer must be a checkable program. The model hands in a Python `solve(grid) -> grid`. The harness runs it on
every training input, compares exactly with the training outputs, and only then runs it on the test input. Typed
grids and `submit_answer(expr)` are not accepted.

Motivation (E012, Qwen tools, medium 48k/24k): `submit_answer` was used in 66% of attempts (37% correct), 19% were
forced at the cap (0% correct) and 14% had no answer (0%); full-set pass@1 24.6% (pair-weighted). DeepSeek's code
mode (E004 code-high) submitted a train-checked `transform()` in 71% of attempts, 86% correct.

## Design (harness `qwen-tools-4`, opt-in `--answer-mode program`)
Code: `evals/arc_harness/agent_qwen.py` (agent loop), `evals/arc_harness/executor.py` (`run_program`,
`run_session_solve`, `score_outputs`, `program_feedback`), runner `evals/run_qwen_tools.py`, tests
`evals/tests/test_qwen_program.py` (19 tests incl. the routing header over HTTP; 42 with `test_qwen_tools.py`; no server or sandbox needed).
Kaggle: `kaggle/src/solver.py` reads `ARC_TOOLS_ANSWER_MODE=program` (default `submit`, unchanged).

- **Defaults unchanged.** `answer_mode="submit"` behaves exactly like `qwen-tools-3` (all 23 old tests and the
  Kaggle fake-tools test pass). The version bump adds bookkeeping only: `usage.prompt_total`, `usage.ideal_cached`,
  `usage.cached` (when the server reports `cached_tokens`), and, with `--record-train-check`, a host-side train check
  of the submitted expression after the attempt (E011 `check_answer_on_train`; no tokens, the model never sees it).
  The config records `answer_mode` (and the program options when it is `program`).
- **Tools.** `run_python` (persistent session, unchanged) and `submit_program(code)` in place of `submit_answer`.
- **Check (host-side, no model tokens).** `run_program` starts a fresh `python -I` process from the attempt's
  executor (inside the Docker sandbox here; `LocalExecutor` on Kaggle), so session state cannot leak in. It execs
  the code with `np` preloaded, calls `solve(grid)` with a SIGALRM limit of 10 s per grid and a 3 GB address-space
  limit, and returns the output (numpy converted) or the error with the last two program frames. One process per
  grid (~0.9 s each in the sandbox, mostly the scipy import). The host compares the outputs with the training
  outputs; the test output never enters the sandbox for comparison.
  - **All train pairs pass:** run on `test_input`; a valid grid (rectangular, 1-30 per side, ints 0-9) is the
    answer and ends the attempt (`program_pass`). An invalid grid or an exception on the test input is a rejection
    ("passes all N training pairs but failed on test_input: ...").
  - **Any fail:** the tool result lists every pair (OK / error / wrong shape got vs expected / N wrong cells), the
    traceback tail of the first failing pair or its first 12 wrong cells `(r,c) expected e got g`, and
    "submission k of 6". The budget note follows, and the attempt continues.
  - At most 6 submissions per attempt; the 6th rejection ends the attempt (`stop_reason=submission_cap`).
- **Budget logic** from qwen-tools-3 (notes on every tool output, soft target, hard cap); the notes name
  `submit_program` ("or at least keep `solve` defined in your session").
- **No typed answer anywhere.** At the hard cap, the context limit or unclosed thinking there is no forced turn. A
  final text reply gets a nudge ("Typed answers are not accepted. Submit your answer as a program ...") at most
  twice, then the attempt ends. Fallbacks, also after an exception: the best submitted program (highest train
  fraction, then latest) and the session's own `solve` (train-checked the same way, run in the session so it may use
  session helpers), whichever reproduces more training pairs; ties go to the session `solve` (the latest code). The
  first candidate with a valid test output is the answer. Otherwise no answer.
- **Recorded per attempt:** `answer_source` (program_pass / program_best_fail / session_solve / None),
  `train_pass`, `train_frac`, `n_submissions`, `program` (the answer's code; for `session_solve` the last
  `run_python` cell that defined `solve`), `submissions` (per submission: pairs passed, failure kinds, test error,
  turn, tokens used, code length), `nudges`, plus everything qwen-tools-3 records.

### Prompt and tool text (program mode, prompt `lean2prog` = `lean2` + this)
> Your answer must be a program: typed grids are not accepted. When your code reproduces every training output and
> its assumptions hold on test_input, call submit_program with self-contained code that defines `solve(grid)`. The
> harness runs it on every training input and checks the outputs exactly before it runs it on the test input; if a
> pair fails it tells you what went wrong and you can fix the program and submit again. Keep your latest `solve`
> defined in your session too: if you run out of budget, the best program you submitted (or that `solve`) is used.

`submit_program` description:
> Submit your answer as a program; this is the only way to answer (typed grids are not accepted). `code` is
> self-contained Python source that defines solve(grid): grid is a list of lists of ints, and solve returns the
> output grid (list of lists or numpy array of ints 0-9). It runs in a fresh Python process: numpy is preloaded as
> np, but include every other import and helper function (your session variables are not available). The harness
> first runs solve on every training input and compares the results exactly with the training outputs. If any pair
> fails, you get feedback (failing pairs, shapes, wrong cells, errors) and the attempt continues. If all pass, it runs
> solve on the test input and that grid is your answer, which ends the attempt. Limit 10s per grid and 6 submissions
> per attempt.

## Servers
- **Local (smoke tests only):** container `qwen38-sglang-radix` replaces `qwen38-sglang`, which is stopped but kept for
  rollback. Same image (`vastai/sglang:v0.5.14-cuda-13.0`), model (`Qwen3.8-27B-AWQ-INT4-sglfix`, served as
  `qwen3.8-27b`), mounts, GPUs (2×RTX 3090, `--tp-size 2`), env and flags, minus `--disable-radix-cache`, plus
  `--enable-cache-report --enable-metrics`.
  - With radix on, SGLang's `extra_buffer` mamba strategy needs about 2 mamba slots per request. It allocated 65
    slots and cut the KV pool from 499k to 344k tokens, and max running requests from 32 to 13. That is enough for
    smoke tests.
  - Cache hit over the smoke runs: 0.955 server-side (`evals/results/cache_local_radix.jsonl`). Per attempt it
    equals the ideal hit (the previous prompt plus its completion), from 0.54 on a 2-turn attempt to 0.97 on 40k
    attempts. KV usage stayed ≤ 0.43, with 0 retractions.
  - It is left running.
- **Eval runs:** the SGLang Model Gateway over 3 single-GPU Vast boxes, set up by another agent (`router/README.md`).
  - AU and JP are RTX PRO 6000 Max-Q; TX joined at 16:00 UTC.
  - Each box serves `nvidia/Qwen3.8-27B-NVFP4` on `vastai/sglang:v0.5.20`, radix on, KV fp8, 262k context, ~939k KV
    tokens and 35 running slots.
  - This is the same checkpoint and SGLang version as E012, which ran on 4×RTX PRO 6000 TP. So these runs can be
    read against E012's dev numbers (below).
  - Manual routing: every request carries `x-smg-routing-key: <run>:<task>:<pair>:<attempt>`
    (`--routing-key-header`), so an attempt stays on one worker, and min_load places new keys.
  - Per-worker cache health comes from the boxes' SGLang logs over ssh every 5 min (`worker_cache_log.py` →
    `evals/results/cache_gw_dev56.jsonl`), because the workers have no `/metrics`.
  - Sandbox: a dedicated `arc-sandbox-e019` (24 GB; `NAME=arc-sandbox-e019 SANDBOX_MEMORY=24g
    evals/sandbox_docker/start.sh`). At ~90 sessions it used 7.7 GB and under 2% CPU.

### Throughput ramp (how much one RTX PRO 6000 does for this tool-use workload)
The table gives means over 5-min log windows. "Running" is SGLang's `#running-req`; attempts waiting on a tool call
are not counted. Sample processes were added one per arm per step at concurrency 12. Windows after 16:52 include
E020's load.

| step (UTC) | in flight (gateway) | running per box | total gen tok/s | per box | per-stream tok/s | window hit | KV peak | queue / retractions |
|---|---|---|---|---|---|---|---|---|
| s0 on 2 boxes (15:44–15:54) | 40 | 20 | 1.21–1.38k | 600–700 | ~30–34 | 0.94–0.98 | 0.75 (AU), 0.89 (JP at 16:00) | 0 / 0 |
| s0 on 3 boxes (16:10–16:15) | 40 | ~13.3 | 1.39–1.42k | 420–490 | ~35 | 0.95–0.98 | ≤ 0.57 | 0 / 0 |
| + s1 (16:20–16:26) | 64 | ~20.7 | 1.82–2.06k | 570–800 | ~30 | 0.91–0.98 | ≤ 0.70 | 0 / 0 |
| + s2, s0 draining (16:31–16:42) | 54–62 | 17–21 | 1.65–1.87k | 520–665 | ~30 | 0.93–0.99 | ≤ 0.78 | 0 / 0 |
| + s3 (16:47) | ~74 | ~24.9 | 1.99k | 610–710 | ~27 | 0.91–0.94 | **0.92 (TX)** | 0 / 0 |
| + s3 + E020 start (16:52) | ~89 | ~27.9 | 2.22k | 715–755 | ~26.5 | 0.93–0.95 | 0.90 (TX), 0.86 (JP) | 0 / 0 |

- **The knee is about 20–21 running per box: ~600–630 gen tok/s per RTX PRO 6000 (Max-Q / WS)**, with hit ≥ 0.93
  and KV ≤ 0.8.
  - From ~13 to ~21 running, throughput rose 27% while per-stream speed fell from 35 to 30 tok/s.
  - Beyond that each step added under 10% (+6.5% at ~25 running), and KV reached 0.90–0.92 on TX with no
    retractions yet.
  - s3 was stopped at that point. The guard also stopped s1/s2 at 16:54 on the 0.90 KV snapshot; they were resumed
    at 17:08.
- **Cache hit stayed high throughout:** the mean over the runs is 0.95 per box (AU 0.953 over 28M prompt tokens, JP
  0.946, TX 0.947). The only low windows were the first one after a box received fresh attempts (0.71–0.72, when
  every prompt is new). There were 0 real retractions; the one retraction line on TX was its `server_args` startup
  line.
- The ideal hit computed by the harness (`usage.ideal_cached`, everything sent before is a prefix) is 0.958 (A) and
  0.968 (B), so the servers lost almost nothing to eviction.
- Single-stream decode on an idle box: 43–54 tok/s (`router/README.md`).

## Runs (dev-56, the E009/E012 dev pairs `experiments/E009_tool_token_efficiency/dev_pairs.txt`, 56 pairs / 49 tasks)
- **Identical settings in both arms:** tier medium, 48k hard / 24k soft, gentle wording, budget notes on every tool
  output, temperature 1.0 / top_p 0.95 / top_k 20, 16k chunks, 60 turns, context 262k, 1 attempt per pair per sample.
- The arms ran concurrently on the same gateway, so both saw the same mix of workers.
- **A, baseline:** `qwen-tools-4` with `answer_mode=submit` (the `qwen-tools-3` behaviour) plus
  `--record-train-check`.
- **B, program-only:** `--answer-mode program`.
- **Files:** `evals/results/qwen_tools_{baseline,program}_gw_dev56{,_s1,_s2,_s3}.jsonl`, logs `log_…`, traces
  `…_traces/`.
  - s0 is attempt index 0, the primary paired comparison.
  - s1 and s2 are extra samples (attempt index 1 and 2) for pass@2 and multi-sample means.
  - s3 was stopped during the ramp; it resumes from its file.
- **Smoke:** `evals/results/qwen_tools_{program,baseline}_smoke*_local.jsonl` on the local AWQ server.
  - program mode: 4/4 easy pairs correct, then 2/4 on harder dev pairs;
  - baseline: 2/2.
- **Launch (one process per arm and sample; resumable):**

  ```bash
  ARC_SANDBOX_URL=http://172.18.0.3:6000 .venv/bin/python evals/run_qwen_tools.py --endpoint local-sglang-router-3x6000 \
    --model nvidia/Qwen3.8-27B-NVFP4 --routing-key-header x-smg-routing-key \
    --pairs-file experiments/E009_tool_token_efficiency/dev_pairs.txt --tier medium --hard 48000 --soft 24000 \
    --attempts 1 --concurrency 20 --answer-mode program --out evals/results/qwen_tools_program_gw_dev56.jsonl
  # baseline: replace --answer-mode program by --record-train-check (and the --out name);
  # extra samples: --attempts 2 --only-attempt 1 --run-name qwen_tools_program_gw_dev56 --out ..._s1.jsonl
  ```
- **Stats:** `compare.py A=<file> B=<file>` (one sample, paired) and `multi.py A=f1,f2 B=f1,f2` (multi-sample).
- **Cost:** $0 API. Vast box time is shared with E020 (3 boxes at ~$1.06–1.09/h).

### s0: paired result (attempt 0 in both arms)
| arm | pass@1 task-weighted | pass@1 pair-weighted | correct | tokens mean / median / correct-median | stops at the cap | submissions / attempt |
|---|---|---|---|---|---|---|
| A submit_answer | 22.4% | 21.4% | 12/56 | 36.7k / 39.3k / 25.5k | 19 (34%) | – |
| B program-only | 21.4% | 19.6% | 11/56 | 39.5k / 48.0k / 28.9k | 31 (55%) | 0.84 |

- **B − A pass@1:** **−1.0 points task-weighted [−11.2, +9.2]** (95% bootstrap over tasks); −1.8 pair-weighted.
  Discordant pairs: 4 solved only by A, 3 only by B. **This is a tie.**
- **E012 reference:** E012's dev attempt 0 (same NVFP4 checkpoint and SGLang, qwen-tools-2) also scored 22.4% /
  21.4% at 36.6k mean tokens, so arm A reproduces E012. B − E012 is −1.0 [−13.3, +10.2].

**Answer sources**

| arm | source | attempts | correct | accuracy | tokens median |
|---|---|---|---|---|---|
| A | submit | 37 (66%) | 12 | 32.4% | 31.2k |
| A | forced (typed at the cap) | 8 (14%) | 0 | 0% | 48.0k |
| A | none | 10 (18%) | 0 | 0% | 48.3k |
| A | salvage | 1 (2%) | 0 | 0% | 48.3k |
| B | program_pass | 22 (39%) | 9 | 40.9% | 27.2k |
| B | session_solve | 8 (14%) | 1 | 12.5% | 48.0k |
| B | program_best_fail | 5 (9%) | 1 | 20.0% | 48.0k |
| B | none | 21 (38%) | 0 | 0% | 48.0k |

- **Train check, B:** P(correct | train pass) = 9/22 = **41%**; P(correct | train fail) = 2/13 = 15%. Both correct
  train-fail answers were fallbacks, at train fractions 0.67 and 0.4.
- **Train check, A:** P(correct | pass) = 5/8 and P(correct | fail) = 0/7. Only 15 of A's 46 answers can be checked,
  because the other 31 submitted `answer` or a literal rather than an expression that reads `test_input`; 7 of those
  31 were correct.
- **Submissions, B:**
  - per attempt: 0 in 26, 1 in 24, 2 in 2, 3 in 1, 4 in 1, 6 in 2 (2 hit the submission cap);
  - rejected submissions: 18 wrong cells, 4 wrong shape, 2 NameError (a helper missing from the self-contained code);
  - 1 passed train but raised on the test input;
  - 4 nudges after text replies.
- **Tokens at the first submission:** median ~29k. 21 of the 22 program_pass attempts passed on their first
  submission; the train-check feedback loop rescued one attempt.
- **Health:** 0 harness errors, 0 tool-call parse failures, 0 executor resets in either arm. Tool errors per attempt
  were 0.73 (A) and 0.98 (B); B's count includes rejected submissions.

### Failure patterns (trace review of s0 B: all 13 train-passing-but-wrong programs, plus 4 cap hits)
**Train-passing programs that were wrong on test (13 of 22; each made exactly one submission and stopped)**

| task / pair | test diff | category | reason |
|---|---|---|---|
| 221dfab4/0 | 118/625 cells | overfit (absolute position) | `r % 6` phase taken from row 0; matched train by coincidence, never looked at test |
| 409aa875/0 | 5/256 | overfit (lookup table) | shape→offset table keyed on exact cell sets plus a default; noted "test shapes … uncertain", submitted at 38k after the budget nudge |
| a395ee82/0 | 63/484 | overfit (offset) | hardcoded anchor cell; noticed the "stray" test marker, then "trust the code passed training" |
| dbff022c/0 | 15/169 | overfit (orientation) | key read row 0 → row 1; the test key is on the bottom edge |
| 78332cb0/1 | 5x23 vs 23x5 | overfit / wrong rule | layout lookup with a hardcoded order; the test stack stays vertical |
| 247ef758/1 | 1/216 | test-only situation | shapes overlap only in the test; wrong paint order |
| d35bdbdc/0 | 15/100 | test-only situation | a solid plus is not detected as a box (the detector found 0 boxes on test) |
| d35bdbdc/2 | 1/100 | test-only edge case | box cut by the border not detected |
| 13e47133/1 | 16/900 | test-only geometry | BFS-distance rings; the concave test region needs rectangular rings |
| 1ae2feb7/0 | 2/200 | near miss | colour placed at index = count instead of every multiple; only the test row shows it |
| f931b4a8/1 | 4/16 | near miss | fill index `(i//h)%h` happens to match train |
| 581f7754/0 | 26/280 | ad-hoc patch | special case added just to fit train; the test has anchors on both axes |
| 7b80bb43/0 | 7/493 | ad-hoc patch | magic constants fitted to 2 train pairs; submitted at 47k |

- **Categories:** overfit to train specifics 5, test-only situation 4, near miss 2, ad-hoc patch 2. Seven of the 13
  are within 16 wrong cells.
- **The prompt's test-assumption check was mostly skipped:**
  - 4 of the 13 never ran code on `test_input`;
  - 3 ran `solve(test_input)` and submitted without inspecting the result;
  - twice the model saw the problem and submitted anyway, under budget pressure.
- With 2-5 training pairs, a train-exact program is a weak filter for Qwen. E004's DeepSeek code mode reached 86%
  correct given a train pass; here it is 41%. Programs are often lookup tables or patched special cases that fit
  train by construction.

**Cap hits (31 of 56 B attempts; 26 never submitted)**
- 19 of the 21 no-answer attempts never wrote a `def solve` at all. Mostly they explored with probe functions and
  never converged on a rule, so neither fallback had anything to run.
- 15 of the 31 have a thinking turn of 15k tokens or more, and 12 were cut inside unclosed thinking. 37% of
  cap-hit tokens are in turns of 10k tokens or more; budget notes only arrive with tool outputs, so a long turn never
  sees one.
- Examples:
  - b99e7126/0: one 38k-token turn enumerating hypotheses by hand.
  - 9bbf930d/0: 77 grid rows copied by hand.
  - 8b9c3697/0: a plausible rule at turn 15, then a 23.6k turn re-deriving it, cut off.
  - d35bdbdc/1: 31 short probes that never converge, while pairs 0 and 2 of the same task found the rule.
- On the 21 pairs where B had no answer, A was correct only once (b99e7126, where both arms spent over 38k on one
  thinking turn). So requiring a program cost almost nothing on the pairs that ran out of budget. The cost of
  program mode is the extra ~8 points of cap hits (B's first submission comes late), offset by A's typed or forced
  answers being 0% correct.

**Contrasts with A**
- 247ef758/1 and a395ee82/0: A was right. A's trace found the general rule (a "don't overwrite" stamp; the marker
  dot as anchor), while B's program hardcoded the train case.
- dd6b8c4b/1 and 136b0064/0: B was right. B took short turns (18k tokens); A spent one huge thinking turn and hit the
  cap.

### Extra samples (partial; s1–s3 stopped to give the pool to E020)
| file | attempts finished | correct | sources (correct / n) |
|---|---|---|---|
| baseline s1 | 17 | 5 | submit 5/10, forced 0/2, none 0/5 |
| program s1 | 16 | 6 | program_pass 6/10, best_fail 0/1, none 0/5 |
| baseline s2 | 5 | 3 | submit 3/5 |
| program s2 | 7 | 5 | program_pass 5/6, none 0/1 |
| program s3 | 2 | 0 | program_pass 0/1, none 0/1 |

- **Not representative:** the finished extra samples are biased toward fast attempts, because the attempts killed
  in flight were the long ones (median tokens: 28k for A's extras, 20k for B's). Pass@2 over the 56 pairs is
  therefore not available.
- On the 16 pairs with an s1 attempt in both arms: A 5, B 6 correct.
- Pooling every finished attempt (unpaired, biased): A 20/78 (25.6%), B 22/81 (27.2%).
- All of this agrees with the s0 tie. Resume (identical settings, own routing keys) with the s1/s2/s3 commands
  above (`--attempts k+1 --only-attempt k`).

### Lesson: size concurrency by the long tail, not by early windows
- The knee above (~20–21 running per box) was measured while most attempts were young.
- At 17:20–17:35 the same ~20 per box, now late in their lives with 40–80k contexts (and sharing the pool with
  E020), pushed every box to 0.97–1.00 KV. That gave queues of 1–3, one retraction on AU, and JP's hit fell to
  0.66–0.84.
- **Rule:** concurrency per box ≈ KV tokens ÷ p90 context per attempt, about 939k ÷ 45–80k ≈ **12–20 per box**,
  not the early-run windows. The attempt context grows to 50–65k at p50–p90 (E012); the working set is the sum over
  live attempts, including those waiting on a tool call.

## Conclusions
1. **Program-only answers tie with submit_answer on accuracy:** B − A = −1.0 points pass@1 [−11.2, +9.2] on dev-56,
   at +2.8k mean tokens per attempt.
   - Under the user's rule this is a tie, not "worse" or "better".
   - It removes every typed and forced answer, which were 0% correct in A anyway.
2. **Every B answer is checkable,** but the train check is a weak filter for Qwen: P(correct | train pass) = 41%,
   versus 86% for DeepSeek in E004.
   - 13 of 22 train-passing programs failed on test: lookup tables, absolute positions and ad-hoc patches, or test
     situations the train pairs never show.
   - Seven of them were within 16 cells.
3. **Program mode moves answers later:** the median first submission comes at ~29k tokens, 55% of attempts hit the
   cap (A: 34%), and 26 of 56 never submitted.
   - Most of those never wrote a `solve` at all, so the fallbacks rarely help: session_solve was 1/8 and
     program_best_fail 1/5.
   - On those pairs A's typed answers were almost never right either (1 of 21).
4. **Prefix caching works through the gateway.** Hit rate was 0.95 per box (ideal 0.96–0.97) with manual routing
   keys. Throughput is ~600–630 gen tok/s per RTX PRO 6000 at ~20 running.

## Recommendation for the Kaggle harness
- **Keep `qwen-tools-3` behaviour (`answer_mode=submit`) as the Kaggle default for now.** Program-only does not raise
  accuracy and costs ~8% more tokens. Under Kaggle's budget scheduler, more cap hits mean fewer finished attempts.
- **If checkable answers are required,** `ARC_TOOLS_ANSWER_MODE=program` is safe to use: accuracy is tied, and the
  CI is wide on 56 pairs.
- **Before program mode can beat A, it needs:**
  1. A test audit when a program passes train: return the test output plus "outside train range" features (new
     colours, sizes, object counts, objects the detectors missed; E017's task summary). Then allow one
     confirm-or-revise turn. That targets 221dfab4, a395ee82, d35bdbdc ×2, dbff022c and 247ef758.
  2. A lint for hardcoded literals: coordinates, absolute modulo phases, exact-shape lookup tables, magic
     thresholds. Ask for a justification (409aa875, 221dfab4, 7b80bb43, 581f7754).
  3. A per-turn thinking cap (~8k, then force a tool call), plus a draft `solve` required in the session by ~16k
     tokens. That attacks the 55% cap hits and gives the fallbacks something to run.
  4. For pass@2: a second program that also passes train but differs on test (anchor-relative vs absolute, paint
     order), ranked by train pass (`evals/train_check.py:ranked_answers` logic).
- **Serving on Kaggle (or any box):** size tool-use concurrency by KV ÷ p90 context (above), and keep radix and
  prefix caching on with per-attempt affinity.

## State at the end
- **Local:** `qwen38-sglang-radix` (radix on) is live on localhost:30000. `qwen38-sglang` is stopped, not removed.
- **Gateway:** E019's runs are stopped and resumable. E020 is using the same gateway and `arc-sandbox-e019` (left
  running). The gateway cache logger (`worker_cache_log.py`) keeps appending to `evals/results/cache_gw_dev56.jsonl`
  for E020; stop it with `pkill -f worker_cache_log.py` when no longer needed.
