# E018: external Qwen3.8-27B Kaggle notebook (madarshbb) vs our submission

Read-only analysis, 2026-10-12. Nothing was run on Kaggle or on a GPU. Cost $0.

## Source
- Notebook: https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent ("ARC Qwen 3.8 27B agent", author Adarsh), competition `arc-prize-2026-arc-agi-2`.
- Pulled version: **v2** (the current one; API `current_version_number` 2). The API refuses v1 (403).
  - Last run time from the API: 2026-10-09. Kernel status at pull time: `CANCEL_ACKNOWLEDGED`.
  - The notebook's md5 is `2a669778…`.
- `vendor/madarshbb_v2/`: the notebook and `kernel-metadata.json`.
- `vendor/madarshbb_v2_output/`: the public output (500 KB):
  - the log;
  - `submission.json` (the fallback);
  - the `%%writefile` modules.
  - These are untrusted files. Never run them, and never start Python from that directory.
- Attached inputs (from `kernel-metadata.json`):
  - Model: `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`. A third-party "repacked" FP8 copy whose hashes nobody has checked.
  - Datasets:
    - `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`: vLLM 0.19.0 and torch 2.10.0. These are the same wheels we hash-checked in E013.
    - `keithtyser/taaf-duck-qwen38-serving-v1`: overlay wheels. The notebook deletes and reinstalls `transformers` and `huggingface_hub` from them.
  - Machine and environment: `NvidiaL4` (4×L4), internet **off**, a pinned Python 3.12 BYOD image.
- Scripts used for this analysis (they read only our own files): `scripts/mix_sim.py`, `scripts/loop_scan.py`, `scripts/grid_format_tokens.py`.

**[F]** marks a fact read in the notebook or its output. **[I]** marks my inference.

## Score claim ("26.81 LB with Qwen 3.8 27B, 8.61 with Qwen Flash, 80% on the 120 validation tasks")
- **[F] This notebook is not the source of the claim, and it states no score anywhere.** I checked:
  - the markdown cell (only the config);
  - every code comment;
  - the kernel metadata;
  - the Kaggle page (it shows only the title when fetched);
  - the output log.
  - The search for "26.81", "8.61", "80%" and "accuracy" found nothing.
- **[F] The public log is from the save run only.** The log says: `not a competition rerun: fallback submission only, agent skipped` and `0 test outputs answered by the agent`.
  - The scored rerun's logs are never public, so the notebook shows no throughput or tok/s numbers either.
- **[F] How the notebook would compute local accuracy:** `run_direct.py main()` (not called anywhere in the submission path).
  - It loads `arc-agi_evaluation_{challenges,solutions}.json` and runs `samples` (default 2) direct answers per task on `cfg["task_ids"]`.
  - It scores pass@2 **task-weighted**: per task, the fraction of test outputs matched by either sample. The result is printed as `DIRECT SCORE x / n_tasks`.
  - The solutions are used only for scoring, never to choose answers. In direct mode the model has no tools, so it cannot read the solutions file.
- **Nothing in the code inflates the score by itself.** I see four ways a number like 80% could still come out of this kind of setup:
  1. **[I]** `task_ids` can be a subset (for example only the tasks that finished, or an easy subset). The printed score is out of `len(ids)`, not out of 120.
  2. **[F]** The REPL agent's `python` tool is a plain subprocess with the full environment and filesystem, and no isolation besides an address-space rlimit.
     - **[I]** During a local eval on Kaggle, `/kaggle/input/**/arc-agi_evaluation_solutions.json` is readable from model-written code. No REPL eval script is included, but any local tool-agent number from this harness could be contaminated this way.
     - Our sandbox has no host mounts, and our Kaggle `LocalExecutor` gets only the task data. Still, our local tool runs on Kaggle could in principle read the solutions file too.
  3. **[I]** "Validation" may mean training tasks (1000 tasks, many carried over from ARC-1, see E015), or pass@k with large k, or the verified rate of the REPL agent ("verified" means all train pairs pass, not that the test is correct). E017 found P(correct | train pass) of only 84–86%, and verified rates run well above test accuracy.
  4. **[I]** Memorization of the public eval set. Our own measurements argue against a large effect:
     - E008 no-tool xhigh with a 240k budget: pass@2 47.3%, pair-weighted.
     - E012 tools at 48k: 34.6%.
     - ARC Prize reports 42.4% semi-private for Qwen3.8-27B.
- **[I] Conclusion:** 80% on the public eval is not credible for Qwen3.8-27B without fine-tuning. It is 33 points above our best measurement with the same model and sampling, and above DeepSeek V4.1-Flash at max effort (75.8%).
  - A 26.81 LB score is plausible and roughly what we expect ourselves.
    - A direct xhigh sample at about 48k scores ~21% pass@1, and two samples score 29% pass@2 (simulated from E008).
    - Our tool agent scores ~25% pass@1 at 36k.
  - The gap between "80%" and 26.81 most likely comes from items 1–3, not from the leaderboard set being much harder.

## What the notebook actually submits
**[F]** The config in cell 12 is `"mode": "direct"`, so the REPL tool agent in the notebook title is **not used**. The submission is plain reasoning without tools:
- Qwen3.8 at `reasoning_effort` xhigh.
- **2 samples per task.** All test inputs of a task go in one prompt, and the last n fenced grids are parsed.
- Samples run in sample-major order: every task's sample 0 runs before any sample 1.
- Tasks are sorted by prompt length, shortest first.

## Side-by-side comparison
| Axis | madarshbb v2 (submitted path) | Ours (E013 v2 `arc2-qwen38-tools-submit`) |
|---|---|---|
| Model | Qwen3.8-27B FP8, third-party "repacked" copy, not verified **[F]** | Official `Qwen/Qwen3.8-27B-FP8` rev 017b9c7, hash-verified mirror |
| Engine | vLLM 0.19.0 from a third-party wheelhouse plus a `transformers` overlay; OpenAI **chat** API with server-side `qwen3` reasoning parser and `qwen3_coder` tool parser **[F]** | vLLM 0.23.0 from our own wheel source; raw `/v1/completions` with the template rendered by hand; our own tool-call parsing |
| Layout | TP=4, no DP, one server **[F]** | TP=4, one server |
| Prefix caching | on (`--enable-prefix-caching`) **[F]**; barely matters for single-turn direct mode **[I]** | on (mandatory rule) |
| KV cache | fp8, `gpu-memory-utilization` 0.90, `max_model_len` 81,920 **[F]**; about 300k KV tokens (our 0.19 bench gave 304k at 0.95 with MTP) **[I]** | fp8 (`TRITON_ATTN`), util 0.95, `max_model_len` 131,072; **1.40M KV tokens** on 0.23 |
| Speculative decoding | MTP, `num_speculative_tokens` 2: method `qwen3_5_mtp` on the first ladder rung, `mtp` on the third **[F]** | **off** (`ARC_MTP` 0). E013 bench: MTP=3 raised decode by 16% (227 vs 195 tok/s at 16 sequences on vLLM 0.19) |
| Other server flags | `--language-model-only` (skips the vision tower), `--reasoning-config` (enables the server-side `thinking_token_budget`), `--generation-config vllm` **[F]** | `--trust-remote-code --no-async-scheduling` |
| Startup robustness | **Server option ladder**: 6 flag sets from most to least ambitious, ending with plain serving. On failure it kills the process group and waits for GPU memory to drain **[F]** | Same flags each time; one retry; a watchdog restarts a dead server |
| Concurrency | 10 client threads, `max_num_seqs` 16 **[F]** | 35 (KV / 40k), `max_num_seqs` 64 |
| Prompt | Their own short instructions (below); grids as digit rows with explicit `(RxC)` dimensions **[F]** | ARC Prize official prompt (JSON grids) plus `lean2sub`; tool descriptions; budget notes |
| Thinking tier | xhigh **[F]** | medium (E012: xhigh is worse at these budgets with tools) |
| Sampling | temperature 1.0, top_p 0.95, top_k 20, seed = 1000·sample + 7 **[F]** | temperature 1.0, top_p 0.95, top_k 20 |
| Tools | none in the submitted mode. The REPL agent (not used) has `python` (fresh subprocess per call with preloaded helpers) and `submit(code)`, which checks the program on train **[F]** | `run_python` (persistent session) and `submit_answer(expr)` |
| Answer extraction | last n digit-only fenced blocks (`grid` fence preferred) in `content` **[F]** | the session grid named by `submit_answer`; fallbacks: forced submit, session `answer`, parse, salvage |
| Per-call budget | `max_tokens` 49,152 with **server-side `thinking_token_budget` 40,960**: vLLM forces `</think>` after 40,960 thinking tokens, leaving ~8k for the answer **[F]** | hard cap min(48k, adaptive per-pair share), soft target 24k, budget notes on every tool output, warning at 75%, 16k chunks, 60 turns |
| Cut-off handling | if no grid is parsed: one **rescue** call with thinking off, effort low, 8k tokens, given the last 16k characters of its own notes and the order "give the final grid(s)" **[F]** | at the cap the harness closes thinking and opens a `submit_answer` call (256 tokens); an EOS mid-thought is continued up to 4 times |
| Attempts | 2 samples per task. `attempt_1` = first parsed answer; `attempt_2` = the next *different* answer, else a copy of the test input. No voting or train check (direct mode cannot check) **[F]** | 1 attempt per pair; `attempt_2` = copy of the test input; a 2nd attempt only with spare capacity |
| Time limits | deadline = start + 11h25m; no new sample within 10 min of it (`min_start_s` 600); request timeout = deadline + 60 s; the subprocess is killed at the deadline + 15 min; the last cell validates and repairs the file **[F]**. Requests still in flight at the deadline are lost, with no forced answer **[I]** | force answers at 11h00, hard stop 11h30, emergency writer 11h42, guard kills the process group 11h45, last cell validates and repairs |
| Rerun detection | `KAGGLE_IS_COMPETITION_RERUN`. The save run writes only the fallback and skips the agent (about 1 GPU-minute) **[F]** | the same variable, or test ids that differ from the public file. The commit run solves a local self-scored test set and fails the commit if that breaks (an end-to-end gate) |
| Fallback | test input copied into both attempts, written in cell 1; rewritten atomically after each answer **[F]** | the same, plus the emergency writer |
| Reported score / throughput | none in the notebook; `tok_per_s` prior 140 (used only to size tool-mode caps) **[F]** | public eval: tools 25.8% / 34.6% (pass@1 / pass@2, task-weighted). Kaggle bench: 227 tok/s decode, 295 tok/s end to end (vLLM 0.19, 16 sequences); 0.23 at concurrency 35 not yet measured under load |

The direct prompt (**[F]**, abridged):
- System: "You are an expert at solving ARC-AGI puzzles. Each puzzle has a few input->output example pairs that share one hidden transformation rule. Grids are 2D arrays of digits 0-9 (colours); 0 is usually the background."
- User: "Solve this ARC puzzle. … Work out the rule, check it against every example, then apply it to the test input(s). At the very end of your answer, give each test output grid as rows of digits inside its own ```grid code block…". This is followed by `## Example i`, `Input (RxC):` and digit rows.

## Throughput and token budget per test pair
- **[F] Theirs:** TP=4, fp8 KV, MTP with 2 tokens (if rung 0 or 2 starts), 10 concurrent requests, vLLM 0.19, about 300k KV.
- **[I] Their throughput is roughly 140–180 tok/s**, matching their own prior of 140. The basis is our 0.19 bench: 195–227 tok/s at 16 sequences, and decode on 4×L4 is memory-bound, so total tok/s grows with batch size.
- **[I] Their total supply is ~5.5–7M tokens** over ~11 h of generation:
  - With 240 tasks / 259 pairs (the public file), that is ~22–27k per pair. Since each sample can use up to 49k, only about half the tasks get even sample 0.
  - With a 120-task hidden set, it is ~45–55k per task: sample 0 everywhere and some sample-1s.
  - Either way, the second samples they plan for are mostly not reached.
- **Ours:**
  - About 295 tok/s end to end, about 11M tokens, about 44k per pair at 1 attempt (259 pairs).
  - vLLM 0.23 with 1.40M KV allows concurrency 35, so if decode keeps scaling, more is possible.
  - **[I] We already have about 2× their token supply.** Compared with this notebook, throughput is not where we lag.
  - Throughput is still our main lever for absolute score:
    - No-tool accuracy climbs steeply with budget: 21% at 48k, 28.5% at 64k, 34.5% at 96k, then flattens.
    - We currently waste `attempt_2`.
- **[I] The prompt matters less, but not zero.** Digit rows with dimensions take **2.5k tokens per pair against 7.0k** for JSON grids (0.36×, Qwen3.8 tokenizer, 172 eval test inputs; `scripts/grid_format_tokens.py`).
  - That is about 4.5k fewer KV tokens per live attempt, out of ~32k mean live KV (E014). It frees cache for concurrency.
  - Its effect on accuracy is untested.

## They do, we don't
1. **Different `attempt_2`.** They use a second, different sample; we copy the test input.
2. **A no-tool xhigh sample with a server-side thinking budget** (`--reasoning-config` and `thinking_token_budget`): a graceful cap that lets the model write its own answer after a forced `</think>`.
3. **MTP on** (2 tokens), with a fallback ladder in case it fails to start.
4. **`--language-model-only`**: the vision tower is not loaded, which leaves more memory for KV. **[I]** Qwen3.8 ships `preprocessor_config.json` and `video_preprocessor_config.json`, so vLLM loads a vision encoder for it by default.
5. **Server option ladder** for safe flag experiments.
6. **Compact grid format** with explicit dimensions.
7. **A rescue turn** for replies with no grid.
8. In the unused REPL agent:
   - a `submit(code)` that checks train and returns a cell diff (`.` marks a correct cell);
   - a nudge after 6 Python calls without a submit;
   - a forced `tool_choice=submit` after N calls;
   - a per-turn thinking cap with an "act now" order;
   - "never count cells in your head" (`IN_PYTHON_RULE`);
   - an effort ladder after cut turns;
   - a repetition-loop guard (zlib ratio < 0.01);
   - context trimming with hysteresis to keep the prefix cache warm;
   - voting over verified programs, then ranking by the number of train pairs passed.

## We do, they don't
- Hash-verified model and our own wheel sources.
- vLLM 0.23: 4.6× the KV, and 3.5× the concurrency.
- A tool agent tuned on the full eval set: per token, it is 2× more accurate than no-tool at equal mean tokens.
- An adaptive per-pair budget computed from the measured rate.
- **Forced answers at the deadline**: they lose requests still in flight.
- An end-to-end commit gate on a local test set.
- A watchdog, an emergency writer, and a guard thread.
- Per-pair prompts.
- Prefix-cache metrics logged during the run.

## Ideas to port, ranked
Expected numbers are task-weighted, on the public eval, from our E008/E012 files (`scripts/mix_sim.py`, 167 pairs, bootstrap over tasks). No-tool attempts are simulated at the cap: an attempt counts as correct only if it finished correct within the cap. This is a lower bound for a graceful thinking budget.

1. **Use `attempt_2` for a no-tool xhigh sample (mixed pair) instead of the test-input copy.**
   - Simulated results:
     - tools + no-tool at 48k: **36.6% vs 34.6%** for tools ×2 (+2.0 [−2.0, +6.2]), at 79.8k vs 71.2k tokens per pair;
     - tools + no-tool at 64k: **41.2%** (+6.6 [+1.8, +11.5]) at 89.6k;
     - no-tool ×2 at 64k: 36.9% at 108k.
   - The two methods solve different pairs.
   - It only pays if the budget allows a second attempt. With ~44k per pair we cannot give every pair both. The practical policy is:
     1. tool attempts for all pairs first (current);
     2. then no-tool xhigh second attempts with the leftover, ordered by "no `submit` answer" or "fails the train check" first (E012 Next #3, E017).
   - Implementation: we already have the v1 plain path (`ARC_AGENT=plain`) with `FORCE_TEXT`. Use a hard cap or a server `thinking_token_budget`.
   - **Test:** $0 offline simulation (done above). A Kaggle commit-run smoke for scheduling.
   - **Risk:** the tool attempts get less budget if the scheduler is wrong. Keep attempt 2 strictly to leftover capacity, with a deadline guard.
2. **Measure and raise throughput: MTP and `--language-model-only` on vLLM 0.23 at concurrency 35.**
   - **[I]** Expected: +10–20% decode from MTP (E013: +16% at 16 sequences, with 68–99% draft acceptance). The gain shrinks at large batches.
   - `--language-model-only` frees the vision weights and activations for KV.
   - Every +10% of tokens per pair is worth about +1–2 pp on a steep budget curve.
   - **Test:** one E014 replay session on Kaggle (~1 GPU-hour of the 6 h/week quota, $0). Pass `ARC_SERVER_EXTRA='["--language-model-only"]'` and `ARC_MTP=2`. Log tok/s, cache hit rate and preemptions.
   - **Risk:**
     - Spec decoding can produce repetition loops. dfranzen reports 6,144 `!` under MTP. Our traces without MTP have 0/463 loops in medium tool runs and 3/334 in no-tool xhigh runs (`scripts/loop_scan.py`).
     - MTP reduced KV on 0.19 (304k vs 376k).
     - Port the **server option ladder** with it, so that a flag failing to start never costs the run.
3. **Per-turn thinking cap with an "act now" order, plus the "never read or count cells in your head" rule** (their `THINKCAP_HINT` and `IN_PYTHON_RULE`).
   - This targets our biggest waste: 34% of attempts hit the cap and use 43% of tokens, with 0 correct, and 54/193 attempts have a single turn over 15k tokens.
   - Their comments report the same pattern: failures had 1–3 turns cut at 16k, while solves used 2–3k per turn.
   - **[I]** Expected: −15–25% tokens per attempt with flat or better accuracy. This is not proven: E017 found no prompt variant that shortens long prose turns on DeepSeek.
   - **Test:** a GPU box for the tool harness (~3 h on a Vast 4×GPU, about $15–20, needs approval), on the 56 dev pairs, paired against E012.
   - **Risk:** low, since the cap is applied client-side in our raw-completions loop.
4. **Compact grid format with explicit dimensions** in the first message of the tool agent: 64% fewer prompt tokens.
   - **[I]** Gives more concurrency headroom and faster prefill. The accuracy effect is unknown: our tuning used ARC Prize JSON, and dimensions may cut miscounting.
   - **Test:** bundle it with idea 3 on the same box run, as a separate arm.
   - **Risk:** this changes a tuned prompt. Keep JSON unless the paired test is at least neutral.
5. **Train-verified `submit` with a cell diff and voting**: lower priority.
   - Our E011 verified submit gave no gain on DeepSeek.
   - The E017 train check is better used to *rank* attempts (`ranked_answers`) than to gate them.
   - Revisit only for the attempt-2 policy in idea 1.

Not worth porting:
- context trimming (we are append-only and peak at 86.5k of 131k);
- a fresh subprocess per Python call (our persistent session is better);
- server-side tool and reasoning parsers (1 parse failure in ~5k calls with ours);
- the rescue turn (E016: 0–1/24 correct);
- the loop guard, unless MTP is turned on.
