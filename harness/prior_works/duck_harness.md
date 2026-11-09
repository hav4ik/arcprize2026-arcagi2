# Tufa Labs' Duck harness (ARC-AGI-3) and Franzen's Milestone-2 changes: what transfers to our ARC-AGI-2 harness

- **Authors:** Tufa Labs (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel).
  The Duck won the ARC-AGI-3 June-30 milestone; Kaggle reported the score as 1.21.
- **Code:** https://github.com/Tufalabs/duck-harness. It has two commits, and `main` = `7652836` (2026-07-01).
  The harness is in `ARC3-Inference/inference/`.
  - `TAAF` is their game-running framework (`tufa-arc-agi-framework/`).
  - `example-run/` is one full run: 25 games x 20 passes. It is 2.8 GB unpacked, but the clone is ~130 MB.
- **Write-ups:**
  - Blog: https://tufalabs.ai/research/duck-harness/ (short, no ablations).
  - Kaggle write-up: discussion 717133. It is JS-rendered and was not readable here.
  - Notebook (infrastructure only; the solver is an attached dataset): `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`.
- **Franzen's fork:**
  - Repo: https://github.com/da-fr/arc-agi-3-solution @ `10882e3` (2026-09-30).
  - Read `WRITEUP.md` (rationale) and `ARC3-Inference/CONFIGURATION.md` (every option).
  - The notebook `dfranzen/arc-agi-3-milestone-2-solution` carries the same changes as a patch on `7652836`. That
    patch, applied to a copy, is byte-identical to the repo's `inference/`.
  - Public LB 27.89 on ARC-AGI-3 Milestone 2.
- Serving and model details are in [franzen_arc3_milestone2_duck_flashnext.md](franzen_arc3_milestone2_duck_flashnext.md).
  This file covers the harness only.
- **Read 2026-10-27, code not run.** [C] = code, [W] = Franzen's write-up, [M] = our own measurement.

## TL;DR
- The Duck is a **minimal coding agent**. Its parts:
  - one OpenAI-style tool, `python(code)`;
  - a **fresh, restricted subprocess per call**, with the game state preloaded as read-only Python objects;
  - an `action(...)` function that the model calls from inside its code, so a search loop can play moves.
  
  The rest of the harness serves that loop:
  - a ~3k-token system prompt;
  - a ~600-token status prompt every game step, plus a grid image;
  - a carried "world model" note;
  - rolling history trimmed from the front.
- Franzen kept the loop. His reported gains, roughly in order:
  - a better model;
  - exposing **animation frames**;
  - perception: 10x images, **diff images**, the game-over frame;
  - **UNDO**;
  - **guards** that stop wasted actions;
  - a larger context with **thinking retained** and the structured world-model notes **turned off**;
  - **retained model-written functions**;
  - a cache-aware **priority scheduler** across games.
  
  He reports no clean per-feature ablations; his numbers are demo-run impressions ([W] §6).
- **For our ARC-AGI-2 harness, most of the general machinery is something we already have.** Our version is often
  stronger:
  - a persistent IPython session rather than function retention;
  - the answer taken from the session;
  - thinking retained;
  - tool-call rescue;
  - KV-gated admission;
  - budget notes.
  
  The rest is game-specific: actions, guards, animations, UNDO, level transfer, vision.
- The ideas left to test are small:
  - turn the existing `--helpers` on (an isolated A/B);
  - an object-level diff in `check()` feedback;
  - middle truncation of tool output.
- Two Duck/Franzen lessons are **"don't" findings that agree with ours**:
  - no carried summary or notes mechanism;
  - never strip prior thinking.
- One Franzen mechanism, **reallocating compute away from stalled work**, does not transfer: on our logs, long
  attempts still solve at the same rate per token as short ones [M].

## Architecture (upstream @ 7652836) [C]

**Run loop.** In `framework/solver.py`, `HarnessSolver` runs games concurrently. `_HarnessGameSession.play()` repeats
the following until `should_stop()` (game won, per-game wall-clock cap, or action cap):
1. Write the state file.
2. Call `ToolAgent.analyze(...)`.
3. On game over, run an automatic `RESET`.

One `analyze()` call is one "analysis step", a Duck turn:
- Build a user prompt (`_build_user_prompt`) from these parts:
  - the outcome of the previous action batch;
  - step and level;
  - valid actions;
  - fixed tool reminders;
  - the carried world model.
- Append it to the retained history.
- Loop chat completions until a tool call executes at least one real action.
  - A reply without a tool call gets a nudge ("You have not acted yet. Investigate first…").
  - If tool-call markup appears in text, it is recovered (`_recover_tool_calls_from_markup`).
- A step can also end by yielding: `LOCAL_ANALYZER_YIELD_SECONDS`, which is 60 s in the Kaggle launcher.

Defaults:
- `configs/inference.json`: model `vrfai/Qwen3.6-27B-FP8` on vLLM with prefix caching; T 0.6, top-p 0.95, top-k 20;
  thinking on with `preserve_thinking`; harness context 32k (Kaggle max-model-len 65k); tool output ~1,024 tokens;
  tool timeout 30 s.
- Concurrency: 16 games on the Kaggle GPU; `example-run` used 32 jobs per GPU on 2 GPUs (64 in total).
- Time: 45-75 min per game.
- Example run (25 games x 20 passes, 6 h 10 min):
  - mean score 1.60, median 0.07;
  - 29.6M tokens, ~59k tokens per game-run;
  - 1,331 generated tok/s on 2 GPUs (`example-run/summary.txt`).

**The one tool** (`_tools`, `_PYTHON_TOOL_DESCRIPTION`): `python(code: str)`, described as "Run one ephemeral Python
snippet against preloaded ASCII game state". The globals are:
- `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`;
- `valid_actions`, `last_action_result`;
- `action(actions)`, which takes a list such as `['LEFT']` or `[{'action':'MOUSE','row':r,'col':c}]`.

Frames expose only `.ascii` (letter-coded colours), `.segmentation`, `.step`, `.level` and `.shape`. *"The raw
numeric grid is intentionally not exposed."* This is a convention only: the view still holds `_grid`.

`action()` sends a JSON line to the host. The host plays the batch through `_HarnessGameSession.step_env`, which
stops early on level-complete, game-over or win. The sandbox globals are then refreshed in place, so one snippet
can act, inspect and act again.

**Sandbox** (`agent/python_tool_sandbox.py`, `run_sandboxed_python`):
- **A new `python -I -S -c <bootstrap>` process for every call**, in a temporary cwd, with a scrubbed environment and
  its own process group.
- Builtins are an allowlist, and `__import__` is replaced by an allowlist of 14 stdlib modules: **no numpy**.
- `RLIMIT_CPU` = timeout + 1, `RLIMIT_FSIZE` 1 MB, `RLIMIT_NOFILE` 32. On timeout the host kills the process group.
- JSON-lines protocol: the host sends {code, state}; the child streams `action` requests and ends with `final` or
  `error` (the traceback is reduced to user frames).
- **Nothing persists between calls**. The prompt says *"Re-import modules or re-define any custom utility logic you
  need."*
- This is a robustness sandbox, not a security boundary: there is no seccomp and no network namespace.

**Perception** (`utils/segmentation.py`, `segment_layer`):
- 4-connected same-colour components, each with:
  - `id`, in reading order;
  - `color`;
  - a translation-invariant `hash` of colour plus shape;
  - `pixels`;
  - `boundary`, the clockwise outer contour reduced to corner points;
  - `children`, the innermost-encloser containment tree.
- `adjacency_list` gives edge-sharing pairs.
- Segmentation is computed lazily when the code reads it.
- Vision (`agent/vision_context.py`): each user turn gets a PNG of the current grid (`MULTIMODAL_CONTEXT=current_grid`,
  upscale 4 in the config, source default 16).

**Memory.**
- `_extract_scientist_note` parses labelled lines from the assistant text: `World model:`, `Goal model:`,
  `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, `Cross-level notes:`.
- They are re-injected into every user prompt (*"BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED
  VERSION OF THE WORLD MODEL"*).
- They are wiped on level change or game over (`_update_summarized_knowledge_from_step_summary`).

**Context** (`_trim_messages_for_context`, `_persistent_history_messages`):
- Character-based token estimate.
- Drop the oldest user-block until the request fits.
- Keep at most 30 assistant messages across steps.
- On a server "context length" error, trim and retry.
- Tool output is head-truncated at ~4 chars/token x 1,024 (`_trim_tool_text`).

**Prompt style** (`agent/prompts.py`; system prompt ~12.7k chars). It is prescriptive about using code for perception
and keeping output small:
- *"Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw
  board by eye."*
- *"Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs,
  coordinates, counts, or tiny local crops."*
- *"Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit
  search algorithm such as BFS."*
- *"Do not ration tool calls when the state is unclear."*
- *"Once the important state variables and action effects are sufficiently understood, stop probing and search in
  the inferred state space."*

## Franzen's changes (da-fr @ 10882e3; enabled set from the notebook's env cell) [C][W]

| Area | What changed (code) | Reported effect [W] |
|---|---|---|
| Model | Qwen3.6-27B, then Qwen3.8-27B, then Qwen3.8-Flash-Next (INT4) | "Each switch brought a substantial improvement" |
| Thinking | Prior reasoning kept in history (`ARC3_REASONING_HISTORY_KEY=reasoning_content`, `preserve_thinking`), as upstream | "removing it leads to a large regression in score" |
| Context | 128k window, 12k output per request. Calibrated text and image token estimate (`_calibrate_from_usage`). **Hysteresis drain**: once over the ~116k input budget, drop old history down to ~58k (`ARC3_CONTEXT_DRAIN_TOKENS` = 58k, in `_trim_messages_for_context`). Assistant-turn cap 150 with drain 30, coalesced | Larger context helped hard games. 93.4% of prompt tokens served from the prefix cache in a demo run |
| World model | `ARC3_MEMORY_SECTIONS=off` | "The largest improvement in this area came from disabling the structured world-model mechanism entirely" |
| Summaries | Rolling model-written summaries (`ARC3_SUMMARY_*`) | No clear gain, not used |
| Function retention | `utils/retained_functions.py`: after each call, AST-validates the top-level `def`s that executed (no decorators, annotations, non-literal defaults or global writes; free names resolved, explicit imports bundled) and replays their source into the next fresh sandbox. Scope `game`. Rejections are reported to the model | Saves regeneration. No number given |
| Tool output | ~3,072 tokens. **Middle truncation** keeps 2/3 head and 1/3 tail (`_trim_tool_text`, `ARC3_MIDDLE_TRUNCATION`). **Action echo** | Keeps tracebacks and guard messages at the end |
| Perception | `frame_diff()` (`utils/frame_diff.py`): Hungarian matching of components gives moved/rotated/appeared/disappeared/recoloured/resized. Rotation-aware `shape_hash`/`pose_hash` in segmentation | Not isolated |
| Images | Upscale 10x, diff image (changed cells in true colour, rest dimmed), game-over death frame and diff | "biggest score improvement in terms of images" was 4x to 10x |
| Animations | Intermediate frames as REPL variables (`last_animation_frames`, `last_animation_timeline`). A hint only when "transient pixels" occur (`utils/animation.py`) | "strongest benefit" of the perception changes. Composite images hurt |
| Actions | `UNDO` exposed (`EXPOSE_UNDO=on`). Action meanings in the system prompt | Undo, action info, level transfer and persistent functions together: "a large improvement" |
| Guards | `gameplay_changed` = change inside a 4-cell border. Batch no-op stop. Stale-state block (a later `action` after an ineffective one). Terminal latch. Guard exceptions derive from `BaseException`, so the model's `except Exception` cannot swallow them | Not isolated. Repeat/death-ledger guards gave no clear gain |
| Prompts | Detailed game-over diagnosis prompt, level-transfer prompt, and repeated openers that say "no new actions" | Qualitative |
| Scheduling | All games start at once. `_PriorityGate` admits 10 streams. Priority P = (A+B)·C (A: score of the current level with action efficiency; B: future-level value 8/7/5/0, faded over the last 20% of the time; C: a Gaussian decay in actions and generated tokens, fitted on 725 level attempts). A game **keeps its slot across tool calls** and re-queues only when a context trim breaks its prefix anyway | Keeps 10 slots busy and moves time to promising games. The fancier variants were no better |
| Yield | By generated tokens (2,048), not seconds | Throughput-independent |

## ARC-AGI-3-specific vs general
- **Game-specific:**
  - `action()` inside the REPL; batch semantics;
  - level/score/game-over flags and prompts;
  - HUD/timer-bar logic and `gameplay_changed`;
  - no-op, stale-state, terminal and death guards;
  - UNDO/RESET;
  - animations;
  - level-transfer prompts;
  - the world-model notes carried across *game steps*;
  - the A/B terms of the priority formula;
  - the per-step status prompt.
- **General:**
  - one Python tool with preloaded structured state;
  - lazy perception helpers (segmentation, object diff) in the REPL;
  - retaining code across calls;
  - retaining thinking;
  - nudges and markup recovery when no tool call is made;
  - tool-output budget with middle truncation;
  - context trimming with hysteresis;
  - cache-aware admission;
  - priority by measured success hazard;
  - an exception class the model cannot swallow.

## Mapping to our harness (B32: `kaggle/dsv41_tools/pinned_candb1/`; live copy `evals/run_deepseek_tools.py`)

| Duck / Franzen idea | Ours | Gap and expected value for ARC-AGI-2 + E128 on 4xL4 |
|---|---|---|
| Python tool with preloaded task state; code does the perceiving | `run_python` in a **persistent** session with `train`, `test_input` (`tools_for`); prompts `lean2sub`/`lean2code` | None. `lean2code` ("decide in code") did not shorten Flash-1017 (E029) |
| Act or answer from code, not typed text | `submit_answer(expr)` evaluated in the session (`submit_answer`), `--final-mode session`, `--fallback-solve` | None |
| Retained functions across calls (Franzen) | The full session persists (`arc_harness.executor.LocalExecutor`; a timeout gets SIGINT with state kept). 0 "state lost" messages in 6,751 tool results of E052 [M] | None: ours is stronger. Duck needs retention only because each call is a new process |
| Retain prior thinking | One user message per attempt; `reasoning_content` of every turn is kept (`--token-exact` keeps exact ids) | None. Franzen's "large regression" when stripped argues against stripping thinking for KV (the Qwen-harness compaction option, E027) |
| Carried world-model notes / summaries | None | Don't add. Franzen's best setting turned them off. Our E047/E049 informed second attempts and the alternative answer gave ≤ ~1 pp |
| Perception helpers in the REPL (segmentation with containment tree and adjacency; `frame_diff`) | `evals/arc_helpers.py` (`objects` with d8 `shape_key`, holes, `touches_border`; `panels`; `check`), **but `--helpers` is not in B32's `HARNESS` list** (`solver_dsv41.py`) | (a) An isolated A/B of `--helpers` on E128: never isolated, only part of the E017 V3 bundle (+2.7 pp, n.s.; helpers used in 30% of attempts). Costs ~300 cached description tokens per attempt; the kit must ship `arc_helpers.py`. (b) Add `children`/`adjacent` to `objects()`: small. Expected value is small: ARC-AGI-2 grids are ≤30x30, and numpy does in 3 lines what segmentation does for 64x64 boards. The push variant (E017 `--task-summary`) saved 9% tokens but misframed 6 of 10 read traces (E026) |
| Rich failure feedback (game-over diagnosis, `frame_diff`) | `check(fn)` prints ≤8 differing cells `(r,c) got/exp` per pair, then ALL PASS or k/n | An object-level summary of got vs expected per failing pair (e.g. "2 objects missing; object at (3,4) colour 2→5"). Value unknown, plausibly a few % fewer debugging tokens. It only works when helpers are on, so test it together with (a) |
| Tool-output budget with middle truncation | `sandbox_client.tool_output` / `LocalExecutor`: **head-only** truncation at 16,000 chars | E052 (360 attempts, 6,751 results) [M]: p50 587 chars, p90 2.2k, p99 7.3k. Truncation is rare (13 outputs, 0.19%), never in a `check()` cell. Tool results are 12.9% of the tool-result plus reasoning/content characters (tool-call code excluded). Middle truncation is free but changes ~0.2% of outputs. A lower cap (~6k chars, middle-cut) would save ≲2% of context: low value |
| Guards; exceptions the model cannot swallow | `submit_answer` validates the grid; `--train-check` records only; I7/I8 nudges tested and not adopted (E052) | No ARC-AGI-2 analogue worth adding. A "refuse to submit when train fails" guard was tested as `--verify-submit` (E011): no gain |
| No-tool-call nudge, markup recovery | `rescue_code`, `rescue_tool_call`, `RESCUE_NUDGE`, `--lenient-parse`, auto-repair | None |
| Time and budget awareness | `budget_note`, `turns_note`, solver wrap-up note (`ARC_WRAPUP_S`), force deadline | None |
| Cache-aware admission (keep a slot across tool calls; admit only what the cache can hold) | KV-gated admission (`ARC_KV_ADMIT_MAX`, `ARC_ADMIT_MAX_WAITING`), `ARC_MIN_ATTEMPT_TOKENS`, prefix-cache retention | Same lesson, already applied (E042/E057b) |
| Priority by success hazard (C decays with tokens spent; move compute from stalled work) | `pick()`: attempt 0 first, then attempt-1 jobs for pairs without a train-passing answer | Our data says no [M]: among attempts still running at t tokens, solves per 1M remaining tokens stay ~19-25 for single-test tasks from t = 0 to 70k (E052 x 3 + E050 eff50, 480 attempts). A fresh second attempt is worth roughly 7 solves per 1M tokens even if it targets only the ~26% of tasks whose first attempt failed (pass@2 − pass@1 ≈ 8 pp of tasks, at ≥ 40k tokens per such attempt). Keep the budgets; aborting long attempts would lose solves |
| Hysteresis context drain | `--context-limit` ends an attempt with < 1,024 tokens of room; no trimming | 0 of 480 E052/E050 attempts stopped on context [M]. Not needed |
| Vision (10x grid images, diff images) | E128 is text-only (vision tensors dropped); E058 DeepSeek perception probe 0/20 exact | Not applicable |

## Measurements made for this note [M]
Script inputs:
- Tool-output statistics come from the traces of E052's three arms (`experiments/E052_next_code_gate/runs/*/out/dsv41_traces.tar.gz`):
  - 360 attempts, 6,751 tool results;
  - status: 5,406 completed, 616 error, 13 timeout;
  - Traceback in 8.0% of results, NameError in 1.9%.
- The hazard figures add E050 eff50 (`experiments/E050_effort_ab/runs/eff50/dsv41_logs/attempts.jsonl`), scored
  against `data/arc-agi_evaluation_solutions.json` (multi-test attempts scored as the mean over their pairs).

| Spent ≥ t tokens | 0 | 10k | 20k | 30k | 40k | 50k | 60k | 70k |
|---|---|---|---|---|---|---|---|---|
| Single-test attempts still running | 284 | 230 | 170 | 123 | 91 | 67 | 50 | 26 |
| P(correct), still running | 74% | 71% | 66% | 56% | 46% | 33% | 24% | 19% |
| Solves per 1M remaining tokens | 23 | 25 | 25 | 23 | 21 | 19 | 20 | 24 |

Multi-test attempts (196) behave the same: 12-18 solves per 1M remaining tokens at every t. The success rate per
attempt falls, but so does the remaining spend, so the marginal value per token stays flat.

## Open questions
- Tufa's own write-up (Kaggle discussion 717133) could not be read here (JS page), so we have no Tufa ablations.
- Franzen's features have no clean ablations (compute-limited, noisy demo runs). Only the image scale, animations,
  thinking retention and world-model-off are named as clear wins.
- Would `--helpers` (plus an object diff in `check`) help E128 in isolation? It needs a GPU A/B on 120 eval tasks.
  The noise band is ~2-3 pp, so an effect under ~3 pp will not be detectable in one pair of runs.
