"""REPL-TIR: the arc2_harness loop on llama.cpp (flash-llama) or dense Qwen vLLM.

A trajectory is a chat with two native tools: python runs a cell in a persistent session and submit
runs a function on every training and test input. Each call starts a sandboxed child
(qwen38_repl_runtime.py) that replays the trajectory's earlier cells first. The served chat template
renders the system+tools+task prefix once; later turns are rendered here in the Flash-Next template
(check_template compares both renderings against the server at startup) and every turn is sent as
token ids: to llama.cpp /completion (LlamaServer) or vLLM /v1/completions (VllmServer).
Qwen3.5-9B instead renders every turn with the served checkpoint template, supporting both
classic and compact layouts of the same Qwen XML tool protocol.

Defaults are the measured repl-v1-150k setting: 150K completion tokens per trajectory, 40K per turn,
reasoning dropped from earlier turns, thinking closed after 24.5K tokens of a turn.
"""

import argparse
import ast
import asyncio
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import dsv4_llama
import qwen38_llama
import qwen38_repl_audit as input_audit
import qwen38_repl_execution as execution_audit
import qwen38_repl_interactions as interaction_audit
import qwen38_repl_disclosure as disclosure


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "qwen38_repl_runtime.py"
TEMPLATE = HERE / "qwen38_repl_prompt.txt"
PYTHON_FLAGS = ("-s", "-B") + (("-P",) if sys.version_info >= (3, 11) else ())
CHILD_ENV = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"}
MAX_CHILD_OUTPUT = 8 << 20
OUTPUT_CHARS = 4000  # qwen38_repl_runtime.MAX_OUTPUT
AUTO_REPAIR = HERE / "havu" / "evals" / "auto_repair.py"  # His E043 typo repair, used as is.
PROGRESS_NOTE_SCHEMA = {"type": "string", "description": (
    "Required compact working memory, outside hidden thinking: three lines beginning Known:, "
    "Hypothesis:, Next test:. Preserve concrete observations/results, the candidate rule or uncertainty, "
    "and the purpose of this call. No generic 'let me examine' preamble. Usually under 100 words.")}
PROGRESS_REPAIR_TOKENS = 1024
# One answer-only continuation at the soft solver checkpoint, never extra task tokens.
TOOL_CONTINUATION_TOKENS = 2048
PROGRESS_REPAIR = (
    "[harness progress-note repair] The pending tools have NOT executed. Before hidden thinking is "
    "removed, preserve the useful working state from the reasoning and previous tool results. Reply "
    "with ONLY three short lines: Known: concrete observations/results and rejected hypotheses; "
    "Hypothesis: the candidate rule and uncertainty; Next test: what the pending call will test or change. "
    "Be explicit if nothing is established yet. Do not invent results or merely say 'let me examine'. "
    "No thinking, code or tool calls; the original pending calls will execute once after a valid note. "
    "Keep it under 100 words.")

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": "Run Python code in the persistent session. Returns printed output and the value "
                       "of a final expression, or the error traceback.",
        "parameters": {"type": "object", "properties": {"progress_note": PROGRESS_NOTE_SCHEMA,
                       "code": {"type": "string", "description": "Python code"}},
                       "required": ["progress_note", "code"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Submit a function defined in the python session as the solution. It is run on every "
                       "training input and test input; accepted if it reproduces all training outputs.",
        "parameters": {"type": "object", "properties": {"progress_note": PROGRESS_NOTE_SCHEMA,
            "function_name": {"type": "string", "description": "Name of the function, default solve"}},
            "required": ["progress_note"]}}},
]

NUDGE_LENGTH = ("Your last turn reached the per-turn token limit before any tool call. Write down your "
                "current hypothesis briefly and test it with a python call now.")
NUDGE_NO_SUBMIT = ("No submission has been accepted yet. Keep working with the tools: test hypotheses "
                   "with python, then call submit with a function that reproduces every training output.")
NUDGE_FORMAT = ("Your tool call could not be parsed. Call a tool in exactly this format:\n<tool_call>\n"
                "<function=python>\n<parameter=progress_note>\nKnown: ...\nHypothesis: ...\nNext test: ...\n"
                "</parameter>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>\n"
                "Replace the ellipses with actual working state, not placeholders.")
WRAP_UP = ("\n\n[harness] About {left}K completion tokens remain for this puzzle. Submit your best solve now; "
           "a partially correct function is better than none.")
FINALIZE = ("[protected finalization] Exploration time is over; {left} completion tokens remain, "
            "including code and tool calls. Preserve any working candidate. Use short reasoning, "
            "implement your best rule as solve(grid), run check(solve), repair concrete training "
            "mismatches and call submit. Do not start a new broad hypothesis search or repeat full "
            "grids. If an accepted candidate already exists, finish unless an executable check "
            "justifies a revision. Training fit does not prove real test correctness.")
CODE_DELIVERY = (
    "[protected code delivery] {left} generated tokens remain INSIDE the original budget. "
    "Deliver your best current executable rule now, not a promise to implement it later. "
    "Use finalize_solver(code=...) with a complete def solve(grid) and any missing helpers; "
    "existing working helpers remain available. This runs the visible-training check and, if it "
    "passes, submits the same solver immediately. Keep reasoning brief to leave room for code. "
    "If repairing an existing helper via python, include check(solve) in that cell, then submit. "
    "Exploratory cells without a solve definition or check(solve) are paused in this window. "
    "Do not fabricate a rule or memorize demonstrations just to pass. Failed checks remain useful "
    "feedback, and any already saved answers are preserved. Test correctness remains unknown.")
SOLVER_CHECKPOINT = (
    "[provisional solver checkpoint] Before more broad exploration, implement your CURRENT best "
    "general hypothesis as solve(grid) and execute check(solve) on every visible training pair. "
    "It need not pass yet: a concrete failure is useful evidence for the next revision. Do not "
    "wait for certainty, print all grids again, or write an identity stub just to satisfy this request. "
    "Use the measured mismatch to test ONE uncertain rule choice at a time. A provisional check "
    "is not acceptance; submit still requires all training pairs to pass. Keep working code intact.")
BOUNDED_SOLVER_CHECKPOINT = (
    "[bounded provisional solver checkpoint] Broad exploration is paused. Within {turns} model turns, "
    "call propose_solver(code=...) with your CURRENT best general hypothesis as def solve(grid), "
    "including any helper definitions. The harness runs it on every visible training input immediately "
    "and reports mismatches. It need NOT pass: a concrete failed check unlocks further revision. "
    "Do not wait for certainty or supply an identity stub merely to unlock the checkpoint. "
    "Exploratory Python without a solve definition or check(solve) is not executed here. "
    "If no executable check finishes in this allowance, this trajectory stops, preserving its work. "
    "A training check does not submit a solution or prove test correctness.")
TARGETED_REVIEW = (
    "[targeted rule audit] Keep the saved preferred candidate unchanged. Identify ONE consequential "
    "rule choice not forced by the demonstrations, supported by specific observations from them. "
    "If a plausible competing interpretation exists, implement it as a separate named function "
    "and call compare_rules(function_name=..., uncertainty=..., evidence=...). The harness checks "
    "all visible training pairs and reports where its test predictions differ from the saved "
    "candidate; no test answers are available. If no defensible alternative exists, call "
    "compare_rules without function_name and explain the evidence ruling out the competing choice. "
    "Do not fabricate uncertainty, patch individual test cells, assume transformation invariance, "
    "or treat self-created synthetic answers as proof of the rule. Use the remaining review budget "
    "for this comparison, not a second open-ended analysis of the whole puzzle.")
DIVERSITY_GUIDANCE = """

# Preserve a second candidate
After saving your preferred function with submit, if a genuinely plausible alternative rule also
fits ALL visible training pairs but gives different test predictions, submit that named function
with preserve_only=true. This saves it for attempt_2 WITHOUT replacing your preferred candidate.
Do not invent an alternative just to fill a slot or change individual test cells. Preserve the
working function while testing a separate function. Repeated identical predictions add no votes.
Real test answers are unavailable. Review uses the same total completion budget, not extra tokens.
"""
REVIEW_GUIDANCE = """

# Generalization review
A submit pass means ONLY that the function reproduces the available demonstrations and returns
valid test grids. Real test answers are unavailable: the harness cannot certify your test rule.
Agreement between your prediction and your own hypothesis is not independent evidence.
After the first pass, challenge the most consequential unsupported assumption with an executable
comparison. Keep the current working function intact while testing a named alternative. Use all
visible training pairs and inspect where their predictions on the real test INPUTS disagree.
Do not invent expected test outputs. Synthetic examples can expose implementation mistakes, but
their expected answers follow an assumed rule and cannot prove that rule. Do not assume rotation,
reflection, recoloring or translation invariance without evidence from the demonstrations.
"""
REVIEW = ("Training fit is not a generalization verdict. Before finishing:\n"
          "1. State the rule and distinguish observed constraints from assumptions. Inspect the test inputs "
          "for cases absent from training: different object counts/relations, boundary contacts, overlaps, "
          "color roles, ties or sizes. Use targeted measurements, not another full-grid dump.\n"
          "2. Identify the most consequential choice the demonstrations do not force. Examples include "
          "clipping versus discarding at an edge, structural roles versus fixed colors, and overlap/order "
          "policies. These are possibilities to investigate, not instructions to apply any particular rule.\n"
          "3. If there is a plausible alternative, implement it as a SEPARATE function; run check on both "
          "functions and compare their test predictions in Python. Print a compact report of training "
          "passes and disagreement coordinates/crops. Keep the original function available. Prefer an "
          "alternative only when the demonstrated relationships support it better, not merely because "
          "its output looks nicer or it passes synthetic examples you invented.\n"
          "4. If both fit and the evidence cannot distinguish them, acknowledge that ambiguity. Do not "
          "claim the test is verified or patch individual test cells. If no plausible alternative remains, "
          "state which evidence rules it out. Resubmit only an evidence-backed revision; otherwise finish "
          "with the current candidate. This review uses the existing budget, not an extra allowance.")
REVIEW_FOLLOWUP = ("The candidate is saved, but test correctness is still unknown. Use the measured "
                   "training constraints and any executed rule comparison to decide whether a revision "
                   "is justified. Do not restart the same review or treat self-consistency as verification.")
ACCEPTED_ASK = ("Review each prediction against the rule. If one is wrong, fix the function and submit again. "
                "If they are right, reply with a short final message and no tool call.")
PREDICTION_IMAGES = ("[Prediction images] The saved candidate's test predictions: each image shows a test input "
                     "(left) and the predicted output (right) in the task's colors. These are predictions, not "
                     "answers. Check that each output follows the relation shown by the training pairs, including "
                     "any object, color or region of the test input that the training pairs do not cover.")
PREDICTION_IMAGE_LIMIT = 2
ELIDED = "[output removed to fit the context window; the session state is unchanged, print again if needed]"
HISTORY_TRIMMED = ("[Earlier complete turns were removed to make context room. The original puzzle and "
                   "Python session state are unchanged. Existing functions and variables remain available; "
                   "use targeted measurements if an earlier observation is needed.]")
STRUCTURE_GUIDANCE = """

# Optional structural analysis and executable search
The python session also provides:
- describe_grid(grid, bg=None, diag=False, multicolor=False, max_objects=12): compact color counts,
  grid symmetries, components, shape IDs, boundary contacts, adjacency and bounding-box containment.
- grid_diff(before, after, bg=None, diag=False, multicolor=False, max_matches=12): size/color changes,
  cell transitions when aligned, and possible component correspondences under rotations/reflections.
  Correspondences are hypotheses, not established object identities; repeated shapes can be ambiguous.
- search_candidates(candidates, max_candidates=256, max_results=5): evaluate a mapping or iterable of
  (name, function) against ALL VISIBLE TRAINING pairs; return ranked exact-match and cell-fit reports.
  It never checks real test answers. Keep the functions in your own candidates mapping to reuse them.

Use these tools only when they answer a specific uncertainty. Background defaults to the most common
color, which can be wrong: compare explicit backgrounds, diagonal connectivity or multicolor objects
when justified. Shape IDs ignore color and, explicitly, rotations/reflections; that does NOT prove the
puzzle is invariant to those transformations. Bounding-box containment is not semantic containment.
Once you have a plausible family of rules, enumerate its choices in Python and test them together,
instead of tracing each variant mentally. For example, compare candidate object-selection criteria or
transformations across all training examples. Capture loop parameters in lambdas with defaults. Reject
rules contradicted by any example; do not patch individual cells or use example-specific lookup tables.
Several rules can fit every example: inspect which assumption distinguishes their test predictions.
Print compact reports, not whole grids or every unsuccessful candidate.
"""
# Qwen's own early-stop text for a thinking budget; the turn then continues as the answer.
THINK_CLOSE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the "
               "thinking directly now.\n</think>\n\n")
CONTEXT_MARGIN = 64  # Prompts are counted exactly; this covers the close text of a capped turn.

# Flash-Next chat template (experiments/qwen38_tir_api/.../check_v3/checkpoint_chat_template.jinja).
EFFORT = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
             "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
             "clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
           "conclusion without unnecessary elaboration.",
}
TOOL_FORMAT = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
    "the specified format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
    "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
    "there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>")
GENERATION_PROMPT = qwen38_llama.GENERATION_PROMPT
SPECIAL_TEXT = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def add_arguments(parser):
    g = parser.add_argument_group("REPL-TIR (--solve-mode tir --tir-interface repl, flash-llama, flash-next, 27b or qwen35-9b)")
    g.add_argument("--tir-repl-trajectories", type=int, default=1,
                   help="Maximum fresh trajectories per task; stops early once two accepted ones agree. "
                        "With --tir-repl-second-attempt cot, 2 means one REPL pass plus one CoT pass, not two REPL passes.")
    g.add_argument("--tir-repl-trajectory-schedule", choices=("sequential", "queued"), default="sequential",
                   help="sequential: finish each task's trajectories before taking another task. queued: "
                        "waiting first attempts have priority, then fresh retries in --queue-order among "
                        "finished tasks. No all-task completion barrier. REPL retries share the task token cap; "
                        "CoT second attempts use independent context windows.")
    g.add_argument("--tir-repl-second-attempt", choices=("repl", "cot"), default="repl",
                   help="cot: queued second pass uses fresh raw CoT, one request per test input, "
                        "preserving the first REPL answer. Requires two queued native-vLLM trajectories.")
    g.add_argument("--tir-repl-cot-context", type=int, default=102400,
                   help="Context ceiling for second-pass CoT (prompt, images and completion together), "
                        "also capped by --context. Each test input gets the full window independently of "
                        "--task-token-budget/--tir-repl-budget. Does not change the first REPL context.")
    g.add_argument("--tir-repl-budget", type=int, default=150_000, help="Completion tokens per trajectory.")
    g.add_argument("--tir-repl-turn-tokens", type=int, default=40_000, help="Completion tokens per turn.")
    g.add_argument("--tir-repl-think-tokens", type=int, default=24_576,
                   help="Thinking tokens per turn before the thinking is closed; 0 never closes it.")
    g.add_argument("--tir-repl-max-turns", type=int, default=60)
    g.add_argument("--tir-repl-max-nudges", type=int, default=3)
    g.add_argument("--tir-repl-repeat-limit", type=int, default=3,
                   help="Warn after this many identical tool turns/results; stop the trajectory at twice "
                        "this count. Empty successful output is excluded. 0 disables this guard.")
    g.add_argument("--tir-repl-error-limit", type=int, default=6,
                   help="Stop a trajectory after this many consecutive failed tool turns; warn halfway. "
                        "0 disables this guard.")
    g.add_argument("--tir-repl-wrap-up", type=int, default=20_000,
                   help="Tell the model to submit once fewer completion tokens than this remain.")
    g.add_argument("--tir-repl-finalize-reserve", type=int, default=0,
                   help="Opt-in hard completion-token reserve for write/check/submit; 0 keeps legacy behavior.")
    g.add_argument("--tir-repl-finalize-think-tokens", type=int, default=2048,
                   help="Thinking cap per protected-finalization turn (code gets separate room).")
    g.add_argument("--tir-repl-finalize-turn-tokens", type=int, default=6000,
                   help="Maximum protected-finalization turn, allowing several tool/repair cycles.")
    g.add_argument("--tir-repl-min-turn", type=int, default=12_000,
                   help="Stop when the context leaves room for fewer tokens than this.")
    g.add_argument("--tir-repl-context-stop", choices=("min-turn", "planned"), default="min-turn",
                   help="Room a turn needs before the context stops a trajectory: min-turn always asks for "
                        "--tir-repl-min-turn; planned asks for no more than the turn's own cap, so a turn "
                        "shortened by the deadline or finalization drops old reasoning and runs instead of stopping.")
    g.add_argument("--tir-repl-keep-reasoning", type=int, default=0,
                   help="Earlier assistant turns that keep their reasoning; -1 keeps all.")
    g.add_argument("--tir-repl-progress-notes", choices=("required", "optional"), default="required",
                   help="Optional removes the progress-note requirement and repair requests (ablation).")
    g.add_argument("--tir-repl-structure-tools", action="store_true",
                   help="Expose compact grid descriptions/diffs and training-only candidate search.")
    g.add_argument("--tir-repl-visual-tools", action=argparse.BooleanOptionalAction, default=False,
                   help="With text-image input, expose view(*grids, reason=...) in Python. "
                        "Keep original puzzle images and at most two recent tool images per request.")
    g.add_argument("--tir-repl-train-mismatch-images", action="store_true",
                   help="Opt-in automatic visible-training mismatch crops from check(); needs visual tools.")
    g.add_argument("--tir-repl-prediction-images", action="store_true",
                   help="After a training-passing preferred submit, show each test input beside its predicted "
                        "output as an image, at most twice per trajectory; needs visual tools.")
    g.add_argument("--tir-repl-grid-text", choices=("compact", "spaced"), default="compact",
                   help="How grid rows are written in the prompt, harness feedback, show() and check(). spaced puts "
                        "a space between cells: DeepSeek's tokenizer reads compact rows 3 digits per token "
                        "('671','111'), so no token is one cell; spaced rows give every cell its own token "
                        "(2 tokens per cell instead of about 1/3). Qwen splits digits either way.")
    g.add_argument("--tir-repl-candidate-pool", action=argparse.BooleanOptionalAction, default=False,
                   help="Preserve distinct training-passing submissions for attempt_2; preferred answer unchanged.")
    g.add_argument("--tir-repl-challenger-tokens", type=int, default=0,
                   help="Opt-in fresh-context challenger after the primary finishes. Uses at most this many "
                        "unused task-budget tokens; fills attempt_2 only, never replaces attempt_1. "
                        "Requires a task token budget, one native-vLLM trajectory and all demonstrations visible.")
    g.add_argument("--tir-repl-review-token-budget", type=int, default=0,
                   help="Opt-in cap on additional completion tokens after the first accepted submit; 0 unlimited.")
    g.add_argument("--tir-repl-review-seconds", type=int, default=0,
                   help="Opt-in wall-time cap after the first accepted submit; 0 unlimited. "
                        "Resubmissions do not reset either review budget.")
    g.add_argument("--tir-repl-finalize-reserve-seconds", type=int, default=0,
                   help="Opt-in wall-time finalization window before the task/global deadline. "
                        "Uses observed turn throughput to shorten turns approaching this window; 0 disables.")
    g.add_argument("--tir-repl-wall-rate", choices=("turn", "decode"), default="turn",
                   help="How wall-time windows become token caps: turn divides recent turns' tokens by their whole "
                        "time, queue wait included; decode takes the longest recent wait before the first token "
                        "off the window and uses the recent decode rate.")
    g.add_argument("--tir-repl-answer-room", type=int, default=512,
                   help="Tokens kept for the answer and tool call when a turn cap closes thinking (at most half "
                        "the turn).")
    g.add_argument("--tir-repl-solver-by-tokens", type=int, default=0,
                   help="Opt-in provisional-solver checkpoint at this generated-token count; 0 disables. "
                        "Uses finalization turn/thinking caps until check(solve) has actually executed.")
    g.add_argument("--tir-repl-solver-max-turns", type=int, default=0,
                   help="Opt-in bounded solver checkpoint: expose propose_solver(code), automatically "
                        "check its function on training inputs, and stop after this many checkpoint "
                        "turns without a completed check. Requires --tir-repl-solver-by-tokens; 0 keeps "
                        "the soft checkpoint. A failed training check still releases the checkpoint.")
    g.add_argument("--tir-repl-solver-checkpoint-action", choices=("stop", "resume"), default="stop",
                   help="With a bounded solver checkpoint, stop (legacy) or resume exploration after "
                        "the allowance without a completed check. Resume preserves the same task budget.")
    g.add_argument("--tir-repl-auto-check-solver", action=argparse.BooleanOptionalAction, default=False,
                   help="Opt-in controller-owned training check after Python cells defining solve(grid) "
                        "or requesting check(solve), including outside the early checkpoint. "
                        "Failed cells never validate a stale solver; no test answers are used.")
    g.add_argument("--tir-repl-solver-recheck-tokens", type=int, default=0,
                   help="Opt-in minimum generated-token interval for full solve checks after successful "
                        "helper/data edits. Requires auto-check-solver; no extra checks after acceptance. "
                        "0 disables. Checks are isolated, training-only, and do not submit answers.")
    g.add_argument("--tir-repl-code-reserve", type=int, default=0,
                   help="Opt-in code-delivery window INSIDE finalize-reserve (>=4000 tokens). "
                        "Before acceptance, protect this boundary from debugging turns; cap thinking "
                        "at 512 tokens and require executable solver/check calls. Expose finalize_solver "
                        "to write, validate and submit in one call. 0 preserves existing behavior.")
    g.add_argument("--tir-repl-targeted-review", action=argparse.BooleanOptionalAction, default=False,
                   help="Offer compare_rules for a bounded executable alternative-rule audit after acceptance. "
                        "Requires --tir-repl-review and a positive --tir-repl-review-token-budget.")
    g.add_argument("--tir-repl-test-reveal", choices=("all", "last-first"), default="all",
                   help="Opt-in staged test inputs: start with the last, then disclose others after "
                        "a saved solver, or all at finalization. Same task/review budgets. Native vLLM, "
                        "one trajectory, no holdout/challenger/input audits. Single-test tasks unchanged.")
    g.add_argument("--tir-repl-input-audit", action=argparse.BooleanOptionalAction, default=False,
                   help="Opt-in executable rule-condition measurements on every train/test INPUT. "
                        "Requires review and all demonstrations visible. With candidate pooling, "
                        "snapshot independently validated check(alt) alternatives after acceptance.")
    g.add_argument("--tir-repl-execution-audit", action=argparse.BooleanOptionalAction, default=False,
                   help="Experimental saved-solver state/effect tracing and explicit mechanism challenge. "
                        "Replaces the required raw-input audit, which remains optional. Requires input-audit "
                        "tools and a positive review token budget; never increases existing budgets.")
    g.add_argument("--tir-repl-interaction-audit", action=argparse.BooleanOptionalAction, default=False,
                   help="Experimental coverage/operation-interaction probes plus an explicit review of "
                        "EVERY test input. Replaces the mandatory raw-input audit; requires input-audit "
                        "tools and a positive review token budget. Unresolved cases do not veto saved answers.")
    g.add_argument("--tir-repl-context-policy", choices=("legacy", "blocks"), default="legacy",
                   help="Blocks persistently evicts oldest complete turns, leaving the raw trace intact.")
    g.add_argument("--tir-repl-context-retain-fraction", type=float, default=0.55,
                   help="Blocks target this fraction of context after a trim, subject to output room; "
                        "the task prefix and latest complete turn are never evicted.")
    g.add_argument("--tir-repl-cache-salt", default=None,
                   help="Optional vLLM cache namespace for controlled experiments; never added to model text.")
    g.add_argument("--tir-repl-prompt-ids", choices=("ledger", "render"), default=None,
                   help="ledger: the next prompt is the served prompt + generated ids + the new messages' tokens, "
                        "so the prefix cache covers every generated token (re-rendering re-splits them). Any "
                        "history edit re-renders. Vision sends exact ids to /inference/v1/generate. "
                        "render: re-render every turn. Default: ledger only for servers built on it (DeepSeek "
                        "V4.1 llama-server), render on vLLM, where a 120-task 27B A/B prefilled the same tokens "
                        "either way and the ledger scored lower (49.3 vs 62.3).")
    g.add_argument("--tir-repl-cell-timeout", type=float, default=20)
    g.add_argument("--tir-repl-memory-mib", type=int, default=4096)
    g.add_argument("--tir-repl-exec-slots", type=int, default=8, help="Concurrent sandbox children.")
    g.add_argument("--tir-repl-output-chars", type=int, default=OUTPUT_CHARS,
                   help="Characters of a cell's printed output the model sees: the first 2/3 and the last 1/3, "
                        "with the middle cut. Tracebacks keep their own 2500-character cap.")
    g.add_argument("--tir-repl-auto-repair", type=int, default=0,
                   help="Opt-in: after a cell fails on a misspelled name (coms -> comps, mmost_common, dequeue), "
                        "rewrite the name with havu/evals/auto_repair.py and rerun the cell, up to this many "
                        "times. The model sees '[auto-corrected: typo -> name]' and the rerun's output; the "
                        "rerun replaces the failed cell in the session. Also makes `from scipy import ndi` work. "
                        "0 disables.")
    g.add_argument("--tir-repl-review", action=argparse.BooleanOptionalAction, default=False,
                   help="After the first training pass, request an executable comparison of uncertain rules; "
                        "report input differences without using test answers. Uses the same token budget.")
    g.add_argument("--tir-repl-require-train-pass", action=argparse.BooleanOptionalAction, default=False,
                   help="Do not return a partial fallback program unless it passes every training pair. "
                        "Train fit is not proof of test correctness; existing behavior is unchanged without this flag.")
    g.add_argument("--tir-repl-holdout-first", action="store_true",
                   help="With >=3 training pairs, hide the first pair from the prompt/session and validate "
                        "it on submit. Return aggregate mismatch feedback; require a holdout pass for selection.")
    g.add_argument("--tir-repl-cache-prompt", action="store_true",
                   help="flash-llama: send cache_prompt=true so a slot reuses the previous turn's KV "
                        "(untested on 4x L4). vLLM reuses it through its prefix cache.")


def validate_options(args):
    schedule = getattr(args, "tir_repl_trajectory_schedule", "sequential")
    if schedule not in ("sequential", "queued"):
        raise ValueError("Unknown REPL trajectory schedule")
    if schedule == "queued" and (args.tir_repl_trajectories < 2
                                or getattr(args, "tir_repl_challenger_tokens", 0)):
        raise ValueError("Queued trajectories require at least two trajectories and no anchored challenger")
    second_attempt = getattr(args, "tir_repl_second_attempt", "repl")
    if second_attempt not in ("repl", "cot"):
        raise ValueError("Unknown REPL second-attempt mode")
    if second_attempt == "cot" and (schedule != "queued" or args.tir_repl_trajectories != 2
                                   or args.qwen38_variant not in ("27b", "flash-next", "qwen35-9b", "dsv4-vllm")):
        raise ValueError("CoT second attempt requires two queued native-vLLM trajectories")
    cot_context = getattr(args, "tir_repl_cot_context", 102400)
    if type(cot_context) is not int or not 1024 <= cot_context <= 2_000_000:
        raise ValueError("--tir-repl-cot-context must be an integer in 1024..2000000")
    reveal = getattr(args, "tir_repl_test_reveal", "all")
    if reveal not in ("all", "last-first"):
        raise ValueError("--tir-repl-test-reveal must be all or last-first")
    if reveal != "all":
        if (args.tir_repl_trajectories != 1 or args.qwen38_variant in ("flash-llama", "dsv4", "dsv4-vllm")
                or any(getattr(args, key, False) for key in ("tir_repl_holdout_first",
                    "tir_repl_challenger_tokens", "tir_repl_input_audit", "tir_repl_execution_audit",
                    "tir_repl_interaction_audit"))):
            raise ValueError("Staged test reveal requires one native-vLLM trajectory, all training pairs "
                             "visible, and no challenger/input/execution/interaction audit")
        if not getattr(args, "tir_repl_finalize_reserve", 0):
            raise ValueError("Staged test reveal requires a positive --tir-repl-finalize-reserve")
    challenger = getattr(args, "tir_repl_challenger_tokens", 0)
    if type(challenger) is not int or challenger < 0 or challenger > 100000:
        raise ValueError("--tir-repl-challenger-tokens must be an integer in 0..100000")
    if challenger:
        if challenger < 2000 or not getattr(args, "task_token_budget", None):
            raise ValueError("Challenger needs >=2000 tokens and --task-token-budget")
        if (args.tir_repl_trajectories != 1 or getattr(args, "tir_repl_holdout_first", False)
                or args.qwen38_variant in ("flash-llama", "dsv4", "dsv4-vllm")):
            raise ValueError("Challenger requires one native-vLLM trajectory with all demonstrations visible")
    if getattr(args, "tir_repl_visual_tools", False) and getattr(args, "input_mode", "text") != "text-image":
        raise ValueError("--tir-repl-visual-tools requires --input-mode text-image")
    if getattr(args, "tir_repl_train_mismatch_images", False) and not getattr(args, "tir_repl_visual_tools", False):
        raise ValueError("--tir-repl-train-mismatch-images requires --tir-repl-visual-tools")
    if getattr(args, "tir_repl_prediction_images", False) and not getattr(args, "tir_repl_visual_tools", False):
        raise ValueError("--tir-repl-prediction-images requires --tir-repl-visual-tools")
    for name, low, high in (("tir_repl_finalize_reserve", 0, args.tir_repl_budget),
                            ("tir_repl_finalize_think_tokens", 1, 1_000_000),
                            ("tir_repl_finalize_turn_tokens", 1000, 1_000_000),
                            ("tir_repl_review_token_budget", 0, 1_000_000),
                            ("tir_repl_review_seconds", 0, 86400),
                            ("tir_repl_finalize_reserve_seconds", 0, 86400),
                            ("tir_repl_answer_room", 256, 16384),
                            ("tir_repl_solver_by_tokens", 0, args.tir_repl_budget),
                            ("tir_repl_solver_recheck_tokens", 0, args.tir_repl_budget),
                            ("tir_repl_code_reserve", 0, args.tir_repl_budget),
                            ("tir_repl_solver_max_turns", 0, 100)):
        value = getattr(args, name, {"tir_repl_finalize_think_tokens": 2048,
                                    "tir_repl_finalize_turn_tokens": 6000}.get(name, 0))
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"--{name.replace('_', '-')} must be an integer in {low}..{high}")
    reserve = getattr(args, "tir_repl_finalize_reserve", 0)
    if reserve and reserve < 2000:
        raise ValueError("--tir-repl-finalize-reserve must be 0 or at least 2000")
    recheck = getattr(args, "tir_repl_solver_recheck_tokens", 0)
    if recheck and (recheck < 1000 or not getattr(args, "tir_repl_auto_check_solver", False)):
        raise ValueError("Solver rechecks require >=1000 tokens and --tir-repl-auto-check-solver")
    delivery = getattr(args, "tir_repl_code_reserve", 0)
    if delivery and not 4000 <= delivery <= reserve:
        raise ValueError("Code reserve must be >=4000 and inside --tir-repl-finalize-reserve")
    checkpoint = getattr(args, "tir_repl_solver_by_tokens", 0)
    if checkpoint and not 1000 <= checkpoint <= args.tir_repl_budget - reserve - 1000:
        raise ValueError("Solver checkpoint must be >=1000 and leave room before finalization")
    if getattr(args, "tir_repl_solver_max_turns", 0) and not checkpoint:
        raise ValueError("--tir-repl-solver-max-turns requires --tir-repl-solver-by-tokens")
    checkpoint_action = getattr(args, "tir_repl_solver_checkpoint_action", "stop")
    if checkpoint_action not in ("stop", "resume"):
        raise ValueError("--tir-repl-solver-checkpoint-action must be stop or resume")
    if checkpoint_action == "resume" and not getattr(args, "tir_repl_solver_max_turns", 0):
        raise ValueError("Resuming a solver checkpoint requires --tir-repl-solver-max-turns")
    if getattr(args, "tir_repl_targeted_review", False) and (
            not args.tir_repl_review or not getattr(args, "tir_repl_review_token_budget", 0)):
        raise ValueError("--tir-repl-targeted-review requires review and a positive review token budget")
    if getattr(args, "tir_repl_input_audit", False) and (
            not args.tir_repl_review or getattr(args, "tir_repl_holdout_first", False)):
        raise ValueError("--tir-repl-input-audit requires --tir-repl-review and all demonstrations visible")
    if getattr(args, "tir_repl_execution_audit", False) and (
            not getattr(args, "tir_repl_input_audit", False) or not getattr(args, "tir_repl_review_token_budget", 0)):
        raise ValueError("--tir-repl-execution-audit requires --tir-repl-input-audit and a positive review token budget")
    if getattr(args, "tir_repl_interaction_audit", False):
        if not getattr(args, "tir_repl_input_audit", False) or not getattr(args, "tir_repl_review_token_budget", 0):
            raise ValueError("--tir-repl-interaction-audit requires --tir-repl-input-audit and a positive review token budget")
        if getattr(args, "tir_repl_execution_audit", False):
            raise ValueError("Use interaction-audit OR execution-audit as separate experimental treatments")
    if getattr(args, "tir_repl_finalize_think_tokens", 2048) + 512 > getattr(args, "tir_repl_finalize_turn_tokens", 6000):
        raise ValueError("Finalization needs at least 512 tokens after its thinking cap for code/tool calls")
    salt = getattr(args, "tir_repl_cache_salt", None)
    if salt is not None and (not 1 <= len(salt) <= 1024 or args.qwen38_variant in ("flash-llama", "dsv4")):
        raise ValueError("--tir-repl-cache-salt requires vLLM and a nonempty string of at most 1024 characters")
    fraction = getattr(args, "tir_repl_context_retain_fraction", 0.55)
    if not math.isfinite(fraction) or not 0.1 <= fraction <= 0.9:
        raise ValueError("--tir-repl-context-retain-fraction must be in 0.1..0.9")
    for name, low, high in (("tir_repl_trajectories", 1, 16), ("tir_repl_budget", 1000, 2_000_000),
                            ("tir_repl_turn_tokens", 1000, 1_000_000), ("tir_repl_think_tokens", 0, 1_000_000),
                            ("tir_repl_max_turns", 1, 500), ("tir_repl_max_nudges", 0, 100),
                            ("tir_repl_repeat_limit", 0, 100), ("tir_repl_error_limit", 0, 100),
                            ("tir_repl_wrap_up", 0, 1_000_000), ("tir_repl_min_turn", 1, 1_000_000),
                            ("tir_repl_keep_reasoning", -1, 1000), ("tir_repl_memory_mib", 256, 65536),
                            ("tir_repl_exec_slots", 1, 256), ("tir_repl_output_chars", 200, 200_000),
                            ("tir_repl_auto_repair", 0, 5)):
        value = getattr(args, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"--{name.replace('_', '-')} must be an integer in {low}..{high}")
    if not math.isfinite(args.tir_repl_cell_timeout) or not 0.1 <= args.tir_repl_cell_timeout <= 600:
        raise ValueError("--tir-repl-cell-timeout must be in 0.1..600 seconds")
    if getattr(args, "tir_repl_auto_repair", 0):
        auto_repair_module()  # Fail before the server starts, not at the first typo.
    if args.tir_repl_min_turn > args.tir_repl_turn_tokens:
        raise ValueError("--tir-repl-min-turn exceeds --tir-repl-turn-tokens")
    if args.tir_repl_turn_tokens + args.tir_repl_min_turn > args.context:
        raise ValueError("--context leaves no room for a REPL turn")


# ----- task text ---------------------------------------------------------------------------


def tool_schemas(args=None):
    tools = deepcopy(TOOLS)
    if getattr(args, "tir_repl_solver_max_turns", 0) or getattr(args, "tir_repl_auto_check_solver", False):
        proposal = deepcopy(TOOLS[0])
        proposal["function"].update(name="propose_solver", description=(
            "Define your current general hypothesis as def solve(grid), with helper code as needed. "
            "The harness immediately checks it on every visible training input and reports mismatches. "
            "A partial or failing hypothesis is useful; no test answers are available. This does not "
            "submit the solution. Call submit separately once your rule is ready."))
        tools.append(proposal)
    if getattr(args, "tir_repl_code_reserve", 0):
        delivery = deepcopy(TOOLS[0])
        delivery["function"].update(name="finalize_solver", description=(
            "Write your best current general rule as def solve(grid), with missing helpers. "
            "Executes the code, independently checks all visible training pairs, and if they pass "
            "immediately submits solve on every test input using normal validation. Failed checks "
            "do not replace any saved answer. Test correctness remains unknown. This is a submission, "
            "unlike propose_solver; it is also available before the final code-delivery window."))
        tools.append(delivery)
    if getattr(args, "tir_repl_progress_notes", "required") == "optional":
        for tool in tools:
            params = tool["function"]["parameters"]
            params["required"].remove("progress_note")
            params["properties"]["progress_note"]["description"] = (
                "Optional concise observations or hypothesis to accompany this call; any format is allowed.")
    if getattr(args, "tir_repl_candidate_pool", False):
        tools[1]["function"]["parameters"]["properties"]["preserve_only"] = {
            "type": "boolean", "description": "Save a training-passing alternative for attempt_2 without "
            "replacing the preferred submission. Default false."}
    if getattr(args, "tir_repl_targeted_review", False) or getattr(args, "tir_repl_input_audit", False):
        tools.append({"type": "function", "function": {
            "name": "compare_rules",
            "description": "After submitting a preferred candidate, test a separately defined alternative "
                "on all training pairs and compare test predictions. A passing alternative is preserved "
                "without replacing the preferred one; used for attempt_2 only when candidate pooling is enabled. "
                "Omit function_name only if evidence rules out a defensible alternative. No test labels are used.",
            "parameters": {"type": "object", "properties": {
                "progress_note": deepcopy(tools[0]["function"]["parameters"]["properties"]["progress_note"]),
                "function_name": {"type": "string", "description": "Name of the separately defined alternative."},
                "uncertainty": {"type": "string", "description": "One concrete competing rule interpretation."},
                "evidence": {"type": "string", "description": "Specific visible-training evidence for or against it."}},
                "required": ["uncertainty", "evidence"] + (
                    ["progress_note"] if getattr(args, "tir_repl_progress_notes", "required") == "required" else [])}}})
    if getattr(args, "tir_repl_input_audit", False):
        tools.append({"type": "function", "function": {
            "name": "audit_inputs",
            "description": "After saving a preferred candidate, execute a named measure(grid) function "
                "on EVERY original training and test input. Return 1..8 scalar/list features about an "
                "actual rule assumption. The harness reports observed values and novelty, NOT test correctness.",
            "parameters": {"type": "object", "properties": {
                "measure_name": {"type": "string", "description": "Previously defined measure(grid) function name."},
                "assumption": {"type": "string", "description": "Rule-dependent condition this measurement tests."},
                "training_evidence": {"type": "string", "description": "Specific demonstrations supporting it; not invented test answers."},
                "progress_note": deepcopy(tools[0]["function"]["parameters"]["properties"]["progress_note"])},
                "required": ["measure_name", "assumption", "training_evidence"] + (
                    ["progress_note"] if getattr(args, "tir_repl_progress_notes", "required") == "required" else [])}}})
    if getattr(args, "tir_repl_execution_audit", False):
        tools.append(execution_audit.tool_schema(
            deepcopy(tools[0]["function"]["parameters"]["properties"]["progress_note"]),
            getattr(args, "tir_repl_progress_notes", "required") == "required"))
        comparison = next(t["function"] for t in tools if t["function"]["name"] == "compare_rules")
        comparison["parameters"]["properties"].update(
            change_scope={"type": "string", "enum": list(execution_audit.SCOPES),
                          "description": "Model-declared scope of the competing explanation, not certified by the harness."},
            discriminator={"type": "string", "description": "Concrete observation separating the core mechanisms; "
                           "if only a small variant is tested, explain what mechanism uncertainty remains."})
        comparison["parameters"]["required"] += ["change_scope", "discriminator"]
    if getattr(args, "tir_repl_interaction_audit", False):
        tools.extend(interaction_audit.tool_schemas(
            deepcopy(tools[0]["function"]["parameters"]["properties"]["progress_note"]),
            getattr(args, "tir_repl_progress_notes", "required") == "required"))
    return tools


def experiment_prompt(system, args):
    """Add opt-in experiment guidance; default prompt stays byte-for-byte identical."""
    if getattr(args, "tir_repl_progress_notes", "required") == "optional":
        start, end = system.find("# Progress notes between tool calls"), system.find("After an error,")
        if start < 0 or end <= start:
            raise ValueError("Optional notes require the built-in REPL prompt with its progress-note section")
        system = (system[:start] + "# Notes\nConcise observations or hypotheses before a tool call are "
                  "optional. No special format is required, and tools run without a note.\n\n" + system[end:])
        system = system.replace("python(progress_note, code)", "python(code, progress_note=None)")
        system = system.replace("The required progress_note is described below. ", "")
        system = system.replace('submit(progress_note, function_name="solve")',
                                'submit(function_name="solve", progress_note=None)')
    if getattr(args, "tir_repl_structure_tools", False):
        system += STRUCTURE_GUIDANCE
    if getattr(args, "tir_repl_visual_tools", False):
        if getattr(args, "qwen38_variant", None) in ("dsv4", "dsv4-vllm"):
            from dsv41_repl import VISUAL_GUIDANCE  # First-N tool images: the prompt only appends.
        else:
            from qwen38_repl_vision import VISUAL_GUIDANCE
        system += VISUAL_GUIDANCE
    if getattr(args, "tir_repl_review", False):
        system += REVIEW_GUIDANCE
    if getattr(args, "tir_repl_candidate_pool", False):
        system += DIVERSITY_GUIDANCE
    if grid_sep(args):
        system += "\nGrids in this conversation are written one row per line, with a space between cells.\n"
    if getattr(args, "tir_repl_train_mismatch_images", False):
        system += ("\ncheck(fn) also attaches a labeled image of its first visible-training mismatch "
                   "per cell, bounded to a local crop where necessary. Use the numeric coordinates "
                   "and crop origins; images do not contain real test answers.\n")
    if getattr(args, "tir_repl_solver_by_tokens", 0):
        system += (f"\n# Early executable hypothesis\nAim to implement and check a provisional solve(grid) "
                   f"within {args.tir_repl_solver_by_tokens} generated tokens, before trying to perfect the rule. "
                   "Failures against visible training pairs are useful; measure them and revise one assumption. "
                   "A function definition or prose claim alone does not count: execute check(solve).\n")
    if getattr(args, "tir_repl_solver_max_turns", 0):
        system += ("\n# Executable solver gate\n" + solver_checkpoint_notice(args)
                   + "\npropose_solver is also available before the checkpoint.\n")
    if getattr(args, "tir_repl_auto_check_solver", False):
        system += ("\n# Automatic executable feedback\nEach Python cell defining solve(grid), or calling "
                   "check(solve), is followed by a controller-owned check of the CURRENT function on all "
                   "visible training pairs. This cannot be bypassed by rebinding check. Implement a real "
                   "provisional rule early; use failing pairs to revise its mechanism or repair an error. "
                   "Do not write a dummy function or memorize examples. Checks do not submit, certify test "
                   "answers, or add tokens. Keep a saved solution intact while testing named alternatives.\n")
    if getattr(args, "tir_repl_solver_recheck_tokens", 0):
        system += (f"\nAfter successful helper/data edits, the harness also rechecks the CURRENT solve "
                   f"against all visible training pairs when at least {args.tir_repl_solver_recheck_tokens} "
                   "generated tokens have elapsed since the last solver check. This is isolated feedback, "
                   "not a submission, and stops after a candidate has been saved.\n")
    if getattr(args, "tir_repl_code_reserve", 0):
        system += ("\n# Deliver executable code\nThe final " + str(args.tir_repl_code_reserve)
                   + " generated tokens are reserved within the existing finalization budget for "
                   "code delivery and concrete repairs, not more exploration. Use finalize_solver(code) "
                   "to define, check and submit your best current solver in one call. You may use this "
                   "tool earlier too. It does not accept a failing solver or certify real test answers.\n")
    if getattr(args, "tir_repl_targeted_review", False):
        system += "\n# Bounded executable review\n" + TARGETED_REVIEW + "\n"
    if getattr(args, "tir_repl_input_audit", False):
        system += (interaction_audit.GUIDANCE if getattr(args, "tir_repl_interaction_audit", False)
                   else execution_audit.GUIDANCE if getattr(args, "tir_repl_execution_audit", False)
                   else input_audit.GUIDANCE)
    return system


def grid_sep(args):
    return " " if getattr(args, "tir_repl_grid_text", "compact") == "spaced" else ""


def fmt(grid, sep=""):
    return "\n".join(sep.join(str(v) for v in row) for row in grid)


def render_task(task, sep=""):
    parts = []
    for i, pair in enumerate(task["train"]):
        a, b = pair["input"], pair["output"]
        parts.append(f"## Example {i}\nInput ({len(a)}x{len(a[0])}):\n{fmt(a, sep)}\n"
                     f"Output ({len(b)}x{len(b[0])}):\n{fmt(b, sep)}")
    for i, pair in enumerate(task["test"]):
        a = pair["input"]
        parts.append(f"## Test input {i}\n({len(a)}x{len(a[0])}):\n{fmt(a, sep)}")
    return ("Solve this puzzle. The same grids are loaded in the python session as train and test_inputs.\n\n"
            + "\n\n".join(parts))


def public(task):
    # Deliberately project fields: test outputs never reach the session or the prompt.
    return deepcopy({"train": [{"input": p["input"], "output": p["output"]} for p in task["train"]],
                     "test": [{"input": p["input"]} for p in task["test"]]})


def split_holdout(task, enabled=False):
    """Deterministic split of demonstrations only; actual test labels are always discarded."""
    visible = public(task)
    held = visible["train"].pop(0) if enabled and len(visible["train"]) >= 3 else None
    return visible, held


HOLDOUT_PROMPT = (
    "\n\n[harness validation] One additional demonstration was reserved before this session. Its input "
    "and output are not available in the prompt, train, test_inputs or check(). The examples above are "
    "the visible training set, renumbered from zero. On submit, after the visible pairs pass, your same "
    "function is tested on the reserved input. Acceptance requires that check too. A failure returns only "
    "aggregate shape/cell mismatch feedback, not the hidden grid. Revise the general rule rather than "
    "guessing hidden cells; do not request or try to recover the reserved data. Real test answers are "
    "never used for this feedback.")


# ----- chat template -----------------------------------------------------------------------


def clean(text):
    """Model or cell text made safe for /tokenize with special parsing: a literal <|im_end|> stays text,
    and lone surrogates become replacement characters."""
    return SPECIAL_TEXT.sub(r"<\\|\1|>", text.encode("utf-8", "replace").decode("utf-8"))


def tojson(value):
    return json.dumps(value, ensure_ascii=False)


def render_system(system, tools, effort):
    text = "<|im_start|>system\n" + (EFFORT[effort] + "\n\n" if EFFORT[effort] else "")
    text += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    text += "".join("\n" + tojson(tool) for tool in tools) + "\n</tools>" + TOOL_FORMAT
    system = system.strip()
    return text + ("\n\n" + system if system else "") + "<|im_end|>\n"


def render_turns(messages):
    """The template's text for the messages after the first user turn, without the generation prompt.
    Every assistant turn keeps a think block (preserve_thinking is undefined in our requests)."""
    out = []
    for i, m in enumerate(messages):
        before = messages[i - 1]["role"] if i else "user"
        after = messages[i + 1]["role"] if i + 1 < len(messages) else None
        content = clean(m.get("content") or "").strip()
        if m["role"] == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif m["role"] == "assistant":
            reasoning = clean(m.get("reasoning_content") or "").strip()
            out.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}")
            for k, call in enumerate(m.get("tool_calls") or []):
                fn = call["function"]
                out.append(("\n\n" if content else "") if k == 0 else "\n")
                out.append(f"<tool_call>\n<function={clean(fn['name'])}>\n")
                for key, value in fn["arguments"].items():
                    out.append(f"<parameter={clean(key)}>\n{clean(value) if isinstance(value, str) else tojson(value)}"
                               "\n</parameter>\n")
                out.append("</function>\n</tool_call>")
            out.append("<|im_end|>\n")
        elif m["role"] == "tool":
            if before != "tool":
                out.append("<|im_start|>user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if after != "tool":
                out.append("<|im_end|>\n")
        else:
            raise ValueError(f"Unexpected message role {m['role']!r}")
    return "".join(out)


def prompt_text(prefix, messages):
    """prefix: the served rendering of system+tools+task without the generation prompt."""
    return prefix + render_turns(messages) + GENERATION_PROMPT


def wire_messages(messages, *, escape_text=False):
    """OpenAI wire arguments; escape literal special tokens before JSON serialization.

    Stored tool arguments are mappings. Escaping their serialized JSON instead
    would create invalid backslash escapes in code containing '<|im_end|>'.
    """
    def escaped(value):
        if isinstance(value, str):
            return clean(value)
        if isinstance(value, list):
            return [escaped(v) for v in value]
        if isinstance(value, dict):
            return {k: escaped(v) for k, v in value.items()}
        return value

    tail = deepcopy(list(messages))
    if escape_text:
        tail = escaped(tail)
    for message in tail:
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    return tail


def template_request(system, user, effort, messages=(), *, tools=None):
    kwargs = dict(enable_thinking=True, reasoning_effort=effort)
    head = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return dict(messages=head + wire_messages(messages), tools=TOOLS if tools is None else tools,
                chat_template_kwargs=kwargs, reasoning_effort=effort)


@dataclass(frozen=True)
class CheckpointPrefix:
    """Immutable task head for native per-turn rendering, plus its initial raw text."""
    request: dict
    text: str

    def with_history(self, messages):
        request = deepcopy(self.request)
        request["messages"] += wire_messages(messages, escape_text=True)
        return request


def native_generation_prompt(text):
    """Allow a thinking prefix or an open assistant turn that emits <think> itself."""
    _, marker, tail = text.rpartition("<|im_start|>assistant\n")
    if not marker or tail.strip() not in ("", "<think>"):
        raise ValueError(f"REPL checkpoint template has an unsupported generation prompt: {text[-80:]!r}")
    return marker + tail


class LlamaServer:
    """llama.cpp: /apply-template renders, /tokenize parses special text, /completion streams token ids."""

    def __init__(self, client, url):
        self.client, self.url = client, url

    async def render(self, request):
        response = await self.client.post(self.url + "/apply-template", json=request, timeout=120)
        response.raise_for_status()
        return response.json()["prompt"]

    async def tokenize(self, text):
        return await dsv4_llama.tokenize(self.client, self.url, text)

    async def complete(self, payload, output, timeout, *, specials, progress, thinking, **guard):
        return await dsv4_llama.stream_completion(self.client, self.url, payload, output, timeout, parse_grid=None,
            specials=specials, progress=progress, thinking=thinking, marker="</think>", **guard)


async def append_prompt(server, prompt, tokens):
    """Continue a text prompt, or rebuild a multimodal assistant prefill with its images intact."""
    if hasattr(server, "append_prompt"):
        return await server.append_prompt(prompt, tokens)
    return prompt + tokens


def with_ids(server, prompt, ids):
    """Exact ids for the request that rendered prompt (a multimodal prompt keeps its images)."""
    return server.with_ids(prompt, ids) if hasattr(server, "with_ids") else ids


class VllmServer:
    """vLLM: the chat form of /tokenize renders (read back with /detokenize), the prompt form encodes text,
    and /v1/completions streams token ids (return_token_ids). stream.jsonl keeps llama.cpp's event lines,
    so dsv4_llama.read_partial reads the generated ids of either server."""

    def __init__(self, client, url, model, *, native_template=False):
        self.client, self.url, self.model = client, url, model
        self.native_template = native_template

    async def post(self, path, body):
        response = await self.client.post(self.url + path, json=dict(body, model=self.model), timeout=120)
        response.raise_for_status()
        return response.json()

    async def render_tokens(self, request):
        # vLLM passes an assistant message's reasoning to the template from its "reasoning" field.
        messages = [{("reasoning" if k == "reasoning_content" else k): v for k, v in m.items()}
                    for m in request["messages"]]
        return (await self.post("/tokenize", dict(messages=messages, tools=request["tools"],
                                                 add_generation_prompt=request.get("add_generation_prompt", True),
                                                 chat_template_kwargs=request["chat_template_kwargs"])))["tokens"]

    async def render(self, request):
        ids = await self.render_tokens(request)
        return (await self.post("/detokenize", dict(tokens=ids)))["prompt"]

    async def tokenize(self, text):
        return (await self.post("/tokenize", dict(prompt=text, add_special_tokens=False)))["tokens"]

    async def specials(self):
        ids = {}
        for name, text in qwen38_llama.SPECIALS:
            tokens = await self.tokenize(text)
            if len(tokens) != 1:
                raise ValueError(f"{text!r} is not a single token in the served vocabulary: {tokens}")
            ids[name] = tokens[0]
        return ids

    async def complete(self, payload, output, timeout, *, specials, progress, thinking, loop_guard=None,
                       loop_action="off", chat_request=None, generate=None):
        """Stream one /v1/completions request; the result has dsv4_llama.stream_completion's fields.

        loop_guard (qwen38_loop.GzipLoopGuard) reads the delta token ids; with loop_action "stop" a confirmed
        hit closes the stream, which aborts the request on the server. generate (a GenerateRequest body with
        one placeholder per image) streams /inference/v1/generate instead, which must expand it back to
        exactly payload["prompt"]."""
        if loop_guard is not None and loop_action not in ("observe", "stop"):
            raise ValueError("A gzip guard needs observe or stop mode")
        payload = dict(payload, model=self.model, stream=True, stream_options=dict(include_usage=True),
                       return_token_ids=True)
        prompt_tokens = len(payload["prompt"])
        endpoint, expanded = "/v1/completions", None
        if chat_request is not None:
            payload.pop("prompt")
            payload.update(chat_request)
            # Keep XML tool calls in the content stream; our parser executes them.
            payload["tool_choice"] = "none"
            endpoint = "/v1/chat/completions"
        elif generate is not None:
            expanded = list(payload["prompt"])
            payload = dict(generate, model=self.model, stream=True, stream_options=dict(include_usage=True),
                           return_token_ids=True, output_mode="text")
            endpoint = "/inference/v1/generate"
        output.mkdir()
        progress.update(prompt_tokens=prompt_tokens, decoded=0, started=time.monotonic())
        start = time.monotonic()
        think_end = specials["think_end"]
        reasoning, answer, think_closed = [], [], not thinking
        usage, finish, stop_reason, error, cancelled, done = {}, None, None, None, False, False
        first, last, count, think_end_tokens, loop_stopped = None, None, 0, 0, False
        generated_ids = []
        try:
            async with asyncio.timeout(timeout):
                async with self.client.stream("POST", self.url + endpoint, json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread())[:2000].decode("utf-8", "replace")
                        raise RuntimeError(f"HTTP {response.status_code}: {body}")
                    with (output / "stream.jsonl").open("x", buffering=1) as raw:
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                break
                            event = json.loads(data)
                            elapsed = time.monotonic() - start
                            if event.get("error") or event.get("object") == "error":
                                raise RuntimeError(str(event.get("error") or event.get("message")))
                            if expanded is not None and event.get("choices"):
                                # The first token-bearing chunk carries the prompt the engine actually ran.
                                if event.get("prompt_token_ids") != expanded:
                                    raise RuntimeError("The server expanded the exact-id prompt differently "
                                                       f"({len(event.get('prompt_token_ids') or [])} vs "
                                                       f"{len(expanded)} ids); no generation from wrong context")
                                expanded = None
                            usage = event.get("usage") or usage
                            for choice in event.get("choices") or []:
                                tokens, text = choice.get("token_ids"), choice.get("text") or ""
                                if chat_request is not None:
                                    delta = choice.get("delta") or {}
                                    text = (delta.get("reasoning") or delta.get("reasoning_content") or "") + (delta.get("content") or "")
                                    if tokens is None and not text and not delta.get("tool_calls"):
                                        tokens = []  # Initial role-only or final stop event.
                                if tokens is None:
                                    raise RuntimeError("The server omitted token_ids; REPL turns need return_token_ids")
                                generated_ids.extend(tokens)
                                count += len(tokens)
                                raw.write(json.dumps([round(elapsed, 3), count, tokens, text]) + "\n")
                                think_end_tokens += tokens.count(think_end)
                                if think_closed:
                                    answer.append(text)
                                elif "</think>" in text:
                                    head, tail = text.split("</think>", 1)
                                    reasoning.append(head)
                                    answer.append(tail)
                                    think_closed = True
                                else:
                                    reasoning.append(text)
                                    think_closed = think_end in tokens
                                if tokens or text:
                                    first = elapsed if first is None else first
                                    last = elapsed
                                finish = choice.get("finish_reason") or finish
                                stop_reason = choice.get("stop_reason", stop_reason)
                                progress["decoded"] = count
                                hit = loop_guard.feed(tokens) if loop_guard is not None else None
                                if hit:
                                    dsv4_llama.write_json(output / "loop_detection.json", dict(action=loop_action, **hit))
                                    print(f"GZIP {output.name}: ratio={hit['ratio']:.2f}, action={loop_action}, "
                                          f"token={hit['observed_completion_tokens']}", flush=True)
                                    # A response that already finished keeps its natural finish.
                                    if loop_action == "stop" and finish is None:
                                        loop_stopped = True
                            if loop_stopped:
                                break  # Closing the stream aborts this request, not the server.
                    if not loop_stopped and (not done or finish is None):
                        raise RuntimeError("Incomplete SSE: missing DONE or finish reason")
        except asyncio.CancelledError:
            error, cancelled = "Cancelled: pipeline stopped or server exited", True
        except TimeoutError:
            error = f"TimeoutError: request budget {timeout:.0f} s (deadline or --request-timeout-seconds)"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        reasoning, content = "".join(reasoning), "".join(answer)
        if chat_request is not None and generated_ids and not cancelled:
            # Reasoning parsers hide delimiter text. Recover the exact completion from its IDs
            # so partial XML calls and thinking-cap continuations use the same contract as text.
            try:
                decoded = (await self.post("/detokenize", dict(tokens=generated_ids)))["prompt"]
                (output / "raw_completion.txt").write_text(decoded)
                if thinking:
                    reasoning, marker, content = decoded.partition("</think>")
                    think_closed = bool(marker)
                else:
                    reasoning, content, think_closed = "", decoded, True
                for token in ("<|im_end|>", "<|endoftext|>"):
                    content = content.removesuffix(token)
                if usage.get("prompt_tokens") is not None and usage["prompt_tokens"] != prompt_tokens:
                    raise RuntimeError(f"Multimodal token count differs: counted={prompt_tokens}, "
                                       f"served={usage['prompt_tokens']}")
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
        if self.native_template and reasoning.lstrip().startswith("<think>"):
            # The 9B template may leave this opening tag for the model to emit.
            # Raw stream ids/text remain intact; histories store reasoning only.
            reasoning = reasoning.lstrip()[len("<think>"):]
        if loop_stopped:
            finish = "loop_detected"
        stop_type = ("limit" if finish == "length" else None if finish != "stop" else
                     "word" if isinstance(stop_reason, str) else "eos")
        decode_rate = count / (last - first) if first is not None and last is not None and last > first else None
        result = dict(content=content, reasoning=reasoning, usage=usage, finish_reason=finish, stop_type=stop_type,
                      truncated=None, think_closed=think_closed, think_end_tokens=think_end_tokens, tool_calls=[],
                      stream_done=done, error=error, cancelled=cancelled, final_grid=None, streamed_tokens=count,
                      elapsed_s=time.monotonic() - start, first_content_s=first, last_content_s=last,
                      timings=dict(predicted_per_second=decode_rate) if decode_rate else None,
                      generation_settings=None, decode_tokens_per_s_estimate=decode_rate)
        if loop_guard is not None:
            result["loop_detection"] = dict(action=loop_action, stopped=loop_stopped, **loop_guard.snapshot())
        dsv4_llama.write_json(output / "result.json", result)
        (output / "reasoning.txt").write_text(reasoning)
        (output / "answer.txt").write_text(content)
        return result


async def apply_template(server, system, user, effort, messages=(), *, tools=None):
    return await server.render(template_request(system, user, effort, messages, tools=tools))


def split_tools(text):
    """(text with the <tools> JSON lines removed, the parsed tool list): llama.cpp may space JSON differently."""
    match = re.search(r"<tools>\n(.*?)\n</tools>", text, re.S)
    if match is None:
        return text, None
    try:
        tools = [json.loads(line) for line in match.group(1).split("\n")]
    except json.JSONDecodeError:
        return text, None
    return text[:match.start(1)] + text[match.end(1):], tools


async def render_prefix(server, system, task, effort, specials, *, holdout_first=False, tools=None,
                        images=None, mm_kwargs=None, image_note=None, test_reveal="all", grid_sep=""):
    """Return a native CheckpointPrefix (or legacy text without the generation prompt), and token ids."""
    visible, held = split_holdout(task, holdout_first)
    visible, _ = disclosure.initial_view(visible, test_reveal)
    user = (render_task(visible, grid_sep) + (HOLDOUT_PROMPT if held is not None else "")
            + disclosure.initial_note(task, test_reveal))
    if images is not None:
        user = [{"type": "text", "text": user + "\n\n" + (image_note or "")}] + [
            part for image in images for part in image.content_parts()]
    if hasattr(server, "render_prefix"):  # A server with its own encoder (DeepSeek-V4.1) checks its own prompt.
        return await server.render_prefix(template_request(system, user, effort, tools=tools), specials)
    if getattr(server, "native_template", False):
        request = template_request(system, user, effort, tools=tools)
        if mm_kwargs:
            request["mm_processor_kwargs"] = mm_kwargs
        ids = await server.render_tokens(request)
        text = (await server.post("/detokenize", dict(tokens=ids)))["prompt"]
        native_generation_prompt(text)
        if not ids or ids[0] != specials["im_start"]:
            raise ValueError("REPL checkpoint prompt must start with the served <|im_start|> token")
        return CheckpointPrefix(request, text), ids
    text = await apply_template(server, system, user, effort, tools=tools)
    if not text.endswith(GENERATION_PROMPT):
        raise ValueError(f"REPL prompt does not end with the thinking generation prompt: {text[-80:]!r}")
    ids = await server.tokenize(text)
    if len(ids) < 4 or ids[0] != specials["im_start"] or specials["think"] not in ids[-2:] or specials["think_end"] in ids:
        raise ValueError("REPL prompt does not tokenize to <|im_start|> ... assistant <think>")
    return text[:-len(GENERATION_PROMPT)], ids


SAMPLE_TASK = {"train": [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}], "test": [{"input": [[0, 0], [2, 0]]}]}
SAMPLE_TURNS = [
    {"role": "assistant", "content": "Check the shapes first.", "reasoning_content": "Mirror?",
     "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "python", "arguments": {
         "code": "print(train[0]['input'].shape)\ndef solve(g):\n    return g[:, ::-1]"}}}]},
    {"role": "tool", "tool_call_id": "c0", "content": "(2, 2)"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "python", "arguments": {"code": "check(solve)"}}},
        {"id": "c2", "type": "function", "function": {"name": "submit", "arguments": {"function_name": "solve"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "pair 0: PASS\n1/1 training pairs pass"},
    {"role": "tool", "tool_call_id": "c2", "content": "Accepted: solve reproduces all 1 training outputs."},
    {"role": "assistant", "content": "Done."},
    {"role": "user", "content": NUDGE_NO_SUBMIT},
]


async def check_template(server, system, effort, *, tools=None):
    """Validate native tool round trips, or compare the legacy local rendering to the server.

    The legacy prefix must match because later turns use the local renderer;
    a legacy multi-turn difference is reported but is not fatal.
    """
    tools = TOOLS if tools is None else tools
    if hasattr(server, "check_template"):
        return await server.check_template(system, effort, tools=tools)
    user = render_task(SAMPLE_TASK)
    served = await apply_template(server, system, user, effort, tools=tools)
    if getattr(server, "native_template", False):
        # Never splice Flash-Next turns into a different checkpoint's template.
        suffix = native_generation_prompt(served)
        _, served_tools = split_tools(served)
        if served_tools != tools:
            raise ValueError("The checkpoint template does not preserve the REPL tool schemas")
        multi = await apply_template(server, system, user, effort, SAMPLE_TURNS, tools=tools)
        native_generation_prompt(multi)
        # Inspect only assistant history, not tool syntax examples in the system prompt.
        _, marker, history = multi.partition("<|im_start|>assistant\n")
        _, calls, malformed = parse_calls(history)
        expected = [(c["function"]["name"], c["function"]["arguments"])
                    for m in SAMPLE_TURNS for c in m.get("tool_calls", [])]
        if not marker or malformed or calls != expected:
            raise ValueError("The checkpoint template does not round-trip the REPL XML tool calls")
        if any(m["content"] not in multi for m in SAMPLE_TURNS if m["role"] == "tool"):
            raise ValueError("The checkpoint template omitted REPL tool feedback")
        return dict(renderer="checkpoint-every-turn", prefix_exact=None, turns_exact=None,
                    tools_equal=True, turns_checked=True, tool_calls_roundtrip=True,
                    reasoning_preserved="Mirror?" in history,
                    generation_prompt=suffix, prefix_sha256=hashlib.sha256(served.encode()).hexdigest(),
                    turns_sha256=hashlib.sha256(multi.encode()).hexdigest())
    local = render_system(system, tools, effort) + f"<|im_start|>user\n{user.strip()}<|im_end|>\n" + GENERATION_PROMPT
    served_rest, served_tools = split_tools(served)
    local_rest, local_tools = split_tools(local)
    report = dict(prefix_exact=served == local, tools_equal=served_tools == local_tools)
    if served_rest != local_rest or served_tools != local_tools:
        raise ValueError("The served chat template renders the REPL system/tools/user prefix differently from "
                         f"the Flash-Next template this module renders later turns in:\n{served!r}\n!=\n{local!r}")
    try:
        served = await apply_template(server, system, user, effort, SAMPLE_TURNS, tools=tools)
    except Exception as exc:  # A server that rejects tool history only loses the comparison.
        report.update(turns_checked=False, turns_error=f"{type(exc).__name__}: {exc}"[:500])
        return report
    local = local[:-len(GENERATION_PROMPT)] + render_turns(SAMPLE_TURNS) + GENERATION_PROMPT
    report.update(turns_checked=True, turns_exact=served == local, reasoning_preserved="Mirror?" in served)
    if served != local:
        at = next((i for i, (a, b) in enumerate(zip(served, local)) if a != b), min(len(served), len(local)))
        report.update(turns_diff_at=at, served_excerpt=served[max(0, at - 200):at + 200],
                      local_excerpt=local[max(0, at - 200):at + 200])
    return report


# ----- tool calls --------------------------------------------------------------------------


def parse_calls(content):
    """The answer after </think>: (text before the first call, [(name, {param: value})], malformed)."""
    head, opened, _ = content.partition("<tool_call>")
    calls = [(m.group(1).strip(), {p.group(1).strip(): p.group(2) for p in PARAM.finditer(m.group(2))})
             for m in CALL.finditer(content)]
    return head.strip(), calls, bool(opened) and content.count("<tool_call>") != len(calls)


@dataclass(frozen=True)
class Dialect:
    """The served model's tool-call and thinking-close markup; Qwen's XML calls unless the server has its own."""
    call_open: str
    call_close: str
    think_close: str  # Closes a capped turn's thinking, which then continues as its answer.
    empty_think: str  # Closes an open think block for a note repair without new thinking.
    nudge_format: str
    nudge_optional: str
    parse: object  # parse_calls's contract: (head, [(name, {param: text})], malformed).


QWEN = Dialect(call_open="<tool_call>", call_close="</tool_call>", think_close=THINK_CLOSE,
               empty_think="\n</think>\n\n", nudge_format=NUDGE_FORMAT,
               nudge_optional=("Call python with <tool_call><function=python><parameter=code>\n"
                               "print(1)\n</parameter></function></tool_call>. progress_note is optional."),
               parse=parse_calls)


def extract_code(arguments, key):
    value = arguments.get(key)
    if value is None:
        # A missing code/function parameter must never execute the progress note as Python.
        strings = [v for k, v in arguments.items() if k != "progress_note" and isinstance(v, str)]
        value = strings[0] if strings else None
    if key == "code":
        if not isinstance(value, str) or not value.strip():
            return None, "python needs a non-empty code string."
        fenced = re.fullmatch(r"\s*```(?:python)?\n(.*?)\n?```\s*", value, re.S)
        return (fenced.group(1) if fenced else value), None
    return (value if isinstance(value, str) and value.strip() else "solve").strip(), None


def validate_progress_note(text):
    """Validate the visible memory contract, not the truth of the model's hypothesis.

    Permit markdown bullets/labels and multiline fields. Reject empty/template/generic preambles;
    an explicitly untested hypothesis is valid (we must not force fabricated measurements).
    """
    if not isinstance(text, str) or len(text) > 1800 or any(x in text for x in ("<think", "</think", "<tool_call",
                                                                                "｜DSML｜", "```")):
        return None
    pattern = r"(?i)(?<!\S)(?:[-*]\s+)?\*{0,2}(Known|Hypothesis|Next test)\*{0,2}:\*{0,2}\s*"
    fields = list(re.finditer(pattern, text))
    if [m.group(1).lower() for m in fields] != ["known", "hypothesis", "next test"]:
        return None
    values = [" ".join(text[m.end():fields[i + 1].start() if i + 1 < 3 else len(text)].split())
              for i, m in enumerate(fields)]
    for value in values:
        if (not 8 <= len(value) <= 600 or value.lower().strip(". ") in ("todo", "placeholder", "not sure")
                or re.match(r"(?i)^(?:let me|let's|i (?:will|want to|need to|shall))\b", value)):
            return None
    return "\n".join(f"{label}: {value}" for label, value in zip(("Known", "Hypothesis", "Next test"), values))


def progress_note_for_calls(content, calls):
    """A visible note can cover a batch; otherwise every call must carry its required parameter."""
    visible = validate_progress_note(content)
    if visible:
        return visible, "visible"
    notes = [validate_progress_note(args.get("progress_note")) for _, args in calls]
    if notes and all(notes):
        return "\n\n".join(dict.fromkeys(notes)), "tool_argument"
    return None, None


# ----- python session ----------------------------------------------------------------------


async def run_child(request, limit):
    """One sandboxed runtime call: its JSON report, or None if it timed out, crashed or broke the protocol."""
    data = json.dumps(request).encode()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, *PYTHON_FLAGS, str(RUNTIME), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, close_fds=True,
        start_new_session=True, cwd=str(HERE), env=CHILD_ENV)
    out = bytearray()
    try:
        async with asyncio.timeout(limit):
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
            while len(out) <= MAX_CHILD_OUTPUT and (chunk := await proc.stdout.read(1 << 16)):
                out += chunk
            await proc.wait()
    except TimeoutError:
        return None
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
    if len(out) > MAX_CHILD_OUTPUT:
        return None
    lines = out.decode("utf-8", "replace").strip().splitlines()
    try:
        report = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None
    if not isinstance(report, dict) or report.get("fatal") or not report.get("sandbox_enabled"):
        return {"fatal": (report or {}).get("fatal") or "no sandboxed result"} if isinstance(report, dict) else None
    return report


@functools.cache
def auto_repair_module():
    spec = importlib.util.spec_from_file_location("havu_auto_repair", AUTO_REPAIR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repair_cell(code, error, names):
    """(code with a misspelled name rewritten, note) or None. His propose() without its side-effect check: a rerun
    replays the journal without the failed cell, so nothing the failed part did survives."""
    repair = auto_repair_module()
    error = re.sub(r'File "<cell \d+>"', 'File "<cell>"', error)  # His traceback parser expects <cell>.
    try:
        found = repair.suggest(code, error, names)
        new = found and repair.rewrite(code, *found)
    except Exception:  # noqa: BLE001 - a repair is optional; the model still gets the original error
        return None
    return (new, repair.NOTE.format(*found[:2])) if new and new != code else None


class Session:
    """A trajectory's cells. Each call starts a child that replays them before its own operation."""

    def __init__(self, task, slots, timeout, memory_mib, cpus=(), *, structure_tools=False, visual_tools=False,
                 train_mismatch_images=False, output_chars=OUTPUT_CHARS, auto_repair=0, grid_sep=""):
        self.task, self.slots, self.timeout, self.memory_mib = task, slots, timeout, memory_mib
        self.output_chars, self.auto_repair = output_chars, auto_repair
        self.cpus = sorted(cpus)
        self.structure_tools = structure_tools
        self.visual_tools, self.last_visuals = visual_tools, []
        self.train_mismatch_images = train_mismatch_images
        self.grid_sep = grid_sep
        self.journal, self.replay_seconds = [], 0.0
        self.stats = Counter()
        self.last_fatal = None
        self.last_tool_failed = False
        self.last_solver_check = {}
        self.last_checked_alternatives = []
        self.journal_test_counts = None  # Only staged disclosure changes replay visibility.
        self.deadline = None  # Optional callable: includes queue wait and the entire sandbox job.

    async def _run(self, op, task=None, *, probe_input=None, capture_alternatives_from=None,
                   journal=None, execution_trace=None, measurement_candidate=None):
        task = task or self.task
        limit = self.replay_seconds * 1.5 + self.timeout * (1 if op[0] in ("cell", "check") else 2 * (
            len(task["train"]) + len(task["test"]))) + 30
        if capture_alternatives_from:
            limit += self.timeout * 2 * (len(task["train"]) + len(task["test"]))
        request = dict(task=task, journal=self.journal if journal is None else journal, op=list(op), timeout=self.timeout,
                       memory_mib=self.memory_mib, cpu_seconds=min(7200, math.ceil(limit) + 5), cpus=self.cpus)
        if self.journal_test_counts is not None:
            request["journal_test_counts"] = self.journal_test_counts[:len(request["journal"])]
        if execution_trace is not None:
            request["execution_trace"] = execution_trace
        if measurement_candidate is not None:
            request["measurement_candidate"] = measurement_candidate
        if self.structure_tools:
            request["structure_tools"] = True
        if self.visual_tools:
            request["visual_tools"] = True
        if self.train_mismatch_images:
            request["train_mismatch_images"] = True
        if self.grid_sep:
            request["grid_sep"] = self.grid_sep
        if self.output_chars != OUTPUT_CHARS:
            request["max_output"] = self.output_chars
        if self.auto_repair:
            request["auto_repair"] = True
        if capture_alternatives_from:
            request["capture_checked_alternatives"] = capture_alternatives_from
        if probe_input is not None:
            # Only the input goes to an isolated evaluator. Its namespace/journal still use the
            # visible task; the held-out answer never enters a model-code process.
            request["probe_input"] = probe_input
        queued = time.monotonic()
        wall_left = max(0, self.deadline() - queued) if self.deadline else None
        try:
            async with asyncio.timeout(wall_left):
                async with self.slots:
                    started = time.monotonic()
                    self.stats["slot_wait_s"] += started - queued
                    try:
                        result = await run_child(request, limit)
                    finally:
                        self.stats["exec_s"] += time.monotonic() - started
        except TimeoutError:
            self.stats["wall_timeouts"] += 1
            return None  # run_child's finally kills/reaps its sandbox on cancellation.
        if result is not None and result.get("fatal"):
            self.stats["fatal"] += 1
            self.last_fatal = result["fatal"]
            return None
        return result

    async def cell(self, code, *, capture_alternatives_from=None):
        self.last_visuals = []
        self.last_solver_check = {}
        self.last_checked_alternatives = []
        self.stats["cells"] += 1
        result = await self._run(("cell", code), capture_alternatives_from=capture_alternatives_from)
        notes = []
        while result and not result["timeout"] and result["error"] and len(notes) < self.auto_repair:
            fix = await asyncio.to_thread(repair_cell, code, result["error"], result.get("names", ()))
            if fix is None:
                break
            self.stats["auto_repair_reruns"] += 1
            rerun = await self._run(("cell", fix[0]), capture_alternatives_from=capture_alternatives_from)
            if rerun is None or rerun["timeout"]:
                break  # Keep the original failure.
            code, result = fix[0], rerun
            notes.append(fix[1])
        if notes:
            self.stats["auto_repair_cells"] += 1
            self.stats["auto_repair_ok"] += not result["error"]
        self.last_tool_failed = result is None or bool(result.get("timeout") or result.get("error"))
        if result is None or result["timeout"]:
            self.stats["timeouts"] += 1
            return (f"{(result or {}).get('output', '')}\n[cell stopped: it exceeded the {self.timeout:g}s time "
                    "limit or crashed the process. Session state is restored to before this cell.]").strip()
        self.journal.append(code)
        if self.journal_test_counts is not None:
            self.journal_test_counts.append(len(self.task["test"]))
        self.last_solver_check = result.get("solver_check", {})
        self.last_checked_alternatives = result.get("checked_alternatives", [])
        self.last_visuals = result.get("visuals", [])
        self.replay_seconds += result["seconds"]
        self.stats["suppressed_training_grids"] += result.get("suppressed_training_grids", 0)
        text = result["output"]
        if result["error"]:
            self.stats["errors"] += 1
            text = f"{text}\n{result['error']}{environment_hint(result['error'])}".strip()
        restored = result.get("restored_inputs", [])
        if restored:
            self.stats["input_repairs"] += 1
            text += ("\n[session help] Restored reserved " + ", ".join(restored)
                     + " from the original puzzle after this cell. Keep transformed data in a different "
                       "variable; your other variables and functions are retained.\n" + DATA_HELP)
        return "".join(note + "\n" for note in notes) + (text or "(no output)")

    async def evaluate(self, name, task=None):
        """Replay the journal on `task` (default: the session's own) and run `name` on its inputs."""
        result = await self._run(("eval", name), task) or {"error": "evaluation exceeded its time limit or crashed"}
        self.last_tool_failed = bool(result.get("error")) or any(
            c.get("error") for split in ("train", "test") for c in result.get(split, []))
        return result

    async def measure_inputs(self, name, *, candidate_name=None):
        self.stats["input_audits"] += 1
        return await self._run(("measure", name), measurement_candidate=candidate_name) or {
            "error": "measurement timed out or sandbox failed"}

    async def trace_saved(self, entry, spec):
        """No live redefinition can alter the accepted solver's execution audit."""
        self.stats["execution_audits"] += 1
        length = entry.get("journal_length")
        if type(length) is not int or not 0 <= length <= len(self.journal):
            return {"error": "Saved solver journal snapshot unavailable", "complete": False}
        return await self._run(("trace", entry["function"]), journal=self.journal[:length],
                               execution_trace=spec) or {"error": "Execution trace timed out or sandbox failed", "complete": False}

    async def propose_solver(self, code, *, require_definition=True):
        """Execute code, then independently check solve on visible training inputs only.

        The check is controller-owned (not an appended call to a model-rebindable
        name), in an isolated replay. Its side effects are deliberately not journaled.
        The function/code state is retained normally, and submit remains separate.
        """
        self.last_solver_check, self.last_visuals = {}, []
        if require_definition and solver_code_kind(code) != "definition":
            self.last_tool_failed = True
            return "Not executed: propose_solver requires a top-level def solve(grid), with any needed helpers."
        feedback = await self.cell(code)
        # A syntax/runtime failure must not check an old solve and unlock this proposal.
        if self.last_tool_failed:
            self.last_solver_check = {}
            return feedback
        return feedback + "\n" + await self.check_solver()

    async def check_solver(self, *, periodic=False):
        """Check current solve and all replayed helpers, without journaling check side effects."""
        self.last_solver_check, self.last_visuals = {}, []
        self.stats["periodic_solver_checks" if periodic else "automatic_solver_checks"] += 1
        result = await self._run(("check", "solve"))
        self.last_tool_failed = result is None or bool(result.get("timeout") or result.get("error"))
        if result is None or result.get("timeout"):
            self.stats["timeouts"] += 1
            return f"[automatic training check stopped: timeout {self.timeout:g}s or sandbox failure]"
        self.last_solver_check = result.get("solver_check", {})
        self.last_visuals = result.get("visuals", [])
        text = result["output"]
        if result.get("error"):
            self.stats["errors"] += 1
            text += f"\n{result['error']}{environment_hint(result['error'])}"
        label = "periodic full-solver" if periodic else "automatic"
        return f"[{label} training-only check; not a submission]\n" + text

    async def evaluate_hidden_input(self, name, grid):
        self.stats["holdout_checks"] += 1
        return await self._run(("probe", name), probe_input=grid)


DATA_HELP = ("The data is already loaded: train is a list of dictionaries, not (input, output) tuples. "
             "Use pair['input'] and pair['output'], both NumPy arrays:\n"
             "for i, pair in enumerate(train):\n"
             "    inp, out = pair['input'], pair['output']\n"
             "    print(i, inp.shape, out.shape)\n"
             "test_inputs is a list of NumPy arrays. Do not retype or reload the grids.")


def environment_hint(error):
    """Actionable feedback for observed environment mistakes, without allowing more sandbox access."""
    if "ModuleNotFoundError" in error or "ImportError" in error:
        return ("\n[session help] Use the preloaded np, ndi, Counter, defaultdict, deque, math, itertools, "
                "functools and the functions objects(grid), show(*grids), check(fn). "
                "objects is a function, not an importable module. No package installation is available.")
    if "PermissionError" in error or "FileNotFoundError" in error:
        return "\n[session help] File and network access are unavailable; no input files are needed.\n" + DATA_HELP
    if any(word in error for word in ("shape", "indices", "unpack", "KeyError", "train", "test_inputs")):
        return "\n[session help] Check how you access the preloaded data.\n" + DATA_HELP
    return ""


def mismatch(got, want, detail=True, sep=""):
    """One-line summary of how got differs from want, plus maps when detail (as the runtime's check())."""
    if len(got) != len(want) or len(got[0]) != len(want[0]):
        return f"shape {len(got)}x{len(got[0])}, expected {len(want)}x{len(want[0])}"
    wrong = [(r, c, got[r][c], want[r][c]) for r in range(len(want)) for c in range(len(want[0]))
             if got[r][c] != want[r][c]]
    text = f"{len(wrong)} wrong cells: " + ", ".join(f"({r},{c}) got {g} want {w}" for r, c, g, w in wrong[:8])
    if len(wrong) > 8:
        text += ", ..."
    if detail:
        marks = "\n".join(sep.join("." if g == w else str(w) for g, w in zip(gr, wr)) for gr, wr in zip(got, want))
        text += f"\nyour output:\n{fmt(got, sep)}\nexpected values at wrong cells ('.' = correct):\n{marks}"
    return text


def sanity_notes(task, grids):
    notes = []
    train_colors = {v for p in task["train"] for row in p["output"] for v in row}
    same_shape = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])
                     for p in task["train"])
    identity = any(p["input"] == p["output"] for p in task["train"])
    for t, (pair, grid) in enumerate(zip(task["test"], grids)):
        src = pair["input"]
        if grid == src and not identity:
            notes.append(f"test {t}: the prediction equals the test input, but no training output equals its input")
        new = {v for row in grid for v in row} - train_colors - {v for row in src for v in row}
        if new:
            notes.append(f"test {t}: uses colors {sorted(new)} that appear in no training output and not in the test input")
        if same_shape and (len(grid), len(grid[0])) != (len(src), len(src[0])):
            notes.append(f"test {t}: every training output has its input's shape, but this prediction does not")
    return notes


def generalization_notes(task):
    """Input-only observations, not error flags or a score for a candidate rule.

    Do not guess a background or equate same-color components with semantic objects.
    More detailed object/relationship checks belong in the model's explicit hypothesis test.
    """
    inputs = [p["input"] for p in task["train"]]
    if not inputs:
        return []
    shapes = {(len(g), len(g[0])) for g in inputs}
    colors = {v for g in inputs for row in g for v in row}
    boundary_colors = lambda g: set(g[0] + g[-1] + [v for row in g for v in (row[0], row[-1])])
    boundaries = set().union(*(boundary_colors(g) for g in inputs))
    notes = []
    for t, pair in enumerate(task["test"]):
        grid = pair["input"]
        shape = (len(grid), len(grid[0]))
        changes = []
        if shape not in shapes:
            changes.append(f"input shape {shape[0]}x{shape[1]} was not seen in training inputs")
        new = {v for row in grid for v in row} - colors
        if new:
            changes.append(f"input colors {sorted(new)} were not seen in training inputs")
        new_boundary = boundary_colors(grid) - boundaries - new
        if new_boundary:
            changes.append(f"known colors {sorted(new_boundary)} now touch a grid edge, unlike in training inputs")
        if changes:
            notes.append(f"test {t}: " + "; ".join(changes))
    return notes


async def check_holdout(session, name, held, record, *, source):
    """Only aggregate training-holdout feedback leaves the controller. No expected grid is logged."""
    result = await session.evaluate_hidden_input(name, held["input"])
    cases = (result or {}).get("test", [])
    case = cases[0] if len(cases) == 1 else None
    report = {"passed": False, "status": "evaluation-error"}
    if result and not result.get("error") and case and not case.get("error") and case.get("grid"):
        got, want = case["grid"], held["output"]
        report.update(predicted_shape=[len(got), len(got[0])], expected_shape=[len(want), len(want[0])])
        if report["predicted_shape"] != report["expected_shape"]:
            report["status"] = "shape-mismatch"
        else:
            wrong = sum(g != w for gr, wr in zip(got, want) for g, w in zip(gr, wr))
            cells = len(want) * len(want[0])
            report.update(passed=wrong == 0, status="pass" if wrong == 0 else "cell-mismatch",
                          wrong_cells=wrong, total_cells=cells, cell_accuracy=1 - wrong / cells)
    record.setdefault("holdout_checks", []).append(dict(turn=record["turns"], source=source, **report))
    return report


def holdout_feedback(report):
    if report["status"] == "shape-mismatch":
        detail = f"predicted shape {report['predicted_shape']}, expected {report['expected_shape']}"
    elif report["status"] == "cell-mismatch":
        detail = (f"{report['wrong_cells']}/{report['total_cells']} wrong cells "
                  f"({report['cell_accuracy']:.1%} cell accuracy); shape matches")
    else:
        detail = "the function could not produce a valid grid within the evaluation limits"
    return ("Held-out demonstration: FAIL; " + detail + ". No hidden grid or cell values are revealed. "
            "The visible examples pass but do not establish the general rule. Revisit assumptions about "
            "ordering, colors, sizes, object counts and boundary cases; revise solve and submit again.")


def pending_input_review(record):
    if record.get("interaction_audit_enabled"):
        return interaction_audit.pending_request(record)
    if record.get("execution_audit_enabled"):
        return execution_audit.pending_request(record)
    if (record.get("input_audit_enabled") and record.get("accepted") is not None
            and record.get("input_audit_completed_for") != record["accepted"]):
        return input_audit.REQUEST
    return None


async def submit(session, task, name, record, review, *, holdout=None, preserve_only=False):
    result = await session.evaluate(name)
    if result.get("error"):
        if result.get("memorization_guard", {}).get("blocked"):
            record.setdefault("memorization_rejections", []).append(dict(
                turn=record["turns"], source="submit", function=name,
                completion_tokens=record.get("completion_tokens"),
                report=result["memorization_guard"]))
        hint = ("\nDefine the function in a python call first, run check(solve), then submit it. "
                "submit evaluates existing code; it does not create a function."
                if "not defined as a function" in result["error"] else "")
        return f"Not accepted: {result['error']}{hint}"
    lines, passed, detailed = [], 0, False
    for i, (pair, case) in enumerate(zip(task["train"], result["train"])):
        if case["error"]:
            lines.append(f"pair {i}: ERROR {case['error']}")
        elif case["grid"] == pair["output"]:
            passed += 1
            lines.append(f"pair {i}: PASS")
        else:
            lines.append(f"pair {i}: FAIL {mismatch(case['grid'], pair['output'], not detailed, session.grid_sep)}")
            detailed = True
    test_errors = [f"test {t}: ERROR {c['error']}" for t, c in enumerate(result["test"]) if c["error"]]
    entry = {"turn": record["turns"], "function": name, "train_passed": passed,
             "test": [c["grid"] for c in result["test"]], "source": result.get("source"),
             "identity": result.get("identity"),
             "journal_length": len(session.journal)}
    record["submissions"].append(entry)
    if record.get("test_disclosure"):
        entry["test_indices"] = list(record["test_disclosure"]["visible_indices"])
        entry["test_at_submission"] = deepcopy(entry["test"])
        entry["submission_test_indices"] = list(entry["test_indices"])
    if passed < len(task["train"]):
        return f"Not accepted: {passed}/{len(task['train'])} training pairs pass.\n" + "\n".join(lines + test_errors)
    if test_errors:
        return ("Not accepted: every training pair passes, but the function fails on a test input.\n"
                + "\n".join(test_errors))
    if record.get("fresh_challenger") and entry["test"] == record["primary_predictions"]:
        entry["duplicate_primary"] = True
        return ("Every training pair passes, but all test predictions duplicate the protected primary. "
                "No alternative saved. Test a defensible competing rule or finish in text; "
                "do not force a difference or patch test cells.")
    if holdout is not None:
        entry["holdout"] = await check_holdout(session, name, holdout, record, source="submit")
        if not entry["holdout"]["passed"]:
            return (f"Not accepted: all {len(task['train'])} visible training pairs pass.\n"
                    + holdout_feedback(entry["holdout"]))
    entry["accepted"] = True
    entry["preserve_only"] = preserve_only
    preferred = record.get("accepted")
    if record.get("input_audit_enabled") and preferred is not None and not preserve_only:
        previous = record["submissions"][preferred]
        identity = entry.get("identity") or {}
        if (identity.get("digest") and identity == previous.get("identity")
                and name == previous.get("function") and entry["test"] == previous.get("test")):
            entry["duplicate_of"] = preferred
            # Keep the raw submission event, but not a new candidate/audit identity.
            pending = pending_input_review(record)
            return ("Training checks passed again. Unchanged solver, helpers, referenced data and predictions; "
                    "the existing preferred candidate is retained. Review budgets have not restarted. "
                    + ("Its requested audit is already complete; do not repeat it. Finish in text, or check one "
                       "specific defensible alternative within the remaining review budget."
                       if pending is None else pending))
    if not preserve_only or record["accepted"] is None:
        record["accepted"] = len(record["submissions"]) - 1
    record.setdefault("first_accepted_completion_tokens", record.get("completion_tokens", 0))
    record.setdefault("first_accepted_seconds", record.get("seconds", 0.0))
    if record.get("fresh_challenger"):
        return ("All training checks passed and the test predictions differ. Alternative saved for "
                "attempt_2 only; attempt_1 remains unchanged. Test correctness is UNKNOWN. Challenger finished.")
    preds = "\n\n".join(f"Test {t} prediction ({len(g)}x{len(g[0])}):\n{fmt(g, session.grid_sep)}"
                        for t, g in enumerate(entry["test"]))
    notes = sanity_notes(task, entry["test"])
    ask = REVIEW_FOLLOWUP if review else ACCEPTED_ASK
    input_notes = []
    if review and not record.get("reviewed"):  # The structured review runs once per trajectory.
        record["reviewed"] = True
        ask = REVIEW
        input_notes = generalization_notes(task)
        entry["generalization_review"] = {
            "test_answers_used": False, "input_observations": input_notes, "requested": True}
    validation = " The held-out demonstration also passes." if holdout is not None else ""
    headline = (f"Training checks passed: {name} reproduces all {len(task['train'])} training outputs."
                f"{validation} Candidate saved; test correctness is UNKNOWN."
                if review else f"Accepted: {name} reproduces all {len(task['train'])} training outputs.{validation}")
    if preserve_only:
        headline += " Saved in the alternative pool; an existing preferred candidate was not replaced."
    if record.get("candidate_pool_enabled"):
        ask += ("\nA separate plausible training-passing function may be submitted with preserve_only=true "
                "for attempt_2. Do not resubmit identical predictions or fabricate a difference.")
    if record.get("targeted_review_enabled") and not record.get("rule_audit_completed"):
        ask = TARGETED_REVIEW
    if pending_input_review(record):
        ask = pending_input_review(record)
    if (not preserve_only and record.get("test_disclosure")
            and len(task["test"]) < record["test_disclosure"]["total"]):
        ask = ("The current candidate is saved; this is NOT test-answer verification. "
               "The controller will reveal the next test input. Keep this solver intact and briefly "
               "state the rule and unresolved assumptions when inspecting the new case.")
    observations = (("Input differences to investigate (not evidence that the prediction is wrong):\n- "
                     + "\n- ".join(input_notes) + "\n\n") if input_notes else "")
    check_label = ("Automatic checks (heuristic observations, not a correctness verdict)"
                   if review else "Automatic checks")
    return (f"{headline}\n\n{preds}\n\n" + observations
            + (check_label + ":\n- " + "\n- ".join(notes) + "\n\n" if notes else "") + ask)


async def compare_rules(session, task, arguments, record, *, holdout=None):
    """Execute an alternative, never score or infer a real test answer.

    Reuse submit's all-training/grid/holdout validation. Saved prediction snapshots
    survive subsequent mutations of the original function; the preferred index stays
    unchanged. Pool-disabled ablations record the comparison but don't select it.
    """
    session.last_tool_failed = False
    preferred = record.get("accepted")
    if preferred is None:
        session.last_tool_failed = True
        return "Save a preferred candidate with submit before comparing alternatives."
    for field in ("uncertainty", "evidence"):
        value = arguments.get(field)
        if not isinstance(value, str) or not 8 <= len(value.strip()) <= 2000:
            session.last_tool_failed = True
            return f"compare_rules needs {field}: 8..2000 characters of specific observations, not a verdict."
    event = dict(turn=record["turns"], completion_tokens=record["completion_tokens"],
                 preferred_submission=preferred, uncertainty=arguments["uncertainty"],
                 evidence=arguments["evidence"], evidence_source="model-stated, not independently verified",
                 test_answers_used=False)
    name = arguments.get("function_name")
    if record.get("execution_audit_enabled"):
        scope, discriminator = arguments.get("change_scope"), arguments.get("discriminator")
        if isinstance(scope, str):
            scope = scope.strip()
        if (scope not in execution_audit.SCOPES or not isinstance(discriminator, str)
                or not 12 <= len(discriminator.strip()) <= 2000):
            session.last_tool_failed = True
            return "compare_rules needs change_scope and a concrete discriminator (12..2000 characters)."
        if (name is None) != (scope == "none"):
            session.last_tool_failed = True
            return "Use scope='none' only when omitting function_name; otherwise describe the alternative's scope."
        event.update(change_scope=scope, discriminator=discriminator,
                     mechanism_claim="model-declared, not independently certified")
    if name is not None and (not isinstance(name, str) or not name.isidentifier()):
        session.last_tool_failed = True
        return "function_name must be an existing Python identifier, or omitted when no defensible alternative exists."
    if name is not None and name == record["submissions"][preferred].get("function"):
        session.last_tool_failed = True
        return "Use a separately named alternative function; keep the preferred function intact."
    if name is None:
        event["status"] = "no-alternative-stated"
        record.setdefault("rule_audits", []).append(event)
        record["rule_audit_completed"] = True
        if record.get("execution_audit_enabled"):
            record["mechanism_review_completed_for"] = preferred
        return ("Review explanation recorded, not independently verified. Preferred candidate retained; "
                "test correctness is still unknown. You may finish; no artificial second candidate is required.")
    before = len(record["submissions"])
    feedback = await submit(session, task, name, record, False, holdout=holdout, preserve_only=True)
    entries = record["submissions"][before:]
    entry = entries[-1] if entries else None
    event.update(function=name, status="evaluated", accepted=bool(entry and entry.get("accepted")),
                 submission_index=before if entry else None)
    record.setdefault("rule_audits", []).append(event)
    record["rule_audit_completed"] = True
    scope_note = ""
    if record.get("execution_audit_enabled"):
        if event["change_scope"] == "mechanism":
            record["mechanism_review_completed_for"] = preferred
        else:
            scope_note = ("\nThis is a model-declared small variant, not a check of the shared core mechanism. "
                          + execution_audit.MECHANISM_REQUEST)
    if not event["accepted"]:
        return "Alternative failed validation; preferred candidate retained.\n" + feedback + scope_note
    old = record["submissions"][preferred]["test"]
    differences = []
    for i, (a, b) in enumerate(zip(old, entry["test"])):
        same_shape = a is not None and b is not None and len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b))
        wrong = [(r, c, x, y) for r, (ar, br) in enumerate(zip(a, b))
                 for c, (x, y) in enumerate(zip(ar, br)) if x != y] if same_shape else None
        differences.append(dict(test_index=i, same_prediction=a == b, same_shape=same_shape,
            differing_cells=len(wrong) if wrong is not None else None,
            first_disagreements=wrong[:8] if wrong is not None else None))
    event["differences"] = differences
    event["eligible_for_attempt_2"] = bool(record.get("candidate_pool_enabled") and
                                            any(not d["same_prediction"] for d in differences))
    return ("Alternative passes all training/validation checks; this does NOT establish test correctness. "
            "Preferred candidate remains unchanged. " +
            ("Distinct predictions saved for attempt_2." if event["eligible_for_attempt_2"] else
             "No additional selected attempt: predictions identical or candidate pooling disabled.") +
            "\nPrediction disagreements (old,new values, not expected answers):\n" +
            json.dumps(differences) + "\nFinish unless a concrete executed check justifies further work." + scope_note)


def closeness(grid, want):
    if grid is None or len(grid) != len(want) or len(grid[0]) != len(want[0]):
        return 0.0
    return sum(a == b for r, s in zip(grid, want) for a, b in zip(r, s)) / (len(want) * len(want[0]))


async def fallback_eval(session, task, record, *, holdout=None):
    """No accepted submit: keep what the session's solve does, for voting as a partial answer."""
    if record["accepted"] is None and "solve" in "".join(session.journal):
        result = await session.evaluate("solve")
        if result.get("memorization_guard", {}).get("blocked"):
            record.setdefault("memorization_rejections", []).append(dict(
                turn=record["turns"], source="fallback", function="solve",
                completion_tokens=record.get("completion_tokens"),
                report=result["memorization_guard"]))
        if not result.get("error"):
            record["fallback"] = {
                "train_passed": sum(c["grid"] == p["output"] for c, p in zip(result["train"], task["train"])),
                "train_close": [closeness(c["grid"], p["output"]) for c, p in zip(result["train"], task["train"])],
                "test": [c["grid"] for c in result["test"]]}
            if record.get("test_disclosure"):
                record["fallback"]["test_indices"] = list(record["test_disclosure"]["visible_indices"])
            if holdout is not None:
                fb = record["fallback"]
                fb["holdout"] = (await check_holdout(session, "solve", holdout, record, source="fallback")
                                 if fb["train_passed"] == len(task["train"]) else
                                 {"passed": False, "status": "visible-training-failed"})


def select(records, n_test, k=2, *, require_train_pass=False, candidate_pool=False):
    """Per test output: grids ranked by votes (accepted submits, and fallback solves that pass every
    training pair), then by accepted votes, then by best fallback quality."""
    picks = []
    for t in range(n_test):
        entries = {}
        for rec in records:
            if rec.get("accepted") is not None:
                entry = rec["submissions"][rec["accepted"]]
                if rec.get("holdout_enabled") and not entry.get("holdout", {}).get("passed"):
                    continue
                grid, vote, accepted, quality = entry["test"][t], 1, 1, (1.0, 1.0)
                if grid is None:  # A staged candidate may not yet cover this original input.
                    continue
            elif rec.get("fallback") and rec["fallback"]["test"][t] is not None:
                fb = rec["fallback"]
                if rec.get("holdout_enabled") and not fb.get("holdout", {}).get("passed"):
                    continue
                n_train = len(fb["train_close"])  # A solve passing every pair counts like a submit.
                if require_train_pass and fb["train_passed"] != n_train:
                    continue
                grid, vote, accepted = fb["test"][t], int(fb["train_passed"] == n_train), 0
                quality = (fb["train_passed"] / n_train, sum(fb["train_close"]) / n_train)
            else:
                continue
            e = entries.setdefault(json.dumps(grid), {"grid": grid, "votes": 0, "accepted": 0, "best": (0.0, 0.0),
                                                      "partial": 0})
            e["votes"] += vote
            e["accepted"] += accepted
            e["partial"] += 1 - vote
            e["best"] = max(e["best"], quality)
        ranked = sorted(entries.values(), key=lambda e: (-e["votes"], -e["accepted"], -e["best"][0], -e["best"][1],
                                                         -e["partial"]))
        # Preserve legacy preferred selection. Only fill otherwise empty attempt slots with
        # distinct, explicitly accepted submissions. Each trajectory contributes at most one
        # vote per grid, regardless of repeated submits. No test answers enter this ranking.
        if candidate_pool and len(ranked) < k:
            alternatives = {}
            for rec in records:
                seen = set()
                for entry in reversed(rec.get("submissions", [])):
                    if not entry.get("accepted") or (rec.get("holdout_enabled") and
                                                     not entry.get("holdout", {}).get("passed")):
                        continue
                    grid = entry["test"][t]
                    key = json.dumps(grid)
                    if grid is None or key in entries or key in seen:
                        continue
                    seen.add(key)
                    e = alternatives.setdefault(key, dict(grid=grid, votes=0, accepted=0,
                        best=(1.0, 1.0), partial=0, candidate_pool=True))
                    e["votes"] += 1
                    e["accepted"] += 1
            ranked.extend(sorted(alternatives.values(), key=lambda e: -e["votes"]))
        picks.append(ranked[:k])
    return picks


# ----- one trajectory ----------------------------------------------------------------------


def turn_floor(args, max_out):
    """Fewest tokens of context room a turn with this cap needs."""
    if getattr(args, "tir_repl_context_stop", "min-turn") == "planned":
        return min(args.tir_repl_min_turn, max_out)
    return args.tir_repl_min_turn


def api_messages(messages, keep):
    """Drop reasoning from all but the last `keep` assistant turns (keep < 0 keeps everything)."""
    if keep < 0:
        return list(messages)
    idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    drop = set(idx[:-keep] if keep else idx)
    return [{k: v for k, v in m.items() if k != "reasoning_content"} if i in drop else m
            for i, m in enumerate(messages)]


class ToolProgressGuard:
    """Detect cross-turn stalls, not repetition inside an individual token stream.

    Execute calls before comparing their results: identical code can legitimately advance notebook
    state. Changing results reset repetition, and empty successful cells are never deemed duplicates.
    Normalized code ignores comment/whitespace-only rewrites; traceback cell numbers are not progress.
    """

    def __init__(self, repeat_limit=3, error_limit=6):
        self.repeat_limit, self.error_limit = repeat_limit, error_limit
        self.previous, self.repeats, self.errors = None, 0, 0

    def observe(self, outcomes):
        parts, observable = [], False
        for name, arguments, text, failed in outcomes:
            args = dict(arguments)
            args.pop("progress_note", None)  # Rewording the note is not a change in tool action.
            if name == "python":
                code, _ = extract_code(args, "code")
                try:
                    code = ast.dump(ast.parse(code), include_attributes=False) if code else code
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    pass
                args = {"code": code}
            if failed:
                text = re.sub(r"<cell \d+>", "<cell>", text)
                text = re.sub(r"line \d+", "line N", text)
            text = text.strip()
            observable |= bool(failed or text not in ("", "(no output)"))
            parts.append((name, args, text, bool(failed)))
        signature = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest() if observable else None
        self.repeats = (self.repeats + 1 if signature == self.previous else 1) if signature else 0
        self.previous = signature
        self.errors = self.errors + 1 if outcomes and all(o[3] for o in outcomes) else 0
        reasons, stop = [], False
        if self.repeat_limit and self.repeats >= self.repeat_limit * 2:
            reasons.append("repeated_tool_result")
            stop = True
        elif self.repeat_limit and self.repeats == self.repeat_limit:
            reasons.append("repeated_tool_result")
        if self.error_limit and self.errors >= self.error_limit:
            reasons.append("consecutive_tool_errors")
            stop = True
        elif self.error_limit and self.errors == max(1, math.ceil(self.error_limit / 2)):
            reasons.append("consecutive_tool_errors")
        if not reasons:
            return None
        return dict(action="stop" if stop else "warn", reasons=reasons,
                    repeated_turns=self.repeats, failed_turns=self.errors, signature_sha256=signature)


def stall_feedback(event):
    if event["action"] == "stop":
        return "[harness] This trajectory has stalled after recovery feedback; saving its candidates and stopping it."
    reason = (f"The same tool actions and results occurred {event['repeated_turns']} turns in a row. "
              if "repeated_tool_result" in event["reasons"] else "")
    if "consecutive_tool_errors" in event["reasons"]:
        reason += f"The last {event['failed_turns']} tool turns all failed. "
    return ("[harness recovery] " + reason + "Do not repeat the unchanged code. Read the traceback/results, "
            "explain what you will change, and test a different hypothesis or a corrected minimal cell. "
            "Use the existing session data and helpers; define solve(grid) and run check(solve) once you "
            "have a candidate rule. Continuing to stall will end this trajectory.\n" + DATA_HELP)


def solver_code_kind(code):
    """Mechanical checkpoint dispatch, never a rule-quality or correctness heuristic."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    if any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body):
        return "definition"
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "check"
           and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "solve"
           for n in ast.walk(tree)):
        return "check"
    return None


def solver_gate_active(args, record):
    return bool(getattr(args, "tir_repl_solver_max_turns", 0)
                and record.get("solver_checkpoint_started_at_tokens") is not None
                and record.get("solver_checkpoint_released_at_tokens") is None
                and record.get("first_solver_check_tokens") is None
                and record.get("accepted") is None)


def solver_checkpoint_notice(args):
    notice = BOUNDED_SOLVER_CHECKPOINT.format(turns=args.tir_repl_solver_max_turns)
    if getattr(args, "tir_repl_solver_checkpoint_action", "stop") == "resume":
        notice = notice.replace(
            "If no executable check finishes in this allowance, this trajectory stops, preserving its work.",
            "If no executable check finishes in this allowance, exploration resumes without resetting "
            "or extending the task budget. Preserve the partial implementation and repair one concrete "
            "failure instead of restarting the same analysis.")
    return notice


def answer_room(args, max_out):
    """Tokens left after a capped turn's thinking for its answer and tool call."""
    return min(getattr(args, "tir_repl_answer_room", 512), max(512, max_out // 2))


def completion_plan(args, record, *, wall_left=None):
    """Reserve generated tokens for code/check/submit; note repairs also count.

    Normal requests cannot cross the reserve boundary. Final turns always leave
    output room after thinking, even with less than the usual thinking cap left.
    The early solver checkpoint alone is soft: an unfinished tool call may use
    one bounded continuation within the remaining hard limits. Initial request
    sizes and all finalization/code/review reserves remain unchanged.
    """
    left = max(0, args.tir_repl_budget - record["completion_tokens"])
    review_budget = getattr(args, "tir_repl_review_token_budget", 0)
    first = record.get("first_accepted_completion_tokens")
    review_seconds = getattr(args, "tir_repl_review_seconds", 0)
    first_seconds = record.get("first_accepted_seconds")
    review_left = (max(0.0, first_seconds + review_seconds - record.get("seconds", 0.0))
                   if review_seconds and first_seconds is not None else None)
    review_active = bool((review_budget and first is not None) or review_left is not None)
    if review_budget and first is not None:
        left = min(left, max(0, first + review_budget - record["completion_tokens"]))
    reserve = getattr(args, "tir_repl_finalize_reserve", 0)
    wall_reserve = getattr(args, "tir_repl_finalize_reserve_seconds", 0)
    recent = [t for t in record.get("turn_log", [])[-3:] if t.get("gen_s", 0) > 0 and t.get("completion", 0) > 0]
    # Conservative recent end-to-end turn throughput.
    rate, wait = (min(t["completion"] / t["gen_s"] for t in recent) * 0.8 if recent else None), 0.0
    if getattr(args, "tir_repl_wall_rate", "turn") == "decode" and recent and all("wait_s" in t for t in recent):
        # A queued request waits once and then decodes; spreading that wait over its tokens underrates long turns.
        wait = max(t["wait_s"] for t in recent)
        decode = sum(max(1.0, t["gen_s"] - t["wait_s"]) for t in recent)
        rate = sum(t["completion"] for t in recent) / decode * 0.8
    wall_final = bool(wall_reserve and wall_left is not None and (
        wall_left <= wall_reserve or rate is not None and (wall_left - wall_reserve - wait) * rate < 1000))
    final = bool(record.get("finalization_started_at_tokens") is not None
                 or reserve and left - reserve < 1000 or wall_final)
    code_reserve = getattr(args, "tir_repl_code_reserve", 0)
    protect_code = bool(code_reserve and record.get("accepted") is None)
    delivery = bool(protect_code and (record.get("code_delivery_started_at_tokens") is not None
                                     or left - code_reserve < 1000 or wall_final))
    final = final or delivery
    checkpoint = getattr(args, "tir_repl_solver_by_tokens", 0)
    needs_solver = bool(checkpoint and record.get("first_solver_check_tokens") is None
                        and record.get("solver_checkpoint_released_at_tokens") is None
                        and record.get("accepted") is None)
    solver_due = bool(needs_solver and checkpoint - record["completion_tokens"] < 1000 and not final)
    max_out = min(args.tir_repl_turn_tokens, left)
    cap = args.tir_repl_think_tokens
    if final or solver_due:
        max_out = min(max_out, getattr(args, "tir_repl_finalize_turn_tokens", 6000))
        cap = min(getattr(args, "tir_repl_finalize_think_tokens", 2048), max(1, max_out - answer_room(args, max_out)))
    if protect_code and not delivery:
        max_out = min(max_out, max(0, left - code_reserve))
    if delivery:
        cap = min(cap, 512)
    if reserve and not final:
        max_out = min(max_out, left - reserve)
    hard_max_out = max_out
    solver_boundary = None
    if needs_solver and not solver_due and not final:
        solver_boundary = checkpoint - record["completion_tokens"]
        max_out = min(max_out, solver_boundary)
    if rate is not None:
        windows = []
        if wall_reserve and wall_left is not None:
            windows.append(wall_left - (0 if final else wall_reserve))
        if review_left is not None:
            windows.append(review_left)
        if windows:
            hard_max_out = min(hard_max_out, max(1000, int(max(0, min(windows) - wait) * rate)))
            max_out = min(max_out, hard_max_out)
    room = answer_room(args, max_out)
    if (reserve or wall_reserve or review_active or needs_solver) and (not cap or cap > max_out - room):
        cap = max(1, max_out - room)
    continuation = (min(TOOL_CONTINUATION_TOKENS, max(0, hard_max_out - max_out))
                    if solver_boundary is not None and max_out == solver_boundary else 0)
    return dict(left=left, max_out=max_out, think_cap=cap, tool_continuation_tokens=continuation,
                phase="code_delivery" if delivery else "finalization" if final else "provisional_solver" if solver_due else
                      "review" if review_active else "exploration",
                review_active=review_active, review_seconds_left=review_left,
                wall_finalization=wall_final, estimated_tokens_per_second=rate)


class Trajectory:
    """One chat with the tools. Requests and tool calls share the pipeline's client and sandbox slots."""

    def __init__(self, ctx, number, task, folder):
        self.ctx, self.number, self.task, self.folder = dict(ctx), number, task, folder
        self.disclosure = (disclosure.Disclosure(task) if
            getattr(ctx["args"], "tir_repl_test_reveal", "all") == disclosure.MODE and len(task["test"]) > 1 else None)
        if self.disclosure:
            task = self.task = self.disclosure.visible
        self.holdout = ctx.get("holdout")
        self.note_recovery_index = None
        args = ctx["args"]
        self.args = args
        self.session = Session(task, ctx["slots"], args.tir_repl_cell_timeout, args.tir_repl_memory_mib,
                               ctx.get("cpus", ()), structure_tools=getattr(args, "tir_repl_structure_tools", False),
                               visual_tools=getattr(args, "tir_repl_visual_tools", False),
                               train_mismatch_images=getattr(args, "tir_repl_train_mismatch_images", False),
                               output_chars=getattr(args, "tir_repl_output_chars", OUTPUT_CHARS),
                               auto_repair=getattr(args, "tir_repl_auto_repair", 0),
                               grid_sep=grid_sep(args))
        self.messages = []  # After the served system+task prefix.
        self.history_start = 0  # Persistent active-window offset into the unmodified raw trace.
        self.previous_prompt = []
        # The last turn's exact served ids, keyed on the history that produced them (see extend).
        self.ledger = None
        prompt_ids = getattr(args, "tir_repl_prompt_ids", None) or (
            "ledger" if getattr(ctx.get("server"), "prompt_ledger", False) else "render")
        self.ledger_enabled = (prompt_ids == "ledger"
                               and (isinstance(ctx.get("server"), VllmServer)
                                    and getattr(ctx.get("server"), "token_transport", True)
                                    or getattr(ctx.get("server"), "prompt_ledger", False)))
        self.dialect = getattr(ctx.get("server"), "dialect", QWEN)
        self.record = {"trajectory": number, "turns": 0, "completion_tokens": 0, "prompt_tokens": 0,
                       "prompt_tokens_max": 0, "submissions": [], "accepted": None, "nudges": 0, "seconds": 0.0,
                       "warned": False, "error": None, "stop": None, "context_max": 0, "context_squeezed": 0,
                       "candidate_pool_enabled": getattr(args, "tir_repl_candidate_pool", False),
                       "finalization_started_at_tokens": None,
                       "code_delivery_started_at_tokens": None,
                       "code_reserve": getattr(args, "tir_repl_code_reserve", 0),
                       "code_delivery_blocked_calls": [],
                       "solver_recheck_tokens": getattr(args, "tir_repl_solver_recheck_tokens", 0),
                       "periodic_solver_checks": [],
                       "solver_checkpoint_started_at_tokens": None, "first_solver_check_tokens": None,
                       "solver_checkpoint_max_turns": getattr(args, "tir_repl_solver_max_turns", 0),
                       "solver_checkpoint_action": getattr(args, "tir_repl_solver_checkpoint_action", "stop"),
                       "solver_checkpoint_released_at_tokens": None,
                       "auto_check_solver_enabled": getattr(args, "tir_repl_auto_check_solver", False),
                       "solver_checkpoint_turns": 0, "solver_checkpoint_blocked_calls": [],
                       "solver_checks": [], "targeted_review_enabled": getattr(args, "tir_repl_targeted_review", False),
                       "memorization_guard_version": 1, "memorization_rejections": [],
                       "rule_audits": [], "rule_audit_completed": False, "rule_audit_nudged": False,
                       "input_audit_enabled": getattr(args, "tir_repl_input_audit", False),
                       "input_audits": [], "input_audit_completed_for": None, "input_audit_nudges": 0,
                       "execution_audit_enabled": getattr(args, "tir_repl_execution_audit", False),
                       "execution_audits": [], "execution_audit_completed_for": None,
                       "mechanism_review_completed_for": None,
                       "interaction_audit_enabled": getattr(args, "tir_repl_interaction_audit", False),
                       "interaction_audits": [], "interaction_review_completed_for": None,
                       "elided_max": 0, "think_capped": 0, "loop_stops": 0, "requests": 0, "decode_seconds": 0.0,
                       "tool_calls": Counter(), "turn_log": [], "tool_guard_events": [],
                       "reasoning_context_drops": 0, "progress_note_protocol": 2,
                       "prediction_images": [],
                       "progress_notes": Counter(), "progress_note_repairs": 0,
                       "progress_note_repair_tokens": 0, "progress_note_warnings": 0,
                       "note_recovery_reasoning_turns": 0, "holdout_enabled": self.holdout is not None,
                       "holdout_original_train_index": 0 if self.holdout is not None else None,
                       "visible_train_pairs": len(task["train"]), "holdout_checks": [],
                       "context_policy": getattr(args, "tir_repl_context_policy", "legacy"),
                       "context_trims": [], "cache_reported_requests": 0, "cached_prompt_tokens": 0,
                       "cache_reported_prompt_tokens": 0,
                       "prompt_ids": prompt_ids, "prompt_ledger": Counter()}
        self.tool_guard = ToolProgressGuard(args.tir_repl_repeat_limit, args.tir_repl_error_limit)
        if self.disclosure:
            self.session.journal_test_counts = []
            self.record["test_disclosure"] = self.disclosure.state()
        self.last = None  # The last stream result, for the pipeline's llama row fields.
        self.turn_wait = 0.0  # Seconds before the first token, summed over the current turn's requests.
        self.started_at = None
        if (getattr(args, "tir_repl_review_seconds", 0)
                or getattr(args, "tir_repl_finalize_reserve_seconds", 0)
                or getattr(args, "tir_repl_solver_recheck_tokens", 0)
                or getattr(args, "tir_repl_code_reserve", 0) or self.disclosure):
            self.session.deadline = self.request_deadline

    def request_deadline(self):
        deadline = self.ctx["deadline"]
        review = getattr(self.args, "tir_repl_review_seconds", 0)
        first = self.record.get("first_accepted_seconds")
        if review and first is not None and self.started_at is not None:
            return min(deadline, self.started_at + first + review)
        return deadline

    async def encode(self, messages):
        prefix, server = self.ctx["prefix"], self.ctx["server"]
        ledger = self.ledger if self.extends(messages) else None
        if isinstance(prefix, CheckpointPrefix):
            full = await server.render_tokens(prefix.with_history(messages))
            if ledger is None:
                return full
            if "turn" not in ledger:
                request = prefix.with_history(ledger["messages"])
                request["add_generation_prompt"] = False
                ledger["turn"] = list(await server.render_tokens(request))
            turn, eos = ledger["turn"], self.ctx["specials"]["eos"]
            # Append-only template, and the turn ends <|im_end|> (+ "\n" in the classic layout): the seam
            # from that <|im_end|> on is the template's own tokens.
            end = len(turn) - 1 if turn[-1:] == [eos] else len(turn) - 2 if turn[-2:-1] == [eos] else None
            if end is None or full[:len(turn)] != turn:
                self.record["prompt_ledger"]["template_changed"] += 1
                return full
            self.record["prompt_ledger"]["extended"] += 1
            return with_ids(server, full, ledger["served"] + list(full[end:]))
        text = prompt_text(prefix, messages)
        if ledger is not None:
            head = ledger.setdefault("text", prefix + render_turns(ledger["messages"]))
            if text.startswith(head) and head.endswith("<|im_end|>\n"):
                self.record["prompt_ledger"]["extended"] += 1
                return ledger["served"] + await server.tokenize("<|im_end|>\n" + text[len(head):])
            self.record["prompt_ledger"]["template_changed"] += 1
        return await server.tokenize(text)

    def extends(self, messages):
        """Whether messages only append to the ledger's history.

        Re-rendering a generated turn re-splits its text (non-canonical BPE, merged seams, parsed tool
        calls, promoted notes) and the prefix cache misses from there. Any edit of the keyed history
        (elision, dropped reasoning, trims, omitted images, revealed tests) re-renders instead."""
        ledger = self.ledger
        if ledger is None:
            return False
        key = ledger["messages"]
        if (ledger["prefix"] is self.ctx["prefix"] and len(messages) > len(key)
                and list(messages[:len(key)]) == key):
            return True
        self.record["prompt_ledger"]["history_changed"] += 1
        return False

    def remember(self, sent, assistant, served):
        """Key the next prompt on this history; served is what the server ran and generated for it."""
        self.ledger = dict(prefix=self.ctx["prefix"], messages=deepcopy(list(sent)) + [deepcopy(assistant)],
                           served=served) if served is not None else None

    async def drop_unrun_call(self, served, start):
        """served cut where the answer opens the calls its parsed assistant drops (truncated or malformed).

        Kept, the next prompt shows code that never ran right before a nudge saying no call was made. None
        (re-render) when the ids have no closed thinking followed by the call opening."""
        call = await self.ctx["server"].tokenize(self.dialect.call_open)
        think_end = self.ctx["specials"].get("think_end")
        answer = max((i + 1 for i in range(start, len(served)) if served[i] == think_end), default=None)
        if answer is None or not call:
            return None
        cut = next((i for i in range(answer, len(served) - len(call) + 1) if served[i:i + len(call)] == call), None)
        return None if cut is None else served[:cut]

    async def fit(self, max_out, limit):
        """Keep requested reasoning; elide old tool output, then reduce completion room if necessary.

        Only drop retained reasoning when even min-turn tokens would not fit. A full turn cap is not
        a reason to silently discard the last reasoning the user explicitly asked to retain.
        """
        if getattr(self.args, "tir_repl_context_policy", "legacy") == "blocks":
            return await self.fit_blocks(max_out, limit)
        keep = self.args.tir_repl_keep_reasoning
        sent = api_messages(self.messages, keep)
        recovery, self.note_recovery_index = self.note_recovery_index, None
        if recovery is not None and 0 <= recovery < len(sent) and self.messages[recovery].get("reasoning_content"):
            # One bounded recovery turn after a failed summary, not permanent retention. Existing
            # context fitting below may still drop this if even the minimum generation cannot fit.
            sent[recovery] = dict(sent[recovery], reasoning_content=self.messages[recovery]["reasoning_content"])
            self.record["note_recovery_reasoning_turns"] += 1
        ids = await self.encode(sent) if self.messages else self.ctx["prefix_ids"]
        if len(ids) + max_out <= limit or not self.messages:
            return sent, ids, 0
        tools = [i for i, m in enumerate(sent) if m["role"] == "tool" and m["content"] != ELIDED][:-4]
        elided = 0
        for i in tools:
            if len(ids) + max_out <= limit:
                break
            sent[i] = {**sent[i], "content": ELIDED}
            elided += 1
            ids = await self.encode(sent)
        need = turn_floor(self.args, max_out)
        for i, message in enumerate(sent):
            if len(ids) + need <= limit:
                break
            if message.get("reasoning_content"):
                sent[i] = {k: v for k, v in message.items() if k != "reasoning_content"}
                self.record["reasoning_context_drops"] += 1
                ids = await self.encode(sent)
        return sent, ids, elided

    async def fit_blocks(self, max_out, limit):
        """Evict whole assistant/tool round trips only; never reinsert evicted history next turn.

        The system/task prefix and sandbox journal are independent of this window. The latest
        round trip stays intact even if too large (the caller then reduces output or stops).
        """
        recovery, self.note_recovery_index = self.note_recovery_index, None

        def window(start):
            tail = api_messages(self.messages[start:], self.args.tir_repl_keep_reasoning)
            if recovery is not None and start <= recovery < len(self.messages):
                m = self.messages[recovery]
                if m.get("reasoning_content"):
                    tail[recovery - start] = dict(tail[recovery - start], reasoning_content=m["reasoning_content"])
            return ([{"role": "user", "content": HISTORY_TRIMMED}] if start else []) + tail

        start = self.history_start
        sent = window(start)
        ids = await self.encode(sent) if self.messages else self.ctx["prefix_ids"]
        before = len(ids)
        if before + max_out > limit:
            target = min(limit - max_out, int(limit * self.args.tir_repl_context_retain_fraction))
            # Each boundary starts an assistant turn; preceding tool responses/nudges leave with
            # their assistant. No orphan tool messages, and at least the newest turn survives.
            boundaries = [i for i in range(start + 1, len(self.messages))
                          if self.messages[i]["role"] == "assistant"]
            for boundary in boundaries:
                self.history_start = boundary
                sent = window(boundary)
                ids = await self.encode(sent)
                if len(ids) <= target:
                    break
            if self.history_start != start:
                self.record["context_trims"].append(dict(turn=self.record["turns"], start_before=start,
                    start_after=self.history_start, prompt_before=before, prompt_after=len(ids), target=target))
        if recovery is not None and self.history_start <= recovery < len(self.messages):
            self.record["note_recovery_reasoning_turns"] += bool(self.messages[recovery].get("reasoning_content"))
        return sent, ids, 0

    async def request(self, prompt, max_tokens, folder, *, thinking, seed):
        ctx, args = self.ctx, self.args
        payload = dict(ctx["payload"], prompt=prompt, max_tokens=max_tokens)
        if getattr(args, "tir_repl_cache_salt", None) is not None:
            payload["cache_salt"] = args.tir_repl_cache_salt
        if "seed" in payload:  # vLLM requests stay unseeded: a seed moves the batch off the FlashInfer sampler.
            payload["seed"] = seed
        if "cache_prompt" in payload:  # llama.cpp; vLLM's prefix cache needs no request field.
            # A ledger server's prompts extend the last turn's served ids, so its slot resumes there.
            payload["cache_prompt"] = bool(args.tir_repl_cache_prompt) or getattr(ctx["server"], "prompt_ledger", False)
        if isinstance(ctx["server"], VllmServer):
            # Our vLLM fork keeps the task prefix's cached states while this turn runs, so the
            # next trajectory, which starts after this one finishes, resumes there instead of
            # re-prefilling the prefix. Other vLLM builds ignore the extra arg.
            payload["vllm_xargs"] = dict(prefix_cache_hold_tokens=len(ctx["prefix_ids"]))
        timeout = min(args.request_timeout_seconds, max(0.001, self.request_deadline() - time.monotonic()))
        progress = ctx["progress"]
        result = await ctx["server"].complete(payload, folder, timeout, specials=ctx["specials"], progress=progress,
                                              thinking=thinking, **ctx["guard"]())
        shared = next((i for i, (a, b) in enumerate(zip(prompt, self.previous_prompt)) if a != b),
                      min(len(prompt), len(self.previous_prompt)))
        self.previous_prompt = list(prompt)
        usage = result.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if type(cached) is int and 0 <= cached <= len(prompt):
            self.record["cache_reported_requests"] += 1
            self.record["cached_prompt_tokens"] += cached
            self.record["cache_reported_prompt_tokens"] += len(prompt)
        else:
            cached = None  # Missing telemetry is unknown, not a zero cache hit rate.
        # The full prompt ids are rebuilt from traj_*.json; keep the request record small.
        dsv4_llama.write_json(folder / "request.json", {**payload, "prompt": None, "prompt_tokens": len(prompt),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "history_start": self.history_start, "context_policy": self.record["context_policy"],
            "request_timeout_seconds": timeout,
            "reported_cached_tokens": cached, "input_prefix_tokens_matching_previous_request": shared})
        tokens = dsv4_llama.generated(result) or 0
        # Same step: the live view moves this request's tokens from decoded to decoded_before.
        progress["decoded_before"] = progress.get("decoded_before", 0) + tokens
        progress["decoded"] = 0
        self.record["requests"] += 1
        rate = result.get("decode_tokens_per_s_estimate")
        if rate:
            self.record["decode_seconds"] += tokens / rate
        first = result.get("first_content_s")
        self.turn_wait += first if first is not None else result.get("elapsed_s") or 0.0
        self.last = result
        return result, tokens

    async def turn(self, sent_ids, max_out, seed, *, think_cap=None, tool_continuation_tokens=0):
        """One assistant turn: (reasoning, answer, finish, completion tokens, error, cancelled). A turn whose
        thinking reaches --tir-repl-think-tokens (or a loop stop while thinking) gets THINK_CLOSE appended and
        continues as its answer within the same max_out. Only the soft solver
        checkpoint can supply tool_continuation_tokens, spent once on finishing
        a truncated call with the exact assistant prefix (including images)."""
        args, number = self.args, self.record["turns"]
        cap = args.tir_repl_think_tokens if think_cap is None else think_cap
        first = min(max_out, cap) if cap else max_out
        folder = self.folder / f"turn_{number:03d}"
        tail_prompt, tail_folder = sent_ids, folder
        result, tokens = await self.request(sent_ids, first, folder, thinking=True, seed=seed)
        tail_result = result
        reasoning, answer, finish = result["reasoning"], result["content"], result["finish_reason"]
        info = dict(think_closed=result["think_closed"], capped=False, loop=finish == "loop_detected")
        if result["error"] or result["cancelled"]:
            return reasoning, answer, finish, tokens, result["error"], result["cancelled"], info
        stopped = finish in ("length", "loop_detected")
        if stopped and not result["think_closed"] or finish == "length" and first < max_out:
            close = [] if result["think_closed"] else await self.ctx["server"].tokenize(self.dialect.think_close)
            generated = [t for event_tokens, _ in dsv4_llama.read_partial(folder)["events"] for t in event_tokens]
            prompt = await append_prompt(self.ctx["server"], sent_ids, generated + close)
            room = min(max_out - tokens, self.ctx["limit"] - len(prompt))
            if room >= 256 and (close or finish == "length"):
                info.update(capped=bool(close))
                self.record["think_capped"] += bool(close)
                tail_prompt, tail_folder = prompt, self.folder / f"turn_{number:03d}b"
                more, extra = await self.request(prompt, room, tail_folder,
                                                 thinking=False, seed=seed + 1)
                tail_result = more
                if close:
                    reasoning += self.dialect.think_close.split("</think>", 1)[0].rstrip("\n")
                answer += more["content"]
                finish, tokens = more["finish_reason"], tokens + extra
                info.update(think_closed=True, loop=info["loop"] or finish == "loop_detected")
                if more["error"] or more["cancelled"]:
                    return reasoning, answer, finish, tokens, more["error"], more["cancelled"], info
        if (tool_continuation_tokens > 0 and finish == "length" and info["think_closed"]
                and self.dialect.parse(answer)[2]
                and answer.rfind(self.dialect.call_open) > answer.rfind(self.dialect.call_close)):
            # Recheck time/budgets after generation: the first request may have
            # consumed the wall-clock reserve. Never borrow finalization tokens.
            now = time.monotonic()
            pending = dict(self.record, completion_tokens=self.record["completion_tokens"] + tokens)
            if self.started_at is not None:
                pending["seconds"] = now - self.started_at
            remaining = completion_plan(args, pending, wall_left=self.ctx["deadline"] - now)
            allowance = min(tool_continuation_tokens, TOOL_CONTINUATION_TOKENS, remaining["max_out"])
            if (allowance >= 256 and not remaining["wall_finalization"]
                    and self.request_deadline() - now >= args.min_request_seconds):
                generated = [t for event_tokens, _ in dsv4_llama.read_partial(tail_folder)["events"]
                             for t in event_tokens]
                if not generated:
                    # Missing raw tokens must not silently restart from an older prefix.
                    return reasoning, answer, finish, tokens, None, False, info
                # Continue the raw assistant prefix via the image-aware helper;
                # never reconstruct the tool call or invent closing tags.
                prompt = await append_prompt(self.ctx["server"], tail_prompt, generated)
                room = min(allowance, self.ctx["limit"] - len(prompt))
                if room >= 256:
                    tail_prompt, tail_folder = prompt, self.folder / f"turn_{number:03d}c"
                    more, extra = await self.request(prompt, room, tail_folder, thinking=False, seed=seed + 2)
                    tail_result = more
                    answer += more["content"]
                    finish, tokens = more["finish_reason"], tokens + extra
                    info["loop"] = info["loop"] or finish == "loop_detected"
                    _, calls, malformed = self.dialect.parse(answer)
                    info["tool_continuation"] = dict(max_tokens=room, tokens=extra,
                        recovered=bool(calls) and not malformed and not more["error"] and not more["cancelled"])
                    if more["error"] or more["cancelled"]:
                        return reasoning, answer, finish, tokens, more["error"], more["cancelled"], info
        if self.ledger_enabled and info["think_closed"]:
            generated = [t for event_tokens, _ in dsv4_llama.read_partial(tail_folder)["events"] for t in event_tokens]
            if tail_result["stop_type"] == "eos":
                generated = generated[:-1]  # Its KV was never computed; the seam re-adds <|im_end|>.
            info["served"] = list(tail_prompt) + generated
        return reasoning, answer, finish, tokens, None, False, info

    async def require_progress_note(self, assistant, calls, sent, *, seed, remaining_turn_tokens):
        """Gate execution, repairing only the note once while the current thinking is still available.

        Never rerun/rewrite the pending code. The repair has no tools and no new thinking budget;
        its tokens count against both the turn and trajectory budgets. Raw responses remain on disk.
        Format failures warn and retain the current reasoning for one recovery turn. Transport,
        cancellation and deadline failures still stop; no code from the repair is ever executed.
        """
        if getattr(self.args, "tir_repl_progress_notes", "required") == "optional":
            notes = list(dict.fromkeys(a["progress_note"] for _, a in calls
                                      if isinstance(a.get("progress_note"), str) and a["progress_note"].strip()))
            if notes:
                assistant["content"] = "\n\n".join(notes + ([assistant["content"]] if assistant["content"] else []))
            self.record["progress_notes"]["optional_present" if notes or assistant["content"] else "optional_absent"] += 1
            return None, False
        note, source = progress_note_for_calls(assistant["content"], calls)
        log = self.record["turn_log"][-1]
        if note is None:
            repair_started = time.monotonic()
            if self.ctx["deadline"] - time.monotonic() < self.args.min_request_seconds:
                return "deadline", False
            left = min(remaining_turn_tokens, self.args.tir_repl_budget - self.record["completion_tokens"])
            if left < 128:
                return self.note_warning(assistant, calls, "repair-budget"), False
            # Do not use api_messages/fit here: that would remove precisely the reasoning we need
            # to summarize. Pending tool responses make a valid native-template round trip.
            messages = list(sent) + [assistant] + [
                {"role": "tool", "tool_call_id": c["id"],
                 "content": "Pending: not executed; a progress note is required first."}
                for c in assistant["tool_calls"]] + [{"role": "user", "content": PROGRESS_REPAIR}]
            ids = await self.encode(messages)
            # Both classic (already-open think block) and compact Qwen templates are supported.
            opened = self.ctx["specials"]["think"] in ids[-2:]
            ids = await append_prompt(self.ctx["server"], ids,
                await self.ctx["server"].tokenize(("" if opened else "<think>") + self.dialect.empty_think))
            room = min(PROGRESS_REPAIR_TOKENS, left, self.ctx["limit"] - len(ids))
            if room < 128:
                return self.note_warning(assistant, calls, "repair-context"), False
            if self.ctx["deadline"] - time.monotonic() < self.args.min_request_seconds:
                return "deadline", False
            folder = self.folder / f"turn_{self.record['turns'] - 1:03d}_note"
            dsv4_llama.write_json(folder.with_suffix(".messages.json"), messages)
            result, tokens = await self.request(ids, room, folder, thinking=False, seed=seed)
            self.record["progress_note_repairs"] += 1
            self.record["progress_note_repair_tokens"] += tokens
            self.record["completion_tokens"] += tokens
            self.record["prompt_tokens"] += len(ids)
            self.record["prompt_tokens_max"] = max(self.record["prompt_tokens_max"], len(ids))
            self.record["context_max"] = max(self.record["context_max"], len(ids) + tokens)
            repair_content, repair_calls, repair_malformed = self.dialect.parse(result["content"])
            log["note_repair"] = dict(prompt=len(ids), completion=tokens, finish=result["finish_reason"],
                                      seconds=round(time.monotonic() - repair_started, 3),
                                      ignored_tool_calls=len(repair_calls),
                                      ignored_incomplete_tool_call=repair_malformed)
            if result["error"] or result["cancelled"]:
                self.record["error"] = result["error"]
                return "cancelled" if result["cancelled"] else "error", bool(result["cancelled"])
            # A tool-opening delimiter establishes the end of the visible note, even if the
            # unwanted tool body is truncated. A bare length-limited note has no such boundary.
            truncated = result["finish_reason"] in ("length", "loop_detected")
            note = (validate_progress_note(repair_content) if truncated and self.dialect.call_open in result["content"]
                    else None if truncated else progress_note_for_calls(repair_content, repair_calls)[0])
            log["note_repair"]["accepted_before_truncated_tool"] = bool(note and truncated)
            if note is None:
                return self.note_warning(assistant, calls, "repair-invalid-or-truncated"), False
            source = "repair"
        log["progress_note"] = source
        self.record["progress_notes"][source] += 1
        # Promote tool-parameter/repaired notes to visible assistant text BEFORE the tool, so
        # history pruning and the offline viewer cannot hide them inside tool argument widgets.
        if source != "visible":
            log["original_content"] = assistant["content"]
            assistant["content"] = note + ("\n\n" + assistant["content"] if assistant["content"] else "")
        return None, False

    def note_warning(self, assistant, calls, reason):
        """Keep model-authored memory verbatim; a formatting error cannot kill a valid pending tool."""
        log = self.record["turn_log"][-1]
        log.update(progress_note="unstructured", progress_note_warning=reason,
                   original_content=assistant["content"])
        self.record["progress_note_warnings"] += 1
        self.record["progress_notes"]["unstructured"] += 1
        raw = [a.get("progress_note") for _, a in calls]
        notes = list(dict.fromkeys(x[:1800] for x in raw if isinstance(x, str) and x.strip()))
        if notes:
            assistant["content"] = "\n\n".join(notes + ([assistant["content"]] if assistant["content"] else []))
        self.note_recovery_index = next((i for i, m in enumerate(self.messages) if m is assistant), None)
        return None

    def show_predictions(self, first):
        """Draw the newest preferred candidate's test predictions beside their inputs, once per prediction set."""
        record = self.record
        index = next((i for i in range(len(record["submissions"]) - 1, first - 1, -1)
                      if record["submissions"][i].get("accepted") and not record["submissions"][i]["preserve_only"]),
                     None)
        shown = record["prediction_images"]
        if (index is None or len(shown) >= PREDICTION_IMAGE_LIMIT
                or shown and record["submissions"][shown[-1]]["test"] == record["submissions"][index]["test"]):
            return
        pairs = [(case["input"], grid) for case, grid in zip(self.task["test"], record["submissions"][index]["test"])]
        try:
            artifacts = self.ctx["visualizer"].predictions(pairs, turn=record["turns"])
        except ValueError as exc:
            self.messages.append({"role": "user", "content": f"[prediction images not rendered: {exc}]"})
            return
        shown.append(index)
        self.messages.append({"role": "user", "content": PREDICTION_IMAGES, "_vision": artifacts})

    def save(self, write_json):
        snapshot = self.summary()
        write_json(self.folder.with_suffix(".json"), dict(snapshot, messages=self.messages,
                                                          journal=self.session.journal, history_start=self.history_start,
                    **({"journal_test_counts": self.session.journal_test_counts} if self.disclosure else {})))
        # Persist the evidence before publishing answers. A long review (or the
        # next tool call) must not keep an already accepted answer out of the
        # pipeline's deadline submission. This callback does no model/tool work.
        if self.ctx.get("on_checkpoint") is not None:
            self.ctx["on_checkpoint"](snapshot)

    def summary(self):
        if self.disclosure:
            self.record["test_disclosure"] = self.disclosure.state()
        return dict(disclosure.original_order(self.record), tool_calls=dict(self.record["tool_calls"]), session=dict(self.session.stats),
                    last_fatal=self.session.last_fatal)

    async def reveal_tests(self, reason, *, all_remaining=False):
        """Disclose inputs, never answers; replay saved solvers with their original cell visibility."""
        state = self.disclosure
        if not state or not state.pending:
            return False
        prefix = self.ctx["prefix"]
        if not isinstance(prefix, CheckpointPrefix):
            raise ValueError("Staged test disclosure needs the native checkpoint template")
        previous_count = len(state.indices)
        indices = state.pending if all_remaining else state.pending[:1]
        event = dict(turn=self.record["turns"], reason=reason, original_indices=indices,
                     completion_tokens=self.record["completion_tokens"], candidates=[], images=[])
        parts = []
        for position, original in enumerate(indices, previous_count):
            grid = state.full["test"][original]["input"]
            label = f"New test_inputs[{position}] (original test {original}); input only, answer UNKNOWN."
            parts.append({"type": "text", "text": label + "\n" + fmt(grid, self.session.grid_sep)})
            if self.ctx.get("visualizer"):
                visualizer = self.ctx["visualizer"]
                picture = visualizer.renderer.test_image(grid, original, label)
                event["images"].append(picture.save(visualizer.root))
                parts.extend(picture.content_parts())
        # Persistent task context, not a prunable Python view. Old request snapshots
        # stay immutable; new inputs never enter earlier journal cells on replay.
        request = deepcopy(prefix.request)
        request["messages"].append({"role": "user", "content": parts})
        state.advance(all_remaining=all_remaining)
        state.events.append(event)
        self.ctx["prefix"] = CheckpointPrefix(request, prefix.text + "\n" + "\n".join(
            p["text"] for p in parts if p["type"] == "text"))
        # Finish structural bookkeeping before any await/cancellation can publish
        # a partial snapshot. Evaluate the preferred candidate first.
        entries = self.record["submissions"]
        prior = [list(e["test"]) for e in entries]
        for entry in entries:
            entry["test"].extend([None] * len(indices))
            entry["test_indices"] = list(state.indices)
        feedback = []
        for number in sorted(range(len(entries)), key=lambda i: i != self.record["accepted"]):
            entry, old = entries[number], prior[number]
            if not entry.get("accepted"):
                continue
            check = dict(submission=number, function=entry["function"], status="not-evaluated")
            event["candidates"].append(check)
            length = entry["journal_length"]
            if self.request_deadline() > time.monotonic():
                self.session.stats["disclosure_evaluations"] += 1
                result = await self.session._run(("eval", entry["function"]),
                                                 journal=self.session.journal[:length]) or {}
                train = result.get("train", [])
                tests = result.get("test", [])
                train_ok = len(train) == len(self.task["train"]) and all(
                    not case.get("error") and case.get("grid") == pair["output"]
                    for pair, case in zip(self.task["train"], train))
                stable = len(tests) == len(state.indices) and all(
                    grid is None or (not case.get("error") and case.get("grid") == grid)
                    for grid, case in zip(old, tests))
                if not result.get("error") and train_ok and stable:
                    check["status"] = "evaluated"
                    check["train_passed"] = len(train)
                    check["errors"] = {str(state.indices[i]): c.get("error") for i, c in enumerate(tests)
                                       if c.get("error")}
                    for i, case in enumerate(tests):
                        if entry["test"][i] is None and not case.get("error"):
                            entry["test"][i] = case.get("grid")
                else:
                    check.update(status="replay-failed", error=result.get("error") or
                                 "Training checks or previously saved predictions changed on replay")
            if number == self.record["accepted"]:
                feedback.append(f"Saved solver {entry['function']}: {check['status']} on the new input(s).")
                for position in range(previous_count, len(state.indices)):
                    grid = entry["test"][position]
                    feedback.append(f"test_inputs[{position}] prediction (correctness UNKNOWN):\n"
                                    f"{fmt(grid, self.session.grid_sep)}"
                                    if grid is not None else
                                    f"test_inputs[{position}]: no valid prediction saved; inspect/repair the solver.")
        self.record["test_disclosure"] = state.state()
        self.record["rule_audit_completed"] = self.record["rule_audit_nudged"] = False
        self.messages.append({"role": "user", "content": (
            f"[test input disclosure: {reason}] Newly visible Python indices "
            f"{list(range(previous_count, len(state.indices)))} are appended above; earlier indices are unchanged. "
            "All training pairs remain visible. No test answer has been checked.\n" + "\n".join(feedback) +
            "\nBriefly restate the current rule and its unresolved assumptions. Check which assumptions "
            "hold on EACH new input (counts, boundaries, geometry, ordering and operation interactions). "
            "Use measurements to investigate differences, not to patch test cells. If revising, rerun ALL "
            "training pairs and review EVERY revealed input, then submit the revised general solver. "
            "Otherwise explain why the saved rule applies and submit the unchanged solver to continue. "
            "Previously saved predictions remain available. The SAME task and first-acceptance review "
            "budgets continue; disclosure grants no extra tokens, time or turns.")})
        return True

    async def run(self, write_json):
        args, ctx, record = self.args, self.ctx, self.record
        self.folder.mkdir()
        start = self.started_at = time.monotonic()
        seed_base = ctx["payload"].get("seed", 0)
        cancelled = False
        try:
            while True:
                record["seconds"] = round(time.monotonic() - start, 1)
                self.save(write_json)
                if (solver_gate_active(args, record)
                        and record["solver_checkpoint_turns"] >= args.tir_repl_solver_max_turns):
                    if getattr(args, "tir_repl_solver_checkpoint_action", "stop") == "resume":
                        record["solver_checkpoint_released_at_tokens"] = record["completion_tokens"]
                        self.messages.append({"role": "user", "content": (
                            "[solver checkpoint released] No executable check completed in the focused "
                            "allowance. Resume exploration with your existing code and observations; "
                            "the task budget is unchanged. Repair one concrete failure or test one "
                            "specific uncertainty, then implement/check the current hypothesis.")})
                    else:
                        record["stop"] = "solver_checkpoint_exhausted"
                        self.messages.append({"role": "user", "content": (
                            "[harness] No executable solver check completed within the bounded checkpoint. "
                            "Stopping this trajectory and preserving its code and existing candidates.")})
                        break
                wall_left = ctx["deadline"] - time.monotonic()
                plan = completion_plan(args, record, wall_left=wall_left)
                left = plan["left"]
                if record["turns"] >= args.tir_repl_max_turns or left < 1000:
                    record["stop"] = ("review_budget" if plan["review_active"] else "budget") if left < 1000 else "max_turns"
                    break
                if wall_left < args.min_request_seconds:
                    record["stop"] = "deadline"
                    break
                if (plan["review_seconds_left"] is not None
                        and plan["review_seconds_left"] <= max(0, args.min_request_seconds)):
                    record["stop"] = "review_time"
                    break
                if self.disclosure and self.disclosure.pending and (
                        (not plan["review_active"] and plan["phase"] in ("finalization", "code_delivery"))
                        or plan["wall_finalization"]
                        or plan["review_active"] and (left <= args.tir_repl_finalize_turn_tokens
                            or plan["review_seconds_left"] is not None and plan["review_seconds_left"] <= 120)):
                    await self.reveal_tests("finalization", all_remaining=True)
                if plan["phase"] == "provisional_solver" and record["solver_checkpoint_started_at_tokens"] is None:
                    record["solver_checkpoint_started_at_tokens"] = record["completion_tokens"]
                    note = (solver_checkpoint_notice(args)
                            if getattr(args, "tir_repl_solver_max_turns", 0) else SOLVER_CHECKPOINT)
                    self.messages.append({"role": "user", "content": note})
                if plan["phase"] in ("finalization", "code_delivery") and record["finalization_started_at_tokens"] is None:
                    record["finalization_started_at_tokens"] = record["completion_tokens"]
                    record["finalization_started_at_seconds"] = record["seconds"]
                    record["finalization_reason"] = "wall_time" if plan["wall_finalization"] else "tokens"
                    note = FINALIZE.format(left=left)
                    if record.get("fresh_challenger"):
                        note = (f"[finish challenger] At most {left} generated tokens remain. "
                                "Use short reasoning and Python to check a meaningful alternative. "
                                "Submit only a function passing ALL demonstrations with distinct test predictions, "
                                "or finish in text if none is defensible. The original answer is already safe. "
                                "Do not force a difference or resubmit the original candidate.")
                    elif plan["review_active"] and record["targeted_review_enabled"]:
                        note = f"[bounded rule audit] At most {left} completion tokens remain. " + TARGETED_REVIEW
                    elif plan["review_active"] and record["candidate_pool_enabled"]:
                        note = (f"[bounded candidate review] At most {left} additional completion tokens remain. "
                                "Keep the saved preferred candidate. Test the most consequential plausible "
                                "alternative using short reasoning and executable checks. If it passes ALL "
                                "training pairs and predicts differently, submit its named function with "
                                "preserve_only=true for attempt_2; otherwise finish. Do not fabricate a "
                                "difference, patch test cells, or treat training fit as proof of the test rule.")
                    if pending_input_review(record):
                        note += "\n" + pending_input_review(record)
                    if plan["wall_finalization"]:
                        note += (f"\n[wall-clock finalization] About {max(0, int(wall_left))} seconds "
                                 "remain before the task deadline, not necessarily enough to use the token "
                                 "allowance. Prioritize executable check/submit now; keep any saved answer. "
                                 "Do not repeat a completed audit or start broad new exploration.")
                    if plan["review_active"]:
                        note += "\nReview is bounded from the FIRST accepted candidate; resubmitting does not restart it."
                    self.messages.append({"role": "user", "content": note})
                if plan["phase"] == "code_delivery" and record["code_delivery_started_at_tokens"] is None:
                    record["code_delivery_started_at_tokens"] = record["completion_tokens"]
                    record["code_delivery_started_at_seconds"] = record["seconds"]
                    self.messages.append({"role": "user", "content": CODE_DELIVERY.format(left=left)})
                max_out = plan["max_out"]
                fit_start = time.monotonic()
                sent, ids, elided = await self.fit(max_out, ctx["limit"])
                gen_start, self.turn_wait = time.monotonic(), 0.0
                room = ctx["limit"] - len(ids)
                if room < turn_floor(args, max_out):
                    record["stop"] = "context"
                    break
                if room < max_out:
                    record["context_squeezed"] += 1
                max_out = min(max_out, room)
                record["elided_max"] = max(record["elided_max"], elided)
                if isinstance(ctx["prefix"], CheckpointPrefix):
                    # Save the exact pruned/elided history handed to the checkpoint template.
                    write_json(self.folder / f"turn_{record['turns']:03d}.messages.json",
                               ctx["prefix"].with_history(sent)["messages"])
                seed = ((seed_base * 1009 + self.number) * 1013 + record["turns"] * 2) % 2**31
                override = (dict(think_cap=min(plan["think_cap"], max(1, max_out - answer_room(args, max_out))))
                            if getattr(args, "tir_repl_finalize_reserve", 0)
                            or getattr(args, "tir_repl_finalize_reserve_seconds", 0) or plan["review_active"]
                            or getattr(args, "tir_repl_solver_by_tokens", 0) else {})
                if plan["tool_continuation_tokens"] and max_out == plan["max_out"]:
                    override["tool_continuation_tokens"] = plan["tool_continuation_tokens"]
                reasoning, answer, finish, tokens, error, cancelled, info = await self.turn(ids, max_out, seed, **override)
                continuation = info.get("tool_continuation")
                turn_limit = max_out + (continuation["max_tokens"] if continuation else 0)
                if solver_gate_active(args, record):
                    record["solver_checkpoint_turns"] += 1
                gen_s = time.monotonic() - gen_start
                record["turns"] += 1
                record["completion_tokens"] += tokens
                record["prompt_tokens"] += len(ids)
                record["prompt_tokens_max"] = max(record["prompt_tokens_max"], len(ids))
                record["context_max"] = max(record["context_max"], len(ids) + tokens)
                record["loop_stops"] += info["loop"]
                if error or cancelled:
                    record["stop"], record["error"] = ("cancelled" if cancelled else "error"), error
                    if not cancelled and error and error.startswith("TimeoutError:") and time.monotonic() >= self.request_deadline():
                        record["request_error_at_stop"] = error  # Preserve diagnostic, not an engine fault.
                        record["error"] = None
                        record["stop"] = "review_time" if self.request_deadline() < ctx["deadline"] else "deadline"
                    break
                content, calls, malformed = self.dialect.parse(answer) if info["think_closed"] else ("", [], False)
                if malformed:
                    # A partial multi-call answer is not atomic. Do not execute
                    # its parsed prefix and risk repeating side effects on retry.
                    calls = []
                assistant = {"role": "assistant", "content": content, "reasoning_content": reasoning}
                if calls:
                    assistant["tool_calls"] = [{"id": f"call_{record['turns']}_{i}", "type": "function",
                                                "function": {"name": name, "arguments": arguments}}
                                               for i, (name, arguments) in enumerate(calls)]
                self.messages.append(assistant)
                if self.ledger_enabled:
                    served = info.get("served")
                    if served is not None and not calls and self.dialect.call_open in answer:
                        served = await self.drop_unrun_call(served, len(ids))
                    self.remember(sent, assistant, served)
                record["turn_log"].append(dict(prompt=len(ids), completion=tokens, finish=finish,
                                               phase=plan["phase"], max_output_tokens=turn_limit,
                                               wall_seconds_left=round(wall_left, 1),
                                               review_seconds_left=plan["review_seconds_left"],
                                               thinking_cap=override.get("think_cap", args.tir_repl_think_tokens),
                                               think_closed=info["think_closed"], capped=info["capped"],
                                               calls=[name for name, _ in calls], tools_executed=[], malformed=malformed,
                                               fit_s=round(gen_start - fit_start, 2), gen_s=round(gen_s, 1),
                                               wait_s=round(min(self.turn_wait, gen_s), 1)))
                if continuation:
                    record["turn_log"][-1].update(tool_continuation=continuation,
                                                 initial_max_output_tokens=max_out)
                print(f"TURN {self.folder.parent.name}/{self.folder.name} {record['turns']}: prompt={len(ids)} "
                      f"gen={tokens} {gen_s:.0f}s total={record['completion_tokens']} finish={finish}"
                      f"{' capped' if info['capped'] else ''} calls={','.join(n for n, _ in calls) or '-'}"
                      f"{' malformed' if malformed else ''}"
                      + (f" tool_continued={continuation['tokens']}"
                         f" tool_recovered={continuation['recovered']}" if continuation else ""), flush=True)
                if not calls:
                    if plan["phase"] == "code_delivery" and record["accepted"] is None:
                        self.messages.append({"role": "user", "content": CODE_DELIVERY.format(
                            left=max(0, args.tir_repl_budget-record["completion_tokens"]))})
                        record["nudges"] += 1
                        continue
                    if solver_gate_active(args, record):
                        remaining = args.tir_repl_solver_max_turns - record["solver_checkpoint_turns"]
                        self.messages.append({"role": "user", "content": (
                            f"[solver checkpoint] {remaining} model turns remain. Call propose_solver with "
                            "code defining your current solve(grid); the harness checks it automatically. "
                            "No special note format is required when progress notes are optional.")})
                        record["nudges"] += 1
                        continue
                    if finish in ("length", "loop_detected") or not info["think_closed"]:
                        self.messages.append({"role": "user", "content": NUDGE_LENGTH})
                        record["nudges"] += 1
                        continue
                    if malformed:
                        nudge = self.dialect.nudge_format
                        if getattr(args, "tir_repl_progress_notes", "required") == "optional":
                            nudge = self.dialect.nudge_optional
                        self.messages.append({"role": "user", "content": nudge})
                        record["nudges"] += 1
                        continue
                    if record.get("fresh_challenger"):
                        record["stop"] = "no-alternative"
                        break
                    if record["accepted"] is not None:
                        if await self.reveal_tests("candidate-review-complete"):
                            continue
                        if pending_input_review(record) and record["input_audit_nudges"] < 2:
                            record["input_audit_nudges"] += 1
                            self.messages.append({"role": "user", "content": pending_input_review(record)})
                            continue
                        if (record["targeted_review_enabled"] and not record["rule_audit_completed"]
                                and not record["rule_audit_nudged"]):
                            record["rule_audit_nudged"] = True
                            self.messages.append({"role": "user", "content": TARGETED_REVIEW})
                            continue
                        record["stop"] = "done"
                        break
                    if record["nudges"] >= args.tir_repl_max_nudges:
                        record["stop"] = "gave_up"
                        break
                    record["nudges"] += 1
                    self.messages.append({"role": "user", "content": NUDGE_NO_SUBMIT})
                    continue
                stop, cancelled = await self.require_progress_note(
                    assistant, calls, sent, seed=seed + 3 if continuation else seed + 2,
                    remaining_turn_tokens=turn_limit - tokens)
                if stop:
                    record["stop"] = stop
                    # Resolve pending calls honestly in the saved trace; none reached the sandbox.
                    self.messages.extend({"role": "tool", "tool_call_id": c["id"],
                                          "content": f"Not executed: {stop}."}
                                         for c in assistant["tool_calls"])
                    break
                if self.ledger is not None:
                    # Promoted notes already sit in the generated tool call; a repaired one is new context.
                    self.remember(sent, assistant, None if record["turn_log"][-1].get("progress_note") == "repair"
                                  else self.ledger["served"])
                tools_start = time.monotonic()
                submissions_before_turn = len(record["submissions"])
                outcomes = []
                visual_feedback = []
                visual_errors = []
                for call in assistant["tool_calls"]:
                    record["seconds"] = round(time.monotonic() - start, 3)
                    if self.request_deadline() <= time.monotonic():
                        self.messages.append({"role": "tool", "tool_call_id": call["id"],
                            "content": "Not executed: task/review wall budget exhausted; saved candidates retained."})
                        continue
                    if record.get("fresh_challenger") and record["accepted"] is not None:
                        self.messages.append({"role": "tool", "tool_call_id": call["id"],
                            "content": "Not executed: first distinct alternative already saved; challenger finished."})
                        continue
                    name, arguments = call["function"]["name"], call["function"]["arguments"]
                    candidates_before = (record["accepted"], len(record["submissions"]))
                    record["tool_calls"][name] += 1
                    gate = solver_gate_active(args, record)
                    final_proposal = name == "finalize_solver" and bool(record["code_reserve"])
                    proposal = final_proposal or name == "propose_solver" and bool(
                        getattr(args, "tir_repl_solver_max_turns", 0) or record["auto_check_solver_enabled"])
                    delivering = plan["phase"] == "code_delivery" and record["accepted"] is None
                    problem = None
                    if name == "python" or proposal:
                        code, problem = extract_code(arguments, "code")
                        blocked = gate and not problem and name == "python" and not solver_code_kind(code)
                        if blocked:
                            problem = ("Not executed: broad exploration is paused at the solver checkpoint. "
                                       "Call propose_solver(code) with def solve(grid) for your current hypothesis; "
                                       "a failed training check is enough to resume revision.")
                            record["solver_checkpoint_blocked_calls"].append(dict(turn=record["turns"], tool=name))
                        if delivering and not problem and name == "python" and not solver_code_kind(code):
                            problem = ("Not executed: protected code-delivery window. Use finalize_solver(code) "
                                       "with your best def solve(grid), or repair a helper and include check(solve) "
                                       "in the same Python cell. Submit an existing checked function if ready.")
                            record["code_delivery_blocked_calls"].append(dict(turn=record["turns"], tool=name))
                        if problem:
                            text = problem
                        else:
                            record["turn_log"][-1]["tools_executed"].append(name)
                            auto_check = bool(record["auto_check_solver_enabled"] and solver_code_kind(code))
                            capture = (record.get("input_audit_enabled") and record["candidate_pool_enabled"]
                                       and record["accepted"] is not None and not proposal and not gate
                                       and not auto_check)
                            if capture:
                                preferred = record["submissions"][record["accepted"]]["function"]
                                text = await self.session.cell(code, capture_alternatives_from=preferred)
                                text += input_audit.capture_checked_alternatives(
                                    self.task, record, self.session.last_checked_alternatives, len(self.session.journal))
                            else:
                                text = (await self.session.propose_solver(code, require_definition=proposal)
                                        if proposal or gate or auto_check else await self.session.cell(code))
                        failed = bool(problem or self.session.last_tool_failed)
                        check = self.session.last_solver_check if not problem else {}
                        check_source = name
                        # Successful helper/global edits can change solve without redefining it.
                        # Rate-limit by generated tokens; never check stale state after a failed cell,
                        # repeat an unchanged journal, or spend accepted-candidate review on rechecks.
                        interval = record["solver_recheck_tokens"]
                        previous = record["solver_checks"][-1] if record["solver_checks"] else None
                        periodic = record["periodic_solver_checks"][-1] if record["periodic_solver_checks"] else {}
                        if (interval and previous and not failed and not check.get("checked")
                                and record["accepted"] is None
                                and len(self.session.journal) > periodic.get("journal_length", -1)
                                and record["completion_tokens"] - max(previous["completion_tokens"],
                                    periodic.get("completion_tokens", 0)) >= interval):
                            text += "\n" + await self.session.check_solver(periodic=True)
                            check, failed = self.session.last_solver_check, self.session.last_tool_failed
                            check_source = "periodic"
                            record["periodic_solver_checks"].append(dict(turn=record["turns"],
                                completion_tokens=record["completion_tokens"],
                                journal_length=len(self.session.journal), failed=failed, report=deepcopy(check)))
                        if check.get("memorization_guard", {}).get("blocked"):
                            record["memorization_rejections"].append(dict(
                                turn=record["turns"], source=name,
                                completion_tokens=record["completion_tokens"],
                                report=check["memorization_guard"]))
                        if check.get("checked"):
                            record["solver_checks"].append(dict(turn=record["turns"], source=check_source,
                                completion_tokens=record["completion_tokens"], **check))
                            if record["first_solver_check_tokens"] is None:
                                record["first_solver_check_tokens"] = record["completion_tokens"]
                        if (final_proposal and not failed and check.get("checked")
                                and check.get("train_passed") == len(self.task["train"])):
                            # Explicit write-and-submit tool: preserve normal guards, holdout and
                            # test-input validation, and publish before the next model turn.
                            text += "\n" + await submit(self.session, self.task, "solve", record,
                                args.tir_repl_review, holdout=self.holdout)
                            failed = self.session.last_tool_failed
                    elif name == "submit":
                        before = len(record["submissions"])
                        fn, problem = extract_code(arguments, "function_name")
                        preserve = str(arguments.get("preserve_only", "false")).lower()
                        if preserve not in ("true", "false"):
                            problem = "preserve_only must be true or false."
                        elif preserve == "true" and not record["candidate_pool_enabled"]:
                            problem = "preserve_only is unavailable in this run."
                        text = problem or await submit(self.session, self.task, fn, record, args.tir_repl_review,
                                                       holdout=self.holdout, preserve_only=preserve == "true")
                        if not problem:
                            record["turn_log"][-1]["tools_executed"].append(name)
                        # A rejected submission still performed a real provisional training check.
                        # Count it for the soft checkpoint too, not only the hard tool gate.
                        # Invalid/undefined functions return before appending a submission.
                        if len(record["submissions"]) > before:
                            entry = record["submissions"][-1]
                            record["solver_checks"].append(dict(turn=record["turns"], source="submit",
                                completion_tokens=record["completion_tokens"], defined=True, checked=True,
                                train_passed=entry["train_passed"], train_total=len(self.task["train"])))
                            if record["first_solver_check_tokens"] is None:
                                record["first_solver_check_tokens"] = record["completion_tokens"]
                        failed = bool(problem or self.session.last_tool_failed)
                    elif name == "audit_inputs" and record.get("input_audit_enabled"):
                        record["turn_log"][-1]["tools_executed"].append(name)
                        text = await input_audit.audit_inputs(self.session, self.task, arguments, record)
                        failed = self.session.last_tool_failed
                    elif name == "audit_interactions" and record.get("interaction_audit_enabled"):
                        record["turn_log"][-1]["tools_executed"].append(name)
                        text = await interaction_audit.audit_interactions(self.session, self.task, arguments, record)
                        failed = self.session.last_tool_failed
                    elif name == "review_interactions" and record.get("interaction_audit_enabled"):
                        record["turn_log"][-1]["tools_executed"].append(name)
                        text = interaction_audit.review_interactions(self.session, self.task, arguments, record)
                        failed = self.session.last_tool_failed
                    elif name == "trace_solver" and record.get("execution_audit_enabled"):
                        record["turn_log"][-1]["tools_executed"].append(name)
                        text = await execution_audit.trace_solver(self.session, self.task, arguments, record)
                        failed = self.session.last_tool_failed
                    elif name == "compare_rules" and (record["targeted_review_enabled"] or record.get("input_audit_enabled")):
                        record["turn_log"][-1]["tools_executed"].append(name)
                        text = await compare_rules(self.session, self.task, arguments, record, holdout=self.holdout)
                        failed = self.session.last_tool_failed
                    else:
                        available = ", ".join(t["function"]["name"] for t in tool_schemas(args))
                        text = f"Unknown tool {name!r}. The tools are {available}."
                        failed = True
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                    if candidates_before != (record["accepted"], len(record["submissions"])):
                        # Checkpoint between calls in the same assistant turn too:
                        # a following audit/Python call may time out or be cancelled.
                        self.save(write_json)
                    if (name == "python" or proposal) and not problem and self.session.last_visuals and self.ctx.get("visualizer"):
                        try:
                            visual_feedback.extend(self.ctx["visualizer"].save(self.session.last_visuals,
                                turn=record["turns"], call_id=call["id"]))
                        except ValueError as exc:
                            visual_errors.append(f"[view not rendered: {exc}; request fewer/smaller crops]")
                    outcomes.append((name, arguments, text, failed))
                if visual_feedback:
                    self.messages.append({"role": "user",
                        "content": "[Python visual feedback] Requested grids/crops, not new evidence or ground truth. "
                                   "Order is left to right; use numerical measurements for exact cells.",
                        "_vision": visual_feedback})
                if visual_errors:
                    self.messages.append({"role": "user", "content": "\n".join(visual_errors)})
                if getattr(args, "tir_repl_prediction_images", False) and self.ctx.get("visualizer"):
                    self.show_predictions(submissions_before_turn)
                # Do not change test_inputs midway through a batch of model-authored
                # tool calls. Only an explicitly saved preferred solver advances it.
                if any(e.get("accepted") and not e.get("preserve_only")
                       for e in record["submissions"][submissions_before_turn:]):
                    await self.reveal_tests("training-passing-submit")
                    self.save(write_json)
                record["turn_log"][-1]["tools_s"] = round(time.monotonic() - tools_start, 2)
                if record.get("fresh_challenger") and record["accepted"] is not None:
                    record["stop"] = "distinct-alternative"
                    break
                if record["turn_log"][-1].get("progress_note_warning"):
                    self.messages.append({"role": "user", "content": (
                        "[harness note warning] The short note repair was incomplete or unavailable. "
                        "The original tool calls above executed normally. Continue from their feedback; "
                        "use a concise Known / Hypothesis / Next test note next time. Current reasoning "
                        "is retained for this recovery turn if it fits the context budget.")})
                event = self.tool_guard.observe(outcomes)
                if event:
                    record["tool_guard_events"].append(dict(turn=record["turns"], **event))
                    self.messages.append({"role": "user", "content": stall_feedback(event)})
                    print(f"TOOL_GUARD {self.folder.parent.name}/{self.folder.name}: "
                          f"turn={record['turns']} action={event['action']} "
                          f"reasons={','.join(event['reasons'])}", flush=True)
                    if event["action"] == "stop":
                        record["stop"] = "tool_stall"
                        break
                left = args.tir_repl_budget - record["completion_tokens"]
                if left < args.tir_repl_wrap_up and not record["warned"]:
                    record["warned"] = True
                    self.messages[-1]["content"] += WRAP_UP.format(left=left // 1000)
        except asyncio.CancelledError:
            record["stop"], record["error"], cancelled = "cancelled", "Cancelled: pipeline stopped or server exited", True
        except Exception as exc:  # noqa: BLE001 - keep the partial record and what the session solved
            record["stop"], record["error"] = "error", f"{type(exc).__name__}: {str(exc)[:500]}"
        if not cancelled and not record.get("fresh_challenger") and ctx["deadline"] - time.monotonic() > 30:
            try:
                await fallback_eval(self.session, self.task, record, holdout=self.holdout)
            except asyncio.CancelledError:
                cancelled = True
        record["seconds"] = round(time.monotonic() - start, 1)
        record["cancelled"] = cancelled
        self.save(write_json)
        return self.summary() if self.disclosure else record


# ----- one task ----------------------------------------------------------------------------


def predictions(records, n_test, *, require_train_pass=False, candidate_pool=False):
    picks = select(records, n_test, require_train_pass=require_train_pass, candidate_pool=candidate_pool)
    return [p[0]["grid"] if p else None for p in picks], [p[1]["grid"] if len(p) > 1 else None for p in picks], picks


async def solve_task(args, task, payload, count, server, output, deadline, *, prefix, specials, progress,
                     guard, slots, write_json, on_predictions=None, cpus=(), visualizer=None,
                     trajectory_state=None):
    """Trajectories until the task budget, --tir-repl-trajectories or two accepted ones agree.

    server: LlamaServer or VllmServer. payload: the request for the served prefix (prompt ids, sampling,
    seed). progress: the pipeline's stream_progress entry; decoded_before carries the tokens of finished
    requests. guard: a function returning the loop-guard keyword arguments of one request."""
    # The pipeline owns this in-memory continuation, not the model. Only one
    # trajectory runs per dispatch; no task has simultaneous mutable solvers.
    queued = trajectory_state is not None
    saved = trajectory_state if queued else {}
    if queued and saved and saved["output"] != str(output.resolve()):
        raise ValueError("Queued trajectory state belongs to a different output directory")
    if not saved:
        output.mkdir()
        (output / "prefix.txt").write_text(prefix.text if isinstance(prefix, CheckpointPrefix) else prefix)
        if isinstance(prefix, CheckpointPrefix):
            write_json(output / "prefix_request.json", prefix.request)
    start = time.monotonic()
    previous_elapsed = saved.get("elapsed_s", 0)
    # Waiting in the task queue is not execution time. The total active task
    # timeout is still shared by its trajectories; the global deadline never moves.
    deadline = min(deadline, start + max(0, args.request_timeout_seconds - previous_elapsed))
    task, holdout = split_holdout(task, getattr(args, "tir_repl_holdout_first", False))
    n_test = len(task["test"])
    progress.setdefault("decoded_before", 0)
    progress.setdefault("task_started", start)
    ctx = dict(args=args, server=server, payload=payload, prefix=prefix, prefix_ids=payload["prompt"],
               specials=specials, progress=progress, guard=guard, slots=slots, cpus=cpus, deadline=deadline,
               limit=args.context - CONTEXT_MARGIN, holdout=holdout, visualizer=visualizer)
    records, last = saved.get("records", []), saved.get("last")
    cancelled = any(r.get("cancelled") for r in records)
    prior_count = len(records)
    top, alt, picks = [None] * n_test, [None] * n_test, [[] for _ in range(n_test)]
    last_selection, last_published = saved.get("last_selection"), saved.get("last_published")

    def publish(current_records):
        """Use the unchanged selector, including the active trajectory once.

        Checkpoint publication is independent of review completion and does not
        mark a task finished, add a vote, relax training/holdout checks, or expose
        labels to inference. Suppress unchanged writes/callbacks during review.
        """
        nonlocal top, alt, picks, last_selection, last_published
        top, alt, picks = predictions(current_records, n_test,
            require_train_pass=getattr(args, "tir_repl_require_train_pass", False),
            candidate_pool=getattr(args, "tir_repl_candidate_pool", False))
        selection = dict(final_grids=top, alternative_grids=alt, picks=picks)
        identity = json.dumps(selection, sort_keys=True)
        if identity != last_selection:
            write_json(output / "selection.json", selection)
            last_selection = identity
        answer_identity = json.dumps([top, alt])
        if (on_predictions is not None and any(g is not None for g in top)
                and answer_identity != last_published):
            on_predictions(deepcopy(top), deepcopy(alt))
            last_published = answer_identity

    task_budget = getattr(args, "task_token_budget", None)
    limit = min(args.tir_repl_trajectories, prior_count + 1) if queued else args.tir_repl_trajectories
    for k in range(prior_count, limit):
        if not more_trajectories(args, records):
            break
        if deadline - time.monotonic() < args.min_request_seconds:
            break
        trajectory_args = copy(args)
        if task_budget is not None:
            remaining = task_budget - sum(r["completion_tokens"] for r in records)
            if remaining < 1000:  # Same minimum usable remainder as Trajectory.run.
                break
            trajectory_args.tir_repl_budget = min(args.tir_repl_budget, remaining)
        # Attach only to this ordinary trajectory; the optional anchored
        # challenger has its own protected-attempt merge, not this selector.
        trajectory = Trajectory(dict(ctx, args=trajectory_args,
            on_checkpoint=lambda current: publish([*records, current])), k, task, output / f"traj_{k}")
        record = await trajectory.run(write_json)
        records.append(trajectory.summary())
        last = trajectory.last or last
        publish(records)
        cancelled = record["cancelled"]
        if cancelled:
            break
        accepted = [json.dumps(r["submissions"][r["accepted"]]["test"]) for r in records if r["accepted"] is not None]
        if len(accepted) - len(set(accepted)) >= 1:
            break
    challenger_diagnostic = None
    if getattr(args, "tir_repl_challenger_tokens", 0) and not cancelled:
        if (records and records[0]["accepted"] is not None
                and any(g is not None and (a is None or a == g) for g, a in zip(top, alt))):
            import qwen38_repl_challenger as challenger
            primary = records[0]["submissions"][records[0]["accepted"]]
            challenger_record, challenger_diagnostic = await challenger.run(
                ctx, task, primary, trajectory.session.journal, top, output / "traj_1", write_json,
                used_tokens=sum(r["completion_tokens"] for r in records))
            if challenger_record is not None:
                alt, picks, changed = challenger.merge_alternative(top, alt, picks, challenger_record)
                challenger_diagnostic["changed_test_indices"] = changed
                records.append(challenger_record)
                cancelled = challenger_record.get("cancelled", False)
                write_json(output / "selection.json", dict(final_grids=top, alternative_grids=alt, picks=picks))
                if on_predictions is not None:
                    on_predictions(top, alt)
        else:
            challenger_diagnostic = dict(status="skipped", reason=(
                "no-accepted-primary" if not records or records[0]["accepted"] is None else "attempt-slots-filled"))
        write_json(output / "challenger.json", challenger_diagnostic)
    # Reconstruct preserved predictions even if no new request fits. A late or
    # cancelled retry must not replace the earlier candidate with placeholders.
    if queued:
        publish(records)
    elapsed = previous_elapsed + time.monotonic() - start
    completion = sum(r["completion_tokens"] for r in records)
    prompt = sum(r["prompt_tokens"] for r in records)
    decode_seconds = sum(r["decode_seconds"] for r in records)
    accepted = any(r["accepted"] is not None for r in records)
    errors = [r["error"] for r in records if r["error"]]
    status = ("cancelled" if cancelled else "accepted" if accepted else
              "fallback" if any(g is not None for g in top) else "error" if errors else "no-solution")
    result = dict(solve_mode="tir", tir_interface="repl", tir_status=status,
                  holdout_enabled=holdout is not None, visible_train_pairs=len(task["train"]),
                  task_token_budget=task_budget,
                  challenger=challenger_diagnostic,
                  task_budget_remaining=None if task_budget is None else max(0, task_budget - completion),
                  tir_rounds=sum(r["turns"] for r in records),
                  trajectories=[{k: v for k, v in r.items() if k not in ("submissions", "turn_log")} for r in records],
                  final_grids=top, alternative_grids=alt,
                  solver_paths=[str(output / "selection.json") if g is not None else None for g in top],
                  error=errors[-1] if errors and not accepted else None, cancelled=cancelled,
                  infrastructure_error=bool(errors) and not accepted,
                  finish_reason=("stop" if records and records[-1]["stop"] == "done"
                                 else records[-1]["stop"] if records else None),
                  trajectory_stop_reason=records[-1]["stop"] if records else None,
                  prediction_available=any(g is not None for g in top),
                  usage=dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
                  elapsed_s=elapsed,
                  decode_tokens_per_s_estimate=completion / decode_seconds if decode_seconds > 0 else None,
                  stop_type=(last or {}).get("stop_type"), think_closed=(last or {}).get("think_closed"),
                  streamed_tokens=completion, timings=(last or {}).get("timings"), loop_detection=None)
    write_json(output / "result.json", result)
    if queued:
        pending = (more_trajectories(args, records)
                   and args.request_timeout_seconds - elapsed > args.min_request_seconds
                   and deadline - time.monotonic() > args.min_request_seconds)
        saved.update(output=str(output.resolve()), records=records, last=last, elapsed_s=elapsed,
                     last_selection=last_selection, last_published=last_published, pending=pending)
        if len(records) > prior_count:
            write_json(output / f"trajectory_{len(records) - 1}_result.json", result)
    return result


def more_trajectories(args, records):
    """Label-blind eligibility shared by immediate and queued trajectory dispatch."""
    if len(records) >= args.tir_repl_trajectories or any(r.get("cancelled") for r in records):
        return False
    task_budget = getattr(args, "task_token_budget", None)
    if task_budget is not None and task_budget - sum(r["completion_tokens"] for r in records) < 1000:
        return False
    accepted = [json.dumps(r["submissions"][r["accepted"]]["test"])
                for r in records if r.get("accepted") is not None]
    return len(accepted) == len(set(accepted))


async def preflight(args, deadline):
    """The sandbox works and is closed: a cell, an evaluation, and a blocked open() before any model call."""
    task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}
    session = Session(task, asyncio.Semaphore(1), min(args.tir_repl_cell_timeout, 20), args.tir_repl_memory_mib,
                      qwen38_llama.free_cpus(args) if args.qwen38_variant == "flash-llama" else [])
    started = time.monotonic()
    cell = await session.cell("def solve(g):\n    return np.fliplr(g)\nprint(ndi.label(np.ones((2, 2)))[1])\n"
                              "check(solve)")
    blocked = await session.cell("open('/etc/hostname').read()")
    evaluation = await session.evaluate("solve")
    report = dict(mode="repl", runtime=str(RUNTIME), cell=cell, blocked=blocked[-200:],
                  evaluation={k: evaluation.get(k) for k in ("error", "train", "test")},
                  seconds=round(time.monotonic() - started, 2), cpus=session.cpus, python=sys.version.split()[0])
    if ("1\npair 0: PASS" not in cell or "PermissionError" not in blocked or evaluation.get("error")
            or evaluation["test"][0]["grid"] != [[4, 3]] or session.stats["fatal"]):
        raise RuntimeError(f"REPL sandbox preflight failed; no unsafe fallback: {report} {session.last_fatal}")
    return report
