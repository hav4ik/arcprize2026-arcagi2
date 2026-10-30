"""Standalone frozen baseline config, 6 hypothesis configs, prompt builders, and voting selector for pilot_2.

Written from scratch for pilot_2 with zero dependencies on pilot_1.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import arc_dsl_and_summary
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import sandbox_verifier


Grid = list[list[int]]


# ==============================================================================
# 1. FROZEN OPTIMAL BASELINE INVARIANTS (NEVER MODIFY ACROSS EXPERIMENTS)
# ==============================================================================

FROZEN_BASELINE_CONFIG: dict[str, Any] = {
    "model_path": "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "dataset_dir": "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "cns_output_root": "/cns/lu-d/home/arazdai/kaggle/pilot_2",
    "tensor_parallel_size": 4,
    "max_model_len": 131072,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": True,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "base_seed": 10000,
    "attempt_seed_stride": 1009,
    "stop_strings": ("</tool_call>", "<|im_end|>"),
    "think_nudge": '\n</think>\n<tool_call>\n{"name": "',
    "turn_think_grace": 2048,
    "think_call_rescue": True,
    "cell_timeout_s": 6.0,
    "max_tool_output_chars": 3500,
    "lock_first_verified_pred": True,
    "replay_cumulative_helpers": True,
    "forbid_train_global_in_solve": True,
}


@dataclasses.dataclass(frozen=True)
class Pilot2HypothesisConfig:
  """Complete, self-contained specification of one pilot_2 experiment."""

  exp_id: str
  title: str
  hypothesis_summary: str
  num_attempts: int
  max_turns: int
  total_token_budget: int
  soft_token_target: int
  turn_think_cap: int
  attempt_prompt_styles: tuple[str, ...]
  attempt_include_task_summary: tuple[bool, ...]
  use_arc_helpers: bool
  auto_check_on_solve: bool
  use_falsification_protocol: bool
  use_equivariance_audit: bool
  use_native_xml_tools: bool = False
  use_two_stage_think_nudge: bool = False
  use_evolutionary_two_wave: bool = False
  use_transcript_compaction: bool = False
  use_havu_b32_harness: bool = False

  temperature: float = FROZEN_BASELINE_CONFIG["temperature"]
  top_p: float = FROZEN_BASELINE_CONFIG["top_p"]
  top_k: int = FROZEN_BASELINE_CONFIG["top_k"]
  base_seed: int = FROZEN_BASELINE_CONFIG["base_seed"]
  attempt_seed_stride: int = FROZEN_BASELINE_CONFIG["attempt_seed_stride"]
  turn_think_grace: int = FROZEN_BASELINE_CONFIG["turn_think_grace"]
  think_call_rescue: bool = FROZEN_BASELINE_CONFIG["think_call_rescue"]
  lock_first_verified_pred: bool = FROZEN_BASELINE_CONFIG["lock_first_verified_pred"]

  def validate(self) -> None:
    """Assert that all frozen baseline invariants hold and attempt tuples match `num_attempts`."""
    assert self.temperature == FROZEN_BASELINE_CONFIG["temperature"], (
        f"Forbidden temperature modification in {self.exp_id}: {self.temperature}"
    )
    assert self.top_p == FROZEN_BASELINE_CONFIG["top_p"], (
        f"Forbidden top_p modification in {self.exp_id}: {self.top_p}"
    )
    assert self.top_k == FROZEN_BASELINE_CONFIG["top_k"], (
        f"Forbidden top_k modification in {self.exp_id}: {self.top_k}"
    )
    assert self.think_call_rescue is True
    assert self.lock_first_verified_pred is True
    assert len(self.attempt_prompt_styles) == self.num_attempts
    assert len(self.attempt_include_task_summary) == self.num_attempts
    for style in self.attempt_prompt_styles:
      assert style in (
          "lean2prog",
          "lean2progroles",
          "lean2sub",
      ), f"Unknown prompt style: {style}"


# ==============================================================================
# 2. CONTROLLED HYPOTHESES FOR PILOT_2 (H1-H6 + FOLLOW-UP H7-H8)
# ==============================================================================

PILOT2_HYPOTHESES: dict[str, Pilot2HypothesisConfig] = {
    "h1_python_tools_base": Pilot2HypothesisConfig(
        exp_id="h1_python_tools_base",
        title="H1: Just Python Tools Using Model (2 Attempts, 20 Turns, Clean Baseline)",
        hypothesis_summary=(
            "Baseline model with raw Python REPL (`run_python`, `submit_program`, `submit_answer`), "
            "2 attempts, 20 max turns, 48k budget, 8k think cap, no host summary, no preloaded DSL."
        ),
        num_attempts=2,
        max_turns=20,
        total_token_budget=48000,
        soft_token_target=24000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2prog", "lean2prog"),
        attempt_include_task_summary=(False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
    ),
    "h2_more_attempts_bo4": Pilot2HypothesisConfig(
        exp_id="h2_more_attempts_bo4",
        title="H2: More Program Attempts (N=4 Attempts with Behavioural Majority Voting)",
        hypothesis_summary=(
            "Single-axis change from H1: increase `num_attempts` from 2 to 4 at temp=1.0 "
            "with behavioral output-grid voting + Occam's Razor token-length tie-breaking."
        ),
        num_attempts=4,
        max_turns=20,
        total_token_budget=48000,
        soft_token_target=24000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2prog", "lean2prog", "lean2prog", "lean2prog"),
        attempt_include_task_summary=(False, False, False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
    ),
    "h3_more_max_turns_40": Pilot2HypothesisConfig(
        exp_id="h3_more_max_turns_40",
        title="H3: More Max Turns (40 Turns) + Auto-Check on `def solve(grid):`",
        hypothesis_summary=(
            "Single-axis change from H2: increase `max_turns` from 20 to 40 and enable "
            "`auto_check_on_solve=True` so defining `def solve(grid):` in `run_python` "
            "immediately reports per-pair diffs and locks `first_tp_pred` upon 100% train pass."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=48000,
        soft_token_target=24000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2prog", "lean2prog", "lean2prog", "lean2prog"),
        attempt_include_task_summary=(False, False, False, False),
        use_arc_helpers=False,
        auto_check_on_solve=True,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
    ),
    "h4_input_desc_and_helpers": Pilot2HypothesisConfig(
        exp_id="h4_input_desc_and_helpers",
        title="H4: Better Input Description (`task_summary` + Novelty Alerts) + Preloaded `arc_helpers` DSL",
        hypothesis_summary=(
            "Single-axis change from H3: add deterministic host perception & novelty summary "
            "(8-connected components, D8 shape letters, input->output diff summary, and "
            "'Outside train range' warnings) plus preloaded `arc_helpers` (`objects`, `bbox`, "
            "`crop`, `bg_colour`, `d8`, `match_d8`, `panels`, `stamp`, `ray`)."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=48000,
        soft_token_target=24000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2prog", "lean2prog", "lean2prog", "lean2prog"),
        attempt_include_task_summary=(True, True, True, True),
        use_arc_helpers=True,
        auto_check_on_solve=True,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
    ),
    "h5_longer_logical_reasoning": Pilot2HypothesisConfig(
        exp_id="h5_longer_logical_reasoning",
        title="H5: Better & Longer Logical Reasoning (`lean2progroles` + Falsification + 64k/12k Budget)",
        hypothesis_summary=(
            "Single-axis change from H4: upgrade reasoning protocol to `lean2progroles` + "
            "explicit functional role assignment & counter-hypothesis falsification, and expand "
            "deliberation budget (`total_token_budget=64000`, `soft_token_target=32000`, `turn_think_cap=12000`)."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=64000,
        soft_token_target=32000,
        turn_think_cap=12000,
        attempt_prompt_styles=(
            "lean2progroles",
            "lean2progroles",
            "lean2progroles",
            "lean2progroles",
        ),
        attempt_include_task_summary=(True, True, True, True),
        use_arc_helpers=True,
        auto_check_on_solve=True,
        use_falsification_protocol=True,
        use_equivariance_audit=False,
    ),
    "h6_adaptive_hybrid_ensemble": Pilot2HypothesisConfig(
        exp_id="h6_adaptive_hybrid_ensemble",
        title="H6: Adaptive Multi-Prompt Ensemble (2x Fast Lean + 2x Deep Roles) + Zero-Token Equivariance Audit",
        hypothesis_summary=(
            "Combines 2 fast Python-first `lean2prog` attempts (no summary bloat) with 2 deep "
            "`lean2progroles` + `task_summary` + falsification attempts for maximum hypothesis "
            "decorrelation, and breaks 1-vs-1 verified ties via zero-token D8/colour equivariance "
            "and AST constant-lint scoring (`use_equivariance_audit=True`)."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=56000,
        soft_token_target=28000,
        turn_think_cap=10000,
        attempt_prompt_styles=(
            "lean2prog",
            "lean2prog",
            "lean2progroles",
            "lean2progroles",
        ),
        attempt_include_task_summary=(False, False, True, True),
        use_arc_helpers=True,
        auto_check_on_solve=True,
        use_falsification_protocol=True,
        use_equivariance_audit=True,
    ),
    "h7_native_xml_and_grace_nudge": Pilot2HypothesisConfig(
        exp_id="h7_native_xml_and_grace_nudge",
        title="H7: Native Qwen3.8 XML `<function=...><parameter=code>` Tool Protocol + 2-Stage Graceful Think Nudge & Final Rescue",
        hypothesis_summary=(
            "Controlled upgrade over H6 (`2x fast lean2prog + 2x deep lean2progroles`, 40 turns, 56k/28k/10k): "
            "replaces JSON-escaped tool calls with Qwen3.8's native `chat_template.jinja` XML "
            "`<tool_call><function=run_python><parameter=code>...</parameter></function></tool_call>` format, "
            "starts assistant turns with `<think>\\n`, and uses a 2-stage graceful think nudge (in-thought "
            "reminder for 2048 tokens before forcing `<function=run_python><parameter=code>`) plus commit pacing."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=56000,
        soft_token_target=28000,
        turn_think_cap=10000,
        attempt_prompt_styles=(
            "lean2prog",
            "lean2prog",
            "lean2progroles",
            "lean2progroles",
        ),
        attempt_include_task_summary=(False, False, True, True),
        use_arc_helpers=True,
        auto_check_on_solve=True,
        use_falsification_protocol=True,
        use_equivariance_audit=True,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=True,
        use_evolutionary_two_wave=False,
        use_transcript_compaction=False,
    ),
    "h8_evolutionary_two_wave_repair": Pilot2HypothesisConfig(
        exp_id="h8_evolutionary_two_wave_repair",
        title="H8: Two-Wave Evolutionary Cross-Attempt Program Repair & Counterfactual Audit + Transcript Compaction",
        hypothesis_summary=(
            "Builds on H7 (`use_native_xml_tools=True`, `use_two_stage_think_nudge=True`) and schedules "
            "the 4 attempts in two evolutionary waves: Wave 1 runs Attempt 0 (`lean2prog`) + Attempt 1 "
            "(`lean2progroles`). Wave 2 (Attempts 2 & 3) uses cross-attempt evolutionary code repair (seeding "
            "Attempt 3 with Wave 1's best partial `def solve(grid):` and exact failing pair diffs when Wave 1 "
            "fails to verify, or seeding counterfactual anti-overfit falsification when Wave 1 verifies) "
            "plus E027 old-turn `<think>` compaction to preserve KV/attention focus over 40 turns."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=56000,
        soft_token_target=28000,
        turn_think_cap=10000,
        attempt_prompt_styles=(
            "lean2prog",
            "lean2progroles",
            "lean2prog",
            "lean2progroles",
        ),
        attempt_include_task_summary=(False, True, False, True),
        use_arc_helpers=True,
        auto_check_on_solve=True,
        use_falsification_protocol=True,
        use_equivariance_audit=True,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=True,
        use_evolutionary_two_wave=True,
        use_transcript_compaction=True,
    ),
    "e031_deepseek_teaching": Pilot2HypothesisConfig(
        exp_id="e031_deepseek_teaching",
        title="E031 DeepSeek Teaching Interface (lean2sub + xhigh Tier + Minimal Effort Header + 8k 2-Stage Think Nudge, N=2)",
        hypothesis_summary=(
            "Optimal E031 / E032 DeepSeek teaching harness: ARC Prize user prompt + lean2sub suffix, "
            "Qwen3.8 native `xhigh` tier header + `minimal` effort header after tools, native XML tool calls "
            "(`run_python` / `submit_answer` in persistent IPython session), 8k per-turn thinking cap with "
            "2-stage graceful think nudge, one-time train-check verification on `submit_answer`, "
            "equivariance tie-breaking, and `force_submit_last` 3-tier session fallback."
        ),
        num_attempts=2,
        max_turns=40,
        total_token_budget=60000,
        soft_token_target=20000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=True,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=True,
        use_evolutionary_two_wave=False,
        use_transcript_compaction=True,
    ),
    "e031_deepseek_teaching_bo4": Pilot2HypothesisConfig(
        exp_id="e031_deepseek_teaching_bo4",
        title="E031 DeepSeek Teaching Interface (lean2sub + xhigh Tier + 2-Wave Evolutionary Repair & Anti-Overfit, N=4)",
        hypothesis_summary=(
            "Optimal E031 / E032 DeepSeek teaching harness with N=4 attempts scheduled in 2 evolutionary waves "
            "(Wave 1 runs 2 parallel attempts; Wave 2 runs evolutionary code repair on near-miss rules or "
            "counterfactual anti-overfit verification), `xhigh` tier header, 8k 2-stage graceful think nudge, "
            "D8+color equivariance tie-breaking, and 3-tier session fallback."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=60000,
        soft_token_target=20000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2sub", "lean2sub", "lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False, False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=True,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=True,
        use_evolutionary_two_wave=True,
        use_transcript_compaction=True,
    ),
    "e031_deepseek_teaching_fast": Pilot2HypothesisConfig(
        exp_id="e031_deepseek_teaching_fast",
        title="Fast E031 DeepSeek Teaching Interface (lean2sub + xhigh Tier + 42k/36t + 8k 2-Stage Nudge, N=2)",
        hypothesis_summary=(
            "Fast Kaggle-6h compliant E031 DeepSeek teaching harness (`lean2sub`, `xhigh` tier + `minimal` effort "
            "header, persistent IPython `run_python` & `submit_answer`) with 42k token budget, 36 max turns, "
            "8k per-turn think cap with 2-stage graceful nudge, equivariance tie-breaking, and `force_submit_last` "
            "session fallback."
        ),
        num_attempts=2,
        max_turns=36,
        total_token_budget=42000,
        soft_token_target=20000,
        turn_think_cap=8192,
        attempt_prompt_styles=("lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=True,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=True,
        use_evolutionary_two_wave=False,
        use_transcript_compaction=True,
    ),
    "havu_b32": Pilot2HypothesisConfig(
        exp_id="havu_b32",
        title="Havu Candidate B32 Harness (Multi-Test + Auto-Repair + Token-Exact + 80k/40k Gentle + 64k Uninterrupted + 4-Path Train-Check, N=2)",
        hypothesis_summary=(
            "Exact replica of Havu's latest best-performing Candidate B32 harness from hav4ik/arcprize2026-arcagi2: "
            "ARC_MULTI_TEST=1 with ARC_MULTI_BUDGET=0.5, retry-once AST-safe identifier typo auto-repair (ARC_AUTO_REPAIR=1), "
            "--token-exact prompt IDs with zero mid-run <think> compaction, 64k uninterrupted per-turn max_tokens, "
            "80k/40k gentle budget notes, isolated child-process LocalExecutor (style='ipython'), A1-A11 fallback fixes, "
            "--rescue-code, GuardedStream thinking gzip loop monitor, and 4-path post-attempt train_check + ranked_answers."
        ),
        num_attempts=2,
        max_turns=40,
        total_token_budget=80000,
        soft_token_target=40000,
        turn_think_cap=64000,
        attempt_prompt_styles=("lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=False,
        use_evolutionary_two_wave=False,
        use_transcript_compaction=False,
        use_havu_b32_harness=True,
    ),
    "havu_b32_bo4": Pilot2HypothesisConfig(
        exp_id="havu_b32_bo4",
        title="Havu Candidate B32 Harness (Multi-Test + Auto-Repair + Token-Exact + 80k/40k Gentle + 64k Uninterrupted + 4-Path Train-Check, N=4)",
        hypothesis_summary=(
            "Controlled N=4 scaling of Havu's Candidate B32 harness (ARC_MULTI_TEST=1, ARC_MULTI_BUDGET=0.5, "
            "ARC_AUTO_REPAIR=1, --token-exact, 80k/40k gentle, 64k uninterrupted per-turn max_tokens, isolated "
            "LocalExecutor, A1-A11 fixes, --rescue-code, GuardedStream loop guard, 4-path train_check + ranked_answers) "
            "for apples-to-apples comparison against e031_deepseek_teaching_bo4 (N=4)."
        ),
        num_attempts=4,
        max_turns=40,
        total_token_budget=80000,
        soft_token_target=40000,
        turn_think_cap=64000,
        attempt_prompt_styles=("lean2sub", "lean2sub", "lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False, False, False),
        use_arc_helpers=False,
        auto_check_on_solve=False,
        use_falsification_protocol=False,
        use_equivariance_audit=False,
        use_native_xml_tools=True,
        use_two_stage_think_nudge=False,
        use_evolutionary_two_wave=False,
        use_transcript_compaction=False,
        use_havu_b32_harness=True,
    ),
}

for _cfg in PILOT2_HYPOTHESES.values():
  _cfg.validate()


# ==============================================================================
# 3. SYSTEM & USER PROMPT BUILDERS (RAW CHATML FOR 100% PREFIX CACHE HIT)
# ==============================================================================

E031_ARC_PRIZE_TEMPLATE = (
    "You are participating in a puzzle solving competition. You are an expert at"
    " solving puzzles.\n\nBelow is a list of input and output pairs with a"
    " pattern. Your goal is to identify the pattern or transformation in the"
    " training examples that maps the input to the output, then apply that"
    " pattern to the test input to give a final output.\n\nRespond in the"
    " format of the training output examples\n\n--Training"
    " Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test"
    " Input--\n{test_input}\n--End of Test Input--\n\nYour response:"
)

E031_LEAN2SUB_SUFFIX = (
    "\n\nYou have a Python tool. The variables `train` (list of {'input',"
    " 'output'} grids) and `test_input` are already loaded in it: use them"
    " directly and never retype or reassign them. Work efficiently: inspect the"
    " grids with code instead of reading them by hand, form a hypothesis, and"
    " write code that computes the output. Never construct, copy or count grids"
    " by hand. Check your rule on every training pair with code before"
    " answering, and if a pair fails, fix the rule rather than explaining the"
    " mismatch away. Before answering, check that the rule's assumptions also"
    " hold on test_input. Keep your reasoning brief."
    "\n\nWhen your code reproduces every training output and its assumptions"
    " hold on test_input, call submit_answer with an expression such as"
    " `solve(test_input)`: the harness evaluates it in your session and that"
    " grid is your answer, so never retype the grid. Keep your current best"
    " guess for the test output in a variable `answer`; if you run out of"
    " budget it is submitted automatically."
)

E031_TOOLS_SPEC_JSON = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
                "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
                "scipy is available. No internet or host files. Timeout 30s. "
                "Use print() to inspect results. Check transformations on all training pairs."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "End this attempt with your test output. `code` is a Python expression evaluated in your session, "
                "e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); "
                "statements that set a variable `answer` also work. The harness reads the grid directly: do not "
                "retype it. Invalid grids return an error and the attempt continues."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]

E031_TOOLS_FOOTER = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)

TIER_SYSTEM_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
    "answer."
)

E031_TIER_MINIMAL = TIER_SYSTEM_XHIGH

E031_EFFORT_HEADER_MINIMAL = (
    "Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. "
    "Do not restate the task, do not re-read or re-derive what code has already printed, and do not re-verify "
    "what has already been checked. Test ideas with small code experiments instead of long deliberation, and "
    "submit as soon as the rule reproduces every training pair."
)

E031_FORCE_SUBMIT_FINAL = (
    "\n\nI am out of budget, so I must submit my best answer now as one short expression: `answer` if I "
    "set it, otherwise solve(test_input) or the name of my best function applied to test_input.\n</think>\n\n"
    "<tool_call>\n<function=submit_answer>\n<parameter=code>\n"
)


def format_e031_budget_and_turns_note(
    used_tokens: int | None = None,
    turn_idx_1based: int | None = None,
    max_turns: int = 40,
    soft_target: int = 20000,
    hard_cap: int = 60000,
    turns_note_threshold: int = 5,
    *,
    cum_gen_tokens: int | None = None,
    turn_idx_after_tool: int | None = None,
    hard_budget: int | None = None,
    stated_hard_cap: int = 60000,
) -> str:
  """Format exact E031 / E032 budget and turns-left notes appended inside `<tool_response>`."""
  toks = int(cum_gen_tokens if cum_gen_tokens is not None else (used_tokens or 0))
  t_idx = int(
      turn_idx_after_tool
      if turn_idx_after_tool is not None
      else (turn_idx_1based or 0)
  )
  real_cap = int(hard_budget if hard_budget is not None else hard_cap)
  display_cap = max(real_cap, int(stated_hard_cap))
  note = ""
  if toks >= int(0.01 * real_cap):
    note = (
        f"\n[budget: {toks / 1000:.0f}k output tokens used; aim to finish within "
        f"{soft_target / 1000:.0f}k (hard limit {display_cap / 1000:.0f}k)]"
    )
    if soft_target <= toks < int(0.75 * real_cap):
      note += (
          " Past the target: submit your best answer now with submit_answer, unless one more experiment "
          "will very likely make every training pair pass."
      )
    elif toks >= int(0.75 * real_cap):
      note += (
          " Budget nearly spent: submit your best answer now with submit_answer "
          "(or at least set `answer`)."
      )
  left = max_turns - t_idx
  if turns_note_threshold and 0 < left <= turns_note_threshold:
    note += (
        f"\n[{left} turn{'s' if left != 1 else ''} left: call submit_answer with your best answer now, "
        "or at least set `answer`.]"
    )
  return note


TOOLS_SPEC_JSON = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a persistent stateful REPL with `train`, `test_input`, "
                "`np`, `scipy`, `ndimage`, and `check(fn)` preloaded. Use this to inspect grids, "
                "test hypotheses, and debug `def solve(grid):`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_program",
            "description": (
                "Submit a Python program defining `def solve(grid):` (taking a 2D numpy array "
                "and returning a 2D numpy array or list of lists of ints 0..9). The harness "
                "verifies `solve(pair['input']) == pair['output']` on all training pairs and, "
                "if all pass, runs `solve(test_input)` as your verified prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code defining `def solve(grid):`.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Fallback tool: directly submit the predicted 2D output grid (list of lists of "
                "ints 0..9) for `test_input`. Always prefer `submit_program` whenever possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "grid": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "2D grid of integers 0..9.",
                    }
                },
                "required": ["grid"],
            },
        },
    },
]

XML_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=run_python>\n"
    "<parameter=code>\n"
    "YOUR_PYTHON_CODE\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- Put raw unescaped Python inside <parameter=code>...</parameter> (do NOT JSON-escape newlines or quotes)\n"
    "</IMPORTANT>"
)

XML_THINK_NUDGE_STAGE1 = (
    "\n\nI have been reasoning for a long time without running anything. "
    "I should stop here and test my current hypothesis with code now "
    "(and use code, not my head, for any counting or cell reading).\n"
)

XML_THINK_FORCE_ACT_STAGE2 = (
    "\n\nLet me test this with code now.\n</think>\n\n"
    "<tool_call>\n<function=run_python>\n<parameter=code>\n"
)

XML_COMMIT_FORCE_ACT_STAGE2 = (
    "\n\nI have used a large part of my budget without defining `def solve(grid)`. "
    "Let me write my current best hypothesis as `def solve(grid):` now so the harness "
    "auto-checks it on every training pair.\n</think>\n\n"
    "<tool_call>\n<function=run_python>\n<parameter=code>\ndef solve(grid):\n"
)

XML_FORCE_SUBMIT_FINAL = (
    "\n\nI am at the end of my budget, so I must submit my best general `def solve(grid):` "
    "program now.\n</think>\n\n"
    "<tool_call>\n<function=submit_program>\n<parameter=code>\n"
)


def build_system_prompt(cfg: Pilot2HypothesisConfig, attempt_idx: int) -> str:
  """Build the system prompt with tool definitions for a specific attempt."""
  style = cfg.attempt_prompt_styles[attempt_idx]
  if style == "lean2sub":
    e031_tools_lines = "".join(
        "\n" + json.dumps(t, ensure_ascii=False) for t in E031_TOOLS_SPEC_JSON
    )
    return (
        f"{TIER_SYSTEM_XHIGH}\n\n"
        "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        f"{e031_tools_lines}{E031_TOOLS_FOOTER}\n\n"
        f"{E031_EFFORT_HEADER_MINIMAL}"
    )

  tools_lines = "\n".join(json.dumps(t) for t in TOOLS_SPEC_JSON)

  if cfg.use_arc_helpers:
    helpers_doc = (
        "\nPreloaded spatial helpers in Python (in addition to `train`, `test_input`, `np`, `scipy`, `ndimage`):\n"
        "- `bg_colour(grid) -> int`: most frequent colour.\n"
        "- `objects(grid, bg=None, connectivity=8, multicolour=False) -> list[dict]`: "
        "connected components sorted top-to-bottom, left-to-right; each dict has "
        "`'colour'`, `'size'`, `'bbox'` `(r0,c0,r1,c1)`, `'patch'`, `'mask'`, `'coords'`.\n"
        "- `bbox(mask_or_coords) -> (r0, c0, r1, c1)` inclusive.\n"
        "- `crop(grid, box=None, bg=0) -> 2D array`.\n"
        "- `d8(patch) -> list[2D array]` (8 dihedral rotations/flips) & `match_d8(a, b, ignore_colour=False) -> bool`.\n"
        "- `panels(grid) -> 2D list of sub-grids` split along full-row/col separator lines.\n"
        "- `stamp(canvas, patch, r0, c0, transparent=None) -> 2D array`.\n"
        "- `ray(grid, r, c, dr, dc, colour=None, stop_at=None)`.\n"
        "- `check(solve)` tests `solve(grid)` on every training pair and prints exact diffs.\n"
    )
  else:
    helpers_doc = (
        "\nPreloaded in Python: `train` (list of `{'input': np.ndarray, 'output': np.ndarray}`), "
        "`test_input` (`np.ndarray`), `np`, `scipy`, `ndimage`, and `check(solve)`.\n"
    )

  if style == "lean2progroles":
    core_instructions = (
        "You are an expert ARC-AGI-2 puzzle solver with a stateful Python REPL.\n"
        "Follow this disciplined workflow:\n"
        "1. ASSIGN ROLES & HYPOTHESIZE: Identify every distinct object, marker, divider, "
        "or colour in the grids and assign its exact functional role (e.g., template, canvas, "
        "anchor/marker, colour legend/key, symmetry axis, obstacle, ray source). Verify that "
        "these roles explain EVERY training example without exception.\n"
    )
    if cfg.use_falsification_protocol:
      core_instructions += (
          "2. FALSIFY NAIVE RULES & CHECK TEST DIFFERENCES: Before finalizing a rule, actively "
          "ask how a naive shortcut could overfit the training pairs. Inspect `test_input` in "
          "Python for novel dimensions, new colours, rotated orientations, or different numbers "
          "of objects (`Outside train range`). Never hardcode training grid sizes or indices.\n"
          "3. VERIFY & SUBMIT PROGRAM: Write a general `def solve(grid):` function, verify it "
          "achieves 100% match across all `train` pairs, inspect `solve(test_input)` for structural "
          "sanity, and call `submit_program`.\n"
      )
    else:
      core_instructions += (
          "2. PROTOTYPE & VERIFY IN PYTHON: Use `run_python` to inspect `train` and `test_input`, "
          "test your transformation rule with `def solve(grid):`, and call `submit_program` once "
          "all training pairs pass.\n"
      )
  else:
    core_instructions = (
        "You are an expert ARC-AGI-2 solver. Solve the transformation puzzle by inspecting "
        "`train` and `test_input` in Python, writing a clean general `def solve(grid):` function "
        "that transforms any input grid into its output grid, verifying it passes all training "
        "examples, and calling `submit_program`.\n"
        "- Keep `<think>` concise and use `run_python` early to test concrete hypotheses.\n"
        "- Never hardcode training dimensions, coordinates, or example-specific constants.\n"
    )

  if cfg.use_native_xml_tools:
    return (
        f"{TIER_SYSTEM_XHIGH}\n\n"
        f"{core_instructions}"
        f"{helpers_doc}\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        f"<tools>\n{tools_lines}{XML_TOOLS_FOOTER}"
    )

  return (
      f"{core_instructions}"
      f"{helpers_doc}\n"
      "# Tools\n\n"
      "You may call one tool per turn by wrapping a JSON object in `<tool_call></tool_call>` tags:\n"
      f"<tools>\n{tools_lines}\n</tools>"
  )


def build_user_prompt(
    train_pairs: list[dict[str, Any]],
    test_input: Grid,
    cfg: Pilot2HypothesisConfig,
    attempt_idx: int,
    extra_seed_note: str = "",
) -> str:
  """Build the initial user message containing the ARC-AGI-2 grids and optional perception/evolutionary summary."""
  style = cfg.attempt_prompt_styles[attempt_idx]
  if style == "lean2sub":
    ex = ""
    for i, pair in enumerate(train_pairs):
      ex += (
          f"--Example {i}-- \n\n INPUT: \n\n"
          + json.dumps(pair["input"])
          + "\n\nOUTPUT: \n\n"
          + json.dumps(pair["output"])
          + "\n\n"
      )
    base_prompt = (
        E031_ARC_PRIZE_TEMPLATE.format(
            training_examples=ex, test_input=json.dumps(test_input)
        )
        + E031_LEAN2SUB_SUFFIX
    )
    if extra_seed_note:
      base_prompt += "\n\n" + extra_seed_note.strip()
    return base_prompt.strip()

  blocks: list[str] = [
      "Find the common rule that maps an input grid to an output grid, given the examples below."
  ]
  for idx, pair in enumerate(train_pairs, start=1):
    inp = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    blocks.append(
        f"\nExample {idx}:\n\n"
        f"Input ({inp.shape[0]}x{inp.shape[1]}):\n{sandbox_verifier.format_grid_compact(inp)}\n"
        f"Output ({out.shape[0]}x{out.shape[1]}):\n{sandbox_verifier.format_grid_compact(out)}"
    )

  t_arr = np.asarray(test_input, dtype=int)
  blocks.append(
      "\nBelow is a test input grid. Predict the corresponding output grid by writing "
      "and verifying `def solve(grid):` in Python and calling `submit_program`.\n\n"
      f"Input ({t_arr.shape[0]}x{t_arr.shape[1]}):\n{sandbox_verifier.format_grid_compact(t_arr)}"
  )

  if cfg.attempt_include_task_summary[attempt_idx]:
    summary_text = arc_dsl_and_summary.build_task_summary(train_pairs, test_input)
    blocks.append("\n" + summary_text)

  if extra_seed_note:
    blocks.append("\n" + extra_seed_note.strip())

  blocks.append(
      f"\n[Budget: up to {cfg.max_turns} tool turns | "
      f"{cfg.total_token_budget} total tokens | "
      f"{cfg.turn_think_cap} max thinking tokens per turn]"
  )
  return "\n".join(blocks)


def format_initial_chatml_transcript(
    system_prompt: str,
    user_prompt: str,
    use_native_xml_tools: bool = False,
) -> str:
  """Format the initial raw Qwen ChatML prompt string."""
  assistant_prefix = (
      "<|im_start|>assistant\n<think>\n"
      if use_native_xml_tools
      else "<|im_start|>assistant\n"
  )
  return (
      f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
      f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
      f"{assistant_prefix}"
  )


def append_turn_to_chatml_transcript(
    transcript: str,
    assistant_output: str,
    tool_response: str,
    budget_note: str = "",
    use_native_xml_tools: bool = False,
) -> str:
  """Append an assistant completion and its `<tool_response>` to the append-only ChatML transcript."""
  clean_assistant = assistant_output
  if clean_assistant.endswith("<|im_end|>"):
    clean_assistant = clean_assistant[: -len("<|im_end|>")]
  if use_native_xml_tools and "</think>" not in clean_assistant:
    if "<tool_call>" in clean_assistant:
      tc_idx = clean_assistant.find("<tool_call>")
      clean_assistant = (
          clean_assistant[:tc_idx].rstrip()
          + "\n</think>\n\n"
          + clean_assistant[tc_idx:]
      )
    else:
      clean_assistant = clean_assistant.rstrip() + "\n</think>"
  clean_note = budget_note.lstrip("\n") if budget_note else ""
  note_suffix = f"\n{clean_note}" if clean_note else ""
  assistant_prefix = (
      "<|im_start|>assistant\n<think>\n"
      if use_native_xml_tools
      else "<|im_start|>assistant\n"
  )
  return (
      f"{transcript}{clean_assistant}<|im_end|>\n"
      f"<|im_start|>user\n<tool_response>\n{tool_response}{note_suffix}\n</tool_response><|im_end|>\n"
      f"{assistant_prefix}"
  )


_SEG_MARKER = "<|im_start|>"
_THINK_BODY_RE = re.compile(
    r"(<\|im_start\|>assistant\n(?:<think>\n)?).*?(\n?</think>)", re.DOTALL
)
_TOOL_RESP_BODY_RE = re.compile(
    r"(<tool_response>\n)(.*?)(\n</tool_response>)", re.DOTALL
)


def compact_chatml_transcript(
    transcript: str,
    keep_recent_turns: int = 2,
    max_old_tool_chars: int = 1500,
) -> str:
  """E027 transcript compaction: empty `<think>` in older completed turns while keeping recent turns verbatim."""
  parts = transcript.split(_SEG_MARKER)
  if len(parts) <= 4:
    return transcript
  head = parts[0]
  segs = [_SEG_MARKER + p for p in parts[1:]]
  done_asst = [
      i
      for i, s in enumerate(segs)
      if s.startswith(_SEG_MARKER + "assistant\n") and s.endswith("<|im_end|>\n")
  ]
  tool_msgs = [
      i
      for i, s in enumerate(segs)
      if s.startswith(_SEG_MARKER + "user") and "<tool_response>" in s
  ]
  old_asst = set(done_asst[:-keep_recent_turns] if keep_recent_turns else done_asst)
  old_tool = set(tool_msgs[:-keep_recent_turns] if keep_recent_turns else tool_msgs)

  def _cut_tool(m: re.Match[str]) -> str:
    body = m.group(2)
    if len(body) <= max_old_tool_chars or body.endswith(" chars compacted]"):
      return m.group(0)
    removed = len(body) - max_old_tool_chars
    return (
        f"{m.group(1)}{body[:max_old_tool_chars]}\n"
        f"[... {removed} chars compacted]{m.group(3)}"
    )

  out_segs: list[str] = []
  for i, s in enumerate(segs):
    if i in old_asst:
      s = _THINK_BODY_RE.sub(
          lambda m: "<|im_start|>assistant\n<think>\n\n</think>", s, count=1
      )
    elif i in old_tool:
      s = _TOOL_RESP_BODY_RE.sub(_cut_tool, s)
    out_segs.append(s)
  return head + "".join(out_segs)


def build_wave2_evolutionary_seed(
    wave1_sandboxes: list[sandbox_verifier.Pilot2AttemptSandbox],
    wave2_attempt_idx: int,
) -> str:
  """Build an evolutionary repair or counterfactual falsification seed from Wave 1 (Attempts 0 & 1)."""
  is_e031 = any(getattr(sb, "use_e031_ipython_mode", False) for sb in wave1_sandboxes)
  submit_tool_str = (
      "`submit_answer` with `solve(test_input)`" if is_e031 else "`submit_program`"
  )
  verified_sbs = [sb for sb in wave1_sandboxes if sb.first_tp_pred is not None]
  partial_sbs = [
      sb
      for sb in wave1_sandboxes
      if sb.best_partial_code is not None
      and (sb.best_partial_train_frac > 0.0 or sb.best_partial_cell_acc >= 0.50)
  ]

  # If Wave 1 failed to verify any program (or on Attempt 3 when Wave 1 had only 0 or 1 verified and a strong partial),
  # seed with the best partial candidate program and its exact failing training diffs for evolutionary repair.
  if (not verified_sbs and partial_sbs) or (
      wave2_attempt_idx == 3 and len(verified_sbs) <= 1 and partial_sbs
  ):
    best_sb = max(
        partial_sbs,
        key=lambda s: (s.best_partial_train_frac, s.best_partial_cell_acc),
    )
    reports_str = "\n".join(best_sb.best_partial_pair_reports[:6])
    code_snippet = (best_sb.best_partial_code or "").strip()
    if len(code_snippet) > 2400:
      code_snippet = code_snippet[:2400] + "\n# ... [truncated]"
    return (
        "[Wave 1 Evolutionary Program Repair Seed]\n"
        f"An earlier attempt wrote a near-miss rule achieving "
        f"{best_sb.best_partial_train_frac*100:.1f}% full-pair pass rate "
        f"({best_sb.best_partial_cell_acc*100:.1f}% mean cell accuracy) across the training examples.\n"
        f"Per-pair diagnostic diffs:\n{reports_str}\n"
        f"Near-miss candidate code from Wave 1:\n```python\n{code_snippet}\n```\n"
        "In Python, diagnose why this candidate failed on the remaining training pair(s) "
        "(or whether it used an overly narrow assumption), fix or rewrite `def solve(grid):` "
        f"so all training pairs pass 100%, and call {submit_tool_str}."
    )

  # If Wave 1 found at least one verified program, seed Wave 2 with a counterfactual anti-overfit audit note.
  if verified_sbs:
    v_sb = verified_sbs[0]
    pred_arr = np.asarray(v_sb.first_tp_pred, dtype=int)
    vals, cnts = np.unique(pred_arr, return_counts=True)
    hist_str = ", ".join(
        f"{int(v)}:{int(c)}"
        for v, c in sorted(zip(vals, cnts), key=lambda x: (-x[1], x[0]))
    )
    return (
        "[Wave 1 Cross-Attempt Falsification & Anti-Overfit Note]\n"
        f"An earlier attempt found a training-passing rule that predicts a "
        f"{pred_arr.shape[0]}x{pred_arr.shape[1]} test output grid with colour counts ({hist_str}) "
        f"and equivariance score {v_sb.first_tp_equivariance_bonus:.2f}.\n"
        "WARNING: Training-passing programs can still fail on `test_input` if they rely on a "
        "2- or 3-example coincidence (e.g. anchoring to row 0 instead of a marker, or assuming symmetry). "
        "Independently inspect `train` and `test_input` in Python, check all novel features of `test_input`, "
        f"derive a clean general `def solve(grid):`, and call {submit_tool_str}."
    )

  return ""


# ==============================================================================
# 4. TOOL CALL PARSER WITH `think_call_rescue` (SUPPORTS XML & JSON)
# ==============================================================================

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_UNCLOSED_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_UNCLOSED_FUNC_RE = re.compile(
    r"<function=([^>\n]+)>\s*<parameter=([^>\n]+)>\n?(.*)",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)


def parse_tool_call_from_output(
    text: str,
    allow_think_call_rescue: bool = True,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
  """Parse `(tool_name, arguments_dict, parse_error)` from XML `<function=...>` or JSON `<tool_call>`."""
  search_region = text
  if not allow_think_call_rescue and "<think>" in text and "</think>" not in text:
    return None, None, "Unclosed <think> block without </think>."

  # 1. Try native Qwen3.8 XML <tool_call><function=NAME><parameter=KEY>...</parameter></function></tool_call>
  xml_matches = list(_XML_TOOL_CALL_RE.finditer(search_region))
  if xml_matches:
    last_m = xml_matches[-1]
    fn_name = last_m.group(1).strip()
    body = last_m.group(2)
    params: dict[str, Any] = {
        k.strip(): v for k, v in _XML_PARAM_RE.findall(body)
    }
    if not params and "<parameter=" in body:
      m_unclosed_p = re.search(
          r"<parameter=([^>\n]+)>\n?(.*)", body, re.DOTALL
      )
      if m_unclosed_p:
        params[m_unclosed_p.group(1).strip()] = m_unclosed_p.group(2).strip()
    if fn_name == "submit_answer" and isinstance(params.get("grid"), str):
      try:
        params["grid"] = json.loads(params["grid"])
      except Exception:
        pass
    if fn_name == "submit_answer" and "grid" not in params and "code" in params:
      try:
        params["grid"] = json.loads(params["code"])
      except Exception:
        pass
    return fn_name, params, None

  # 2. Try unclosed XML <function=NAME><parameter=KEY>... (e.g. after force-act nudge hit length cap)
  xml_open_matches = list(_XML_UNCLOSED_FUNC_RE.finditer(search_region))
  if xml_open_matches:
    last_open = xml_open_matches[-1]
    fn_name = last_open.group(1).strip()
    p_key = last_open.group(2).strip()
    p_val = last_open.group(3)
    for trail in (
        "</parameter></function></tool_call>",
        "</parameter>\n</function>\n</tool_call>",
        "</parameter>\n</function>",
        "</parameter>",
    ):
      if p_val.endswith(trail):
        p_val = p_val[: -len(trail)]
    params = {p_key: p_val.strip()}
    if fn_name == "submit_answer" and isinstance(params.get("grid"), str):
      try:
        params["grid"] = json.loads(params["grid"])
      except Exception:
        pass
    return fn_name, params, None

  # 3. Try JSON <tool_call>{"name": ..., "arguments": ...}</tool_call>
  matches = list(_TOOL_CALL_RE.finditer(search_region))
  raw_json = None
  if matches:
    raw_json = matches[-1].group(1).strip()
  else:
    m_open = _UNCLOSED_TOOL_CALL_RE.search(search_region)
    if m_open:
      candidate = m_open.group(1).strip()
      for suffix in ("", '"}', '"}}\n', "}", "}}"):
        try:
          obj = json.loads(candidate + suffix)
          if isinstance(obj, dict) and "name" in obj:
            raw_json = candidate + suffix
            break
        except Exception:
          continue

  if raw_json is None:
    return None, None, "No `<tool_call>...</tool_call>` found in assistant response."

  try:
    payload = json.loads(raw_json)
  except Exception as exc:
    return None, None, f"Invalid JSON inside `<tool_call>`: {exc}"

  if not isinstance(payload, dict):
    return None, None, "Tool call JSON must be an object."

  name = payload.get("name")
  args = payload.get("arguments", {})
  if isinstance(args, str):
    try:
      args = json.loads(args)
    except Exception as exc:
      return str(name), None, f"Could not parse tool `arguments` JSON string: {exc}"
  if not isinstance(args, dict):
    return str(name), None, "Tool `arguments` must be a dictionary."
  return str(name), args, None



# ==============================================================================
# 5. FROZEN BEHAVIOURAL MAJORITY VOTING & OCCAM'S RAZOR SELECTOR
# ==============================================================================


@dataclasses.dataclass
class AttemptCandidateSummary:
  """Summary of a completed attempt used by the candidate selector."""

  attempt_idx: int
  prompt_style: str
  has_task_summary: bool
  total_turns: int
  total_gen_tokens: int
  train_pass: bool
  verified_grid: Grid | None
  verified_turn: int | None
  verified_gen_tokens: int | None
  equivariance_bonus: float
  secondary_verified_grids: list[tuple[Grid, int, int, float]] = dataclasses.field(
      default_factory=list
  )
  partial_grid: Grid | None = None
  partial_train_frac: float = 0.0
  partial_cell_acc: float = 0.0
  partial_gen_tokens: int = 0
  fallback_grid: Grid | None = None
  fallback_gen_tokens: int = 0


def _grid_cell_similarity(a: Grid, b: Grid) -> float:
  if len(a) != len(b) or not a or not b or len(a[0]) != len(b[0]):
    return 0.0
  arr_a = np.asarray(a, dtype=int)
  arr_b = np.asarray(b, dtype=int)
  return float(np.mean(arr_a == arr_b))


def select_top_two_predictions(
    attempts: list[AttemptCandidateSummary],
    test_input: Grid,
    use_equivariance_tiebreak: bool = False,
) -> tuple[Grid, Grid, dict[str, Any]]:
  """Rank all candidate grids across attempts and return `(attempt_1, attempt_2, diagnostics)`."""
  candidates: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}

  def _ensure_entry(grid: Grid) -> dict[str, Any]:
    key = sandbox_verifier.grid_to_canonical_key(grid)
    if key not in candidates:
      candidates[key] = {
          "grid": grid,
          "tier": 2,
          "verified_attempts": set(),
          "primary_attempts": set(),
          "total_attempts": set(),
          "best_train_frac": 0.0,
          "best_cell_acc": 0.0,
          "max_equivariance_bonus": 0.0,
          "min_gen_tokens": 10**9,
          "min_turn": 10**9,
          "min_attempt_idx": 10**9,
      }
    return candidates[key]

  primary_grids_per_attempt: list[tuple[int, Grid]] = []
  for att in attempts:
    if att.train_pass and att.verified_grid is not None:
      entry = _ensure_entry(att.verified_grid)
      entry["tier"] = 0
      entry["verified_attempts"].add(att.attempt_idx)
      entry["primary_attempts"].add(att.attempt_idx)
      entry["total_attempts"].add(att.attempt_idx)
      entry["best_train_frac"] = 1.0
      entry["best_cell_acc"] = 1.0
      entry["max_equivariance_bonus"] = max(
          entry["max_equivariance_bonus"], float(att.equivariance_bonus)
      )
      toks = (
          att.verified_gen_tokens
          if att.verified_gen_tokens is not None
          else att.total_gen_tokens
      )
      trn = att.verified_turn if att.verified_turn is not None else att.total_turns
      entry["min_gen_tokens"] = min(entry["min_gen_tokens"], int(toks))
      entry["min_turn"] = min(entry["min_turn"], int(trn))
      entry["min_attempt_idx"] = min(entry["min_attempt_idx"], att.attempt_idx)
      primary_grids_per_attempt.append((att.attempt_idx, att.verified_grid))

      for sec_grid, sec_turn, sec_toks, sec_eq in att.secondary_verified_grids:
        sec_entry = _ensure_entry(sec_grid)
        sec_entry["tier"] = 0
        sec_entry["verified_attempts"].add(att.attempt_idx)
        sec_entry["total_attempts"].add(att.attempt_idx)
        sec_entry["best_train_frac"] = 1.0
        sec_entry["best_cell_acc"] = 1.0
        sec_entry["max_equivariance_bonus"] = max(
            sec_entry["max_equivariance_bonus"], float(sec_eq)
        )
        sec_entry["min_gen_tokens"] = min(sec_entry["min_gen_tokens"], int(sec_toks))
        sec_entry["min_turn"] = min(sec_entry["min_turn"], int(sec_turn))
        sec_entry["min_attempt_idx"] = min(sec_entry["min_attempt_idx"], att.attempt_idx)

      # Also register any distinct partial/fallback grid from this attempt at Tier 1/2
      # so `attempt_2` is never a wasted duplicate of `attempt_1` if an alternative hypothesis exists.
      ver_key = sandbox_verifier.grid_to_canonical_key(att.verified_grid)
      if (
          att.partial_grid is not None
          and sandbox_verifier.grid_to_canonical_key(att.partial_grid) != ver_key
      ):
        p_entry = _ensure_entry(att.partial_grid)
        p_entry["tier"] = min(p_entry["tier"], 1)
        p_entry["total_attempts"].add(att.attempt_idx)
        p_entry["best_train_frac"] = max(
            p_entry["best_train_frac"], float(att.partial_train_frac)
        )
        p_entry["best_cell_acc"] = max(
            p_entry["best_cell_acc"], float(att.partial_cell_acc)
        )
        p_entry["min_gen_tokens"] = min(
            p_entry["min_gen_tokens"],
            int(att.partial_gen_tokens or att.total_gen_tokens),
        )
        p_entry["min_attempt_idx"] = min(p_entry["min_attempt_idx"], att.attempt_idx)
      if (
          att.fallback_grid is not None
          and sandbox_verifier.grid_to_canonical_key(att.fallback_grid) != ver_key
      ):
        f_entry = _ensure_entry(att.fallback_grid)
        f_entry["total_attempts"].add(att.attempt_idx)
        f_entry["min_gen_tokens"] = min(
            f_entry["min_gen_tokens"],
            int(att.fallback_gen_tokens or att.total_gen_tokens),
        )
        f_entry["min_attempt_idx"] = min(f_entry["min_attempt_idx"], att.attempt_idx)

    elif att.partial_grid is not None:
      entry = _ensure_entry(att.partial_grid)
      entry["tier"] = min(entry["tier"], 1)
      entry["primary_attempts"].add(att.attempt_idx)
      entry["total_attempts"].add(att.attempt_idx)
      entry["best_train_frac"] = max(entry["best_train_frac"], float(att.partial_train_frac))
      entry["best_cell_acc"] = max(entry["best_cell_acc"], float(att.partial_cell_acc))
      entry["min_gen_tokens"] = min(
          entry["min_gen_tokens"], int(att.partial_gen_tokens or att.total_gen_tokens)
      )
      entry["min_attempt_idx"] = min(entry["min_attempt_idx"], att.attempt_idx)
      primary_grids_per_attempt.append((att.attempt_idx, att.partial_grid))

    elif att.fallback_grid is not None:
      entry = _ensure_entry(att.fallback_grid)
      entry["primary_attempts"].add(att.attempt_idx)
      entry["total_attempts"].add(att.attempt_idx)
      entry["min_gen_tokens"] = min(
          entry["min_gen_tokens"], int(att.fallback_gen_tokens or att.total_gen_tokens)
      )
      entry["min_attempt_idx"] = min(entry["min_attempt_idx"], att.attempt_idx)
      primary_grids_per_attempt.append((att.attempt_idx, att.fallback_grid))

  for _, entry in candidates.items():
    sims = [_grid_cell_similarity(entry["grid"], p_grid) for _, p_grid in primary_grids_per_attempt]
    max_sim = max(sims) if sims else 0.0
    threshold = 0.5 * max_sim
    entry["peer_sim_support"] = sum(1 for s in sims if s >= threshold and s > 0.0)

  def _sort_key(item: tuple[tuple[tuple[int, ...], ...], dict[str, Any]]) -> tuple[Any, ...]:
    _, e = item
    tier = int(e["tier"])
    v_votes = len(e["verified_attempts"])
    p_votes = len(e["primary_attempts"])
    t_votes = len(e["total_attempts"])
    peer_sup = int(e["peer_sim_support"])
    eq_bonus = float(e["max_equivariance_bonus"]) if use_equivariance_tiebreak else 0.0
    min_tok = int(e["min_gen_tokens"])
    min_trn = int(e["min_turn"])
    min_att = int(e["min_attempt_idx"])

    if tier == 0:
      return (
          0,
          -v_votes,
          -p_votes,
          -t_votes,
          -peer_sup,
          -eq_bonus,
          min_tok,
          min_trn,
          min_att,
      )
    if tier == 1:
      return (
          1,
          -round(float(e["best_train_frac"]), 6),
          -round(float(e["best_cell_acc"]), 6),
          -p_votes,
          -t_votes,
          -peer_sup,
          min_tok,
          min_att,
          0,
      )
    return (
        2,
        -p_votes,
        -t_votes,
        -peer_sup,
        min_tok,
        min_att,
        0,
        0,
        0,
    )

  ranked = sorted(candidates.items(), key=_sort_key)

  fallback_default = sandbox_verifier.to_grid(test_input)
  if not ranked:
    return fallback_default, fallback_default, {"num_unique_candidates": 0, "ranked": []}

  attempt_1 = ranked[0][1]["grid"]
  attempt_2 = ranked[1][1]["grid"] if len(ranked) >= 2 else attempt_1

  diag_ranked = []
  for _, e in ranked[:6]:
    diag_ranked.append({
        "tier": int(e["tier"]),
        "shape": [len(e["grid"]), len(e["grid"][0]) if e["grid"] else 0],
        "verified_votes": len(e["verified_attempts"]),
        "primary_votes": len(e["primary_attempts"]),
        "total_votes": len(e["total_attempts"]),
        "peer_sim_support": int(e["peer_sim_support"]),
        "best_train_frac": round(float(e["best_train_frac"]), 4),
        "best_cell_acc": round(float(e["best_cell_acc"]), 4),
        "max_equivariance_bonus": round(float(e["max_equivariance_bonus"]), 4),
        "min_gen_tokens": int(e["min_gen_tokens"]),
        "min_turn": int(e["min_turn"]) if e["min_turn"] < 10**8 else None,
        "min_attempt_idx": int(e["min_attempt_idx"]),
    })

  return (
      attempt_1,
      attempt_2,
      {
          "num_unique_candidates": len(ranked),
          "ranked_candidates": diag_ranked,
      },
  )
