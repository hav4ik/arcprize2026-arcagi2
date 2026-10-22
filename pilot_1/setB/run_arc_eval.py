"""ARC-AGI-2 Evaluation & Ablation Runner (Steps .0 to .5) for Gemma-4 & Qwen."""

import base64
import builtins
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import dataclasses
import io
import json
import math
import os
import random
import re
import statistics
import struct
import threading
import time
import traceback
from typing import Any
import zlib

from absl import app
from absl import flags
from google import genai
import numpy as np
import tqdm
from google3.learning.deepmind.boq.data.ditto.client import ditto
from google3.pyglib import gfile

_VARIANT = flags.DEFINE_string(
    "variant",
    "custom",
    "Optional preset name (gemma_dense_2_1..2_5, custom, etc.) or run label.",
)
_RUN_NAME = flags.DEFINE_string(
    "run_name",
    "",
    "Output filename prefix in --out_dir (defaults to --variant if empty).",
)
_MODEL = flags.DEFINE_string(
    "model",
    "gemma-4-31b-it",
    "Model name to evaluate (e.g. gemma-4-31b-it).",
)
_MODEL_OVERRIDE = flags.DEFINE_string(
    "model_override", "", "Optional legacy alias for --model."
)
_DITTO_CONFIG_OVERRIDE = flags.DEFINE_string(
    "ditto_config_override", "", "Optional Ditto config ID override."
)
_WORKERS = flags.DEFINE_integer(
    "workers",
    14,
    "Number of concurrent worker threads (each runs attempts in parallel).",
)
_LIMIT_TASKS = flags.DEFINE_integer(
    "limit_tasks", 0, "If >0, limit to first N tasks (0 = all 120 tasks)."
)
_TASK_IDS = flags.DEFINE_string(
    "task_ids",
    "",
    "Optional comma-separated list of specific task_ids to run.",
)
_TASK_IDS_FILE = flags.DEFINE_string(
    "task_ids_file",
    "",
    "Optional JSON file (CNS or local) containing either a list of task_ids or"
    " a dict with key --task_ids_key.",
)
_TASK_IDS_KEY = flags.DEFINE_string(
    "task_ids_key",
    "hard_25",
    "Key inside --task_ids_file when it is a dict (e.g. hard_25, hard_50,"
    " dev_hard_25, dev, holdout).",
)
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens",
    196608,
    "Single-call maxOutputTokens cap.",
)
_NUM_TOKENS = flags.DEFINE_integer(
    "num_tokens",
    0,
    "Alias for --max_tokens and --total_token_budget if >0.",
)
_TOTAL_TOKEN_BUDGET = flags.DEFINE_integer(
    "total_token_budget",
    196608,
    "Total output token budget across turns/continuations per attempt.",
)
_MAX_TURNS = flags.DEFINE_integer(
    "max_turns",
    0,
    "Max turns per attempt (0 = preset default: 5 for tools, 3 for continue).",
)
_NUM_ATTEMPTS = flags.DEFINE_integer(
    "num_attempts",
    0,
    "Number of parallel attempts per test pair (0 = preset default: 2 or 3).",
)
_THINKING_LEVEL = flags.DEFINE_string(
    "thinking_level",
    "",
    "Override thinking level: HIGH or MINIMAL.",
)
_TEMPERATURE = flags.DEFINE_float(
    "temperature",
    -1.0,
    "If >=0, override sampling temperature for attempts.",
)
_TOP_P = flags.DEFINE_float(
    "top_p",
    -1.0,
    "If >0, override nucleus sampling top_p (default 0.95).",
)
_TOP_K = flags.DEFINE_integer(
    "top_k",
    0,
    "If >0, override top_k sampling (default 20).",
)
_USE_TOOLS = flags.DEFINE_boolean(
    "use_tools",
    None,
    "If set, enable/disable multi-turn Python REPL verification harness.",
)
_RICH_DIFF = flags.DEFINE_boolean(
    "rich_diff",
    None,
    "If set, enable/disable cell-by-cell ASCII grid diffs in Python REPL.",
)
_KEEP_UNVERIFIED_GRID = flags.DEFINE_boolean(
    "keep_unverified_grid",
    True,
    "If True, keep submitted_grid from Python REPL even if train_passed=False.",
)
_USE_PROMPT_TRICKS = flags.DEFINE_boolean(
    "use_prompt_tricks",
    None,
    "If set, enable/disable lean2roles + host task summary + 1-shot prompt.",
)
_HYBRID_PROMPTS = flags.DEFINE_boolean(
    "hybrid_prompts",
    False,
    "If True, use clean tool prompt (2.3) on Attempts 1-2 and prompt tricks"
    " (2.4) on Attempt 3+ for maximum trajectory diversity.",
)
_MULTI_HYPOTHESIS_THINKING = flags.DEFINE_boolean(
    "multi_hypothesis_thinking",
    False,
    "If True, instruct model to formulate and test 2 competing hypotheses"
    " (Hypothesis A vs B) to deepen internal reasoning.",
)
_DSL_HELPERS = flags.DEFINE_boolean(
    "dsl_helpers",
    True,
    "If True, pre-load ARC DSL helper functions (find_objects, crop_bbox,"
    " most_common_color) into the Python REPL.",
)
_TRANSPOSE_AUG = flags.DEFINE_boolean(
    "transpose_aug",
    False,
    "If True, run one parallel attempt on the transposed puzzle (grid.T) and"
    " transpose its predicted output back (ARChitects/MindsAI dihedral trick).",
)
_USE_IMAGES = flags.DEFINE_boolean(
    "use_images",
    False,
    "If True, render the ARC training & test grids as a color PNG image and"
    " attach inlineData alongside the text prompt for multimodal perception.",
)
_USE_TOKEN_PACING = flags.DEFINE_boolean(
    "use_token_pacing",
    None,
    "If set, enable/disable token pacing notes.",
)
_MID_THOUGHT_CONTINUE = flags.DEFINE_boolean(
    "mid_thought_continue",
    None,
    "If set, enable/disable mid-thought continuation recovery.",
)
_FORCED_FINAL_TURN = flags.DEFINE_boolean(
    "forced_final_turn",
    None,
    "If set, enable/disable 4k forced final JSON extraction turn.",
)
_RANK_ATTEMPTS = flags.DEFINE_boolean(
    "rank_attempts",
    None,
    "If set, rank attempts by (train_passed, best_train_score, consensus,"
    " parsed_ok) and deduplicate identical grids.",
)
_OUT_DIR = flags.DEFINE_string(
    "out_dir",
    "/usr/local/google/home/arazdai/.gemini/jetski/brain/9365b34b-25fa-4237-80e2-ed2698f0bdea/scratch/arc_runs",
    "Directory to store JSONL checkpoints and summary reports.",
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    "AQ.Ab8RN6IRn0fpv5yBRybeBUbG0scGVpVCJnI9kZ0JgoqaTvcq2w",
    "Optional Gemini API key for direct AI Studio calls.",
)
_REVERSE = flags.DEFINE_bool(
    "reverse",
    False,
    "If True, process pending work items in reverse order (from end to start).",
)
_SAVE_OUTPUTS = flags.DEFINE_bool(
    "save_outputs",
    True,
    "If True, save full model thought_text, answer_text, tool_trace, and"
    " predicted grids in JSONL.",
)

ARC_EVAL_DIR = "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation"

AI_STUDIO_DITTO_CONFIGS = [
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gemini_evals_ai_studio_1",
    "gdm-cs-all",
    "gdm-cs-gdm",
    "gdm-cs-ai-studio",
]

COLOR_NAMES = {
    0: "0:black",
    1: "1:blue",
    2: "2:red",
    3: "3:green",
    4: "4:yellow",
    5: "5:grey",
    6: "6:magenta",
    7: "7:orange",
    8: "8:azure",
    9: "9:maroon",
}


@dataclasses.dataclass
class VariantSpec:
  """Configuration for a specific evaluation run."""

  name: str
  model_name: str
  backend_family: str  # "gemini" or "qwen_openai"
  thinking_level: str
  max_request_tokens: int
  total_token_budget: int
  soft_token_target: int
  per_turn_max_tokens: int
  temp_att1: float
  temp_att2: float
  top_p: float
  top_k: int
  mid_thought_continue: bool
  forced_final_turn: bool
  use_tools: bool
  use_prompt_tricks: bool
  use_token_pacing: bool
  rank_and_retry3: bool
  concise_visible_cot: bool = False
  preferred_ditto_cfg: str = "gdm-cs-all"
  max_turns: int = 5
  num_attempts: int = 2
  rich_diff: bool = True
  keep_unverified_grid: bool = True
  hybrid_prompts: bool = False
  multi_hypothesis_thinking: bool = False
  dsl_helpers: bool = True
  transpose_aug: bool = False
  use_images: bool = False


def _apply_flag_overrides(spec: VariantSpec) -> VariantSpec:
  """Applies explicit CLI flag overrides on top of any preset VariantSpec."""
  if _RUN_NAME.value:
    spec.name = _RUN_NAME.value
  if _MODEL_OVERRIDE.value:
    spec.model_name = _MODEL_OVERRIDE.value
  elif flags.FLAGS["model"].present or (
      _MODEL.value and spec.name == "custom"
  ):
    spec.model_name = _MODEL.value
  if any(k in spec.model_name.lower() for k in ("qwen", "qwq", "deepseek", "kimi", "glm")):
    spec.backend_family = "qwen_openai"
  if _THINKING_LEVEL.value:
    spec.thinking_level = _THINKING_LEVEL.value.upper()
  if _NUM_TOKENS.value > 0:
    spec.max_request_tokens = _NUM_TOKENS.value
    spec.per_turn_max_tokens = _NUM_TOKENS.value
    if _TOTAL_TOKEN_BUDGET.value > 0 and _TOTAL_TOKEN_BUDGET.value != 196608:
      spec.total_token_budget = max(_TOTAL_TOKEN_BUDGET.value, _NUM_TOKENS.value)
    else:
      spec.total_token_budget = max(spec.total_token_budget, _NUM_TOKENS.value)
  else:
    if _MAX_TOKENS.value > 0:
      spec.max_request_tokens = _MAX_TOKENS.value
      spec.per_turn_max_tokens = _MAX_TOKENS.value
    if _TOTAL_TOKEN_BUDGET.value > 0:
      spec.total_token_budget = _TOTAL_TOKEN_BUDGET.value
  if _TEMPERATURE.value >= 0.0:
    spec.temp_att1 = _TEMPERATURE.value
    spec.temp_att2 = _TEMPERATURE.value
  if _TOP_P.value > 0.0:
    spec.top_p = _TOP_P.value
  if _TOP_K.value > 0:
    spec.top_k = _TOP_K.value
  if _USE_TOOLS.value is not None:
    spec.use_tools = _USE_TOOLS.value
  if _USE_PROMPT_TRICKS.value is not None:
    spec.use_prompt_tricks = _USE_PROMPT_TRICKS.value
  if _RICH_DIFF.value is not None:
    spec.rich_diff = _RICH_DIFF.value
  else:
    spec.rich_diff = True
  spec.keep_unverified_grid = _KEEP_UNVERIFIED_GRID.value
  spec.hybrid_prompts = _HYBRID_PROMPTS.value
  spec.multi_hypothesis_thinking = _MULTI_HYPOTHESIS_THINKING.value
  spec.dsl_helpers = _DSL_HELPERS.value
  spec.transpose_aug = _TRANSPOSE_AUG.value
  spec.use_images = _USE_IMAGES.value
  if _USE_TOKEN_PACING.value is not None:
    spec.use_token_pacing = _USE_TOKEN_PACING.value
  if _MID_THOUGHT_CONTINUE.value is not None:
    spec.mid_thought_continue = _MID_THOUGHT_CONTINUE.value
  if _FORCED_FINAL_TURN.value is not None:
    spec.forced_final_turn = _FORCED_FINAL_TURN.value
  if _RANK_ATTEMPTS.value is not None:
    spec.rank_and_retry3 = _RANK_ATTEMPTS.value
  if _NUM_ATTEMPTS.value > 0:
    spec.num_attempts = _NUM_ATTEMPTS.value
  else:
    spec.num_attempts = 3 if spec.rank_and_retry3 else 2
  if _MAX_TURNS.value > 0:
    spec.max_turns = _MAX_TURNS.value
  else:
    spec.max_turns = 5 if spec.use_tools else (3 if spec.mid_thought_continue else 1)
  return spec


def get_variant_spec(variant: str) -> VariantSpec:
  """Returns the VariantSpec for a named preset or custom flag-driven run."""
  default_req_toks = _NUM_TOKENS.value or _MAX_TOKENS.value or 196608
  default_budget = _NUM_TOKENS.value or _TOTAL_TOKEN_BUDGET.value or 196608

  # If running in pure flag-driven mode (e.g. --variant=custom or any custom run name)
  if not (
      variant.startswith("gemma_moe_")
      or variant.startswith("gemma_dense_")
      or variant.startswith("qwen_")
  ):
    m_name = _MODEL_OVERRIDE.value or _MODEL.value or "gemma-4-31b-it"
    fam = "qwen_openai" if "qwen" in m_name.lower() else "gemini"
    pref_cfg = _DITTO_CONFIG_OVERRIDE.value or "gdm-cs-all"
    spec = VariantSpec(
        name=_RUN_NAME.value or variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=default_budget,
        per_turn_max_tokens=default_req_toks,
        temp_att1=1.0,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=True,
        forced_final_turn=True,
        use_tools=True,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=True,
        preferred_ditto_cfg=pref_cfg,
        max_turns=5,
        num_attempts=3,
        rich_diff=True,
        keep_unverified_grid=True,
    )
    return _apply_flag_overrides(spec)

  # Determine model family prefix for legacy presets
  if variant.startswith("gemma_moe"):
    m_name = _MODEL_OVERRIDE.value or "gemma-4-26b-a4b-it"
    fam = "gemini"
    pref_cfg = _DITTO_CONFIG_OVERRIDE.value or "gemini_evals_ai_studio_1"
    step = variant.removeprefix("gemma_moe_")
  elif variant.startswith("gemma_dense"):
    m_name = _MODEL_OVERRIDE.value or "gemma-4-31b-it"
    fam = "gemini"
    pref_cfg = _DITTO_CONFIG_OVERRIDE.value or "gdm-cs-all"
    step = variant.removeprefix("gemma_dense_")
  elif variant.startswith("qwen"):
    m_name = _MODEL_OVERRIDE.value or (
        _MODEL.value if flags.FLAGS["model"].present else "gdm/qwen3.6-35b-fp8"
    )
    fam = "qwen_openai"
    pref_cfg = _DITTO_CONFIG_OVERRIDE.value or "gemini_evals_together_dedicated"
    step = variant.removeprefix("qwen_")
  else:
    raise ValueError(f"Unknown variant prefix: {variant}")

  # Normalize step identifier (e.g. "1_1" or "2_1" -> "1")
  if step in ("high", "1_0", "2_0"):
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=32768,
        total_token_budget=32768,
        soft_token_target=32768,
        per_turn_max_tokens=32768,
        temp_att1=0.7,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=False,
        forced_final_turn=False,
        use_tools=False,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=False,
        preferred_ditto_cfg=pref_cfg,
    )
    return _apply_flag_overrides(spec)
  if step == "low":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="MINIMAL",
        max_request_tokens=16384,
        total_token_budget=16384,
        soft_token_target=16384,
        per_turn_max_tokens=16384,
        temp_att1=0.7,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=False,
        forced_final_turn=False,
        use_tools=False,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=False,
        concise_visible_cot=True,
        preferred_ditto_cfg=pref_cfg,
    )
    return _apply_flag_overrides(spec)

  substep = step.split("_")[-1]
  if substep == "1":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=default_budget,
        per_turn_max_tokens=default_req_toks,
        temp_att1=0.7,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=False,
        forced_final_turn=False,
        use_tools=False,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=False,
        preferred_ditto_cfg=pref_cfg,
    )
  elif substep == "2":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=default_budget,
        per_turn_max_tokens=default_req_toks,
        temp_att1=1.0,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=True,
        forced_final_turn=True,
        use_tools=False,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=False,
        preferred_ditto_cfg=pref_cfg,
    )
  elif substep == "3":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=default_budget,
        per_turn_max_tokens=default_req_toks,
        temp_att1=1.0,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=True,
        forced_final_turn=True,
        use_tools=True,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=False,
        preferred_ditto_cfg=pref_cfg,
    )
  elif substep == "4":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=48000,
        per_turn_max_tokens=default_req_toks,
        temp_att1=1.0,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=True,
        forced_final_turn=True,
        use_tools=True,
        use_prompt_tricks=True,
        use_token_pacing=True,
        rank_and_retry3=False,
        preferred_ditto_cfg=pref_cfg,
    )
  elif substep == "5":
    spec = VariantSpec(
        name=variant,
        model_name=m_name,
        backend_family=fam,
        thinking_level="HIGH",
        max_request_tokens=default_req_toks,
        total_token_budget=default_budget,
        soft_token_target=default_budget,
        per_turn_max_tokens=default_req_toks,
        temp_att1=1.0,
        temp_att2=1.0,
        top_p=0.95,
        top_k=20,
        mid_thought_continue=True,
        forced_final_turn=True,
        use_tools=True,
        use_prompt_tricks=False,
        use_token_pacing=False,
        rank_and_retry3=True,
        preferred_ditto_cfg=pref_cfg,
        rich_diff=True,
        keep_unverified_grid=True,
        hybrid_prompts=True,
    )
  else:
    raise ValueError(f"Unsupported variant substep: {variant}")
  return _apply_flag_overrides(spec)


def fmt_grid(g: list[list[int]]) -> str:
  return "[\n" + ",\n".join("  " + json.dumps(row) for row in g) + "\n]"


def grid_meta(g: list[list[int]]) -> str:
  h = len(g)
  w = len(g[0]) if h > 0 else 0
  counts: dict[int, int] = {}
  for r in g:
    for v in r:
      counts[v] = counts.get(v, 0) + 1
  hist = ", ".join(
      f"{COLOR_NAMES.get(k, str(k))}={v}"
      for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
  )
  return f"shape={h}x{w} ({h} rows, {w} cols), colors=[{hist}]"


def extract_components(
    g: list[list[int]], bg_color: int = 0
) -> list[dict[str, Any]]:
  """Extracts 4-connected foreground components for the host-computed perception summary."""
  h = len(g)
  w = len(g[0]) if h > 0 else 0
  visited = [[False] * w for _ in range(h)]
  comps = []
  for r in range(h):
    for c in range(w):
      color = g[r][c]
      if color == bg_color or visited[r][c]:
        continue
      q = deque([(r, c)])
      visited[r][c] = True
      cells = []
      while q:
        cr, cc = q.popleft()
        cells.append((cr, cc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
          nr, nc = cr + dr, cc + dc
          if (
              0 <= nr < h
              and 0 <= nc < w
              and not visited[nr][nc]
              and g[nr][nc] == color
          ):
            visited[nr][nc] = True
            q.append((nr, nc))
      rs = [p[0] for p in cells]
      cs = [p[1] for p in cells]
      r0, r1 = min(rs), max(rs)
      c0, c1 = min(cs), max(cs)
      comps.append({
          "color": color,
          "size": len(cells),
          "bbox": (r0, c0, r1, c1),
          "shape": (r1 - r0 + 1, c1 - c0 + 1),
      })
  comps.sort(key=lambda x: (-x["size"], x["bbox"][0], x["bbox"][1]))
  return comps


def build_host_task_summary(
    task: dict[str, Any], test_input: list[list[int]]
) -> str:
  """Pre-computes a perception summary and test-vs-train novelty audit (E017/E020 trick)."""
  lines = ["### Host-Computed Perception & Structural Summary"]
  train_in_shapes = []
  train_in_colors: list[set[int]] = []
  train_comp_counts = []

  for i, pair in enumerate(task["train"]):
    inp, out = pair["input"], pair["output"]
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    train_in_shapes.append((ih, iw))
    in_c = {v for r in inp for v in r}
    out_c = {v for r in out for v in r}
    train_in_colors.append(in_c)
    added_c = sorted(out_c - in_c)
    removed_c = sorted(in_c - out_c)
    # Estimate dominant background color (most frequent in input)
    flat_in = [v for r in inp for v in r]
    bg = Counter(flat_in).most_common(1)[0][0] if flat_in else 0
    in_comps = extract_components(inp, bg_color=bg)
    out_comps = extract_components(out, bg_color=bg)
    train_comp_counts.append(len(in_comps))
    comp_preview = ", ".join(
        f"c{c['color']}(n={c['size']},box={c['bbox'][0]}..{c['bbox'][2]}x{c['bbox'][1]}..{c['bbox'][3]})"
        for c in in_comps[:8]
    )
    if len(in_comps) > 8:
      comp_preview += f", ... (+{len(in_comps) - 8} more)"
    lines.append(
        f"- Train {i+1}: in={ih}x{iw} -> out={oh}x{ow} | bg={bg} |"
        f" in_comps={len(in_comps)} -> out_comps={len(out_comps)} |"
        f" colors_added={added_c} colors_removed={removed_c}\n"
        f"  In objects: [{comp_preview}]"
    )

  th, tw = len(test_input), len(test_input[0])
  flat_test = [v for r in test_input for v in r]
  t_bg = Counter(flat_test).most_common(1)[0][0] if flat_test else 0
  t_comps = extract_components(test_input, bg_color=t_bg)
  t_colors = set(flat_test)
  t_preview = ", ".join(
      f"c{c['color']}(n={c['size']},box={c['bbox'][0]}..{c['bbox'][2]}x{c['bbox'][1]}..{c['bbox'][3]})"
      for c in t_comps[:10]
  )
  if len(t_comps) > 10:
    t_preview += f", ... (+{len(t_comps) - 10} more)"
  lines.append(
      f"- Test Input: shape={th}x{tw} | bg={t_bg} | comps={len(t_comps)} |"
      f" colors={sorted(t_colors)}\n"
      f"  Test objects: [{t_preview}]"
  )

  # Novelty Audit (Test vs Train range)
  warnings = []
  min_h = min(s[0] for s in train_in_shapes)
  max_h = max(s[0] for s in train_in_shapes)
  min_w = min(s[1] for s in train_in_shapes)
  max_w = max(s[1] for s in train_in_shapes)
  if not (min_h <= th <= max_h and min_w <= tw <= max_w):
    warnings.append(
        f"Test input dimensions ({th}x{tw}) lie OUTSIDE training input range"
        f" (rows {min_h}..{max_h}, cols {min_w}..{max_w}). Do NOT hardcode"
        " grid dimensions!"
    )
  all_train_colors = set().union(*train_in_colors) if train_in_colors else set()
  new_test_colors = sorted(t_colors - all_train_colors)
  if new_test_colors:
    warnings.append(
        f"Test input contains NEW color(s) {new_test_colors} never seen in any"
        " training input! Ensure your rule handles arbitrary object/marker"
        " colors."
    )
  if train_comp_counts:
    min_cc, max_cc = min(train_comp_counts), max(train_comp_counts)
    if not (min_cc <= len(t_comps) <= max_cc):
      warnings.append(
          f"Test input component count ({len(t_comps)}) is outside the training"
          f" range [{min_cc}, {max_cc}]."
      )
  if warnings:
    lines.append("### [!] TEST-VS-TRAIN NOVELTY AUDIT WARNINGS:")
    for w_msg in warnings:
      lines.append(f"  * WARNING: {w_msg}")
  return "\n".join(lines)


WORKED_EXAMPLE_1SHOT = """### Brief Worked Example (Pattern & Tool Workflow)
Suppose Train 1..3 show a 10x10 grid with a grey (5) horizontal line dividing top and bottom, where blue (1) shapes on top are reflected across the grey line into red (2) shapes on the bottom.
Ideal workflow:
1. Assign roles: `0` = background, `5` = reflection axis divider, `1` = source template, `2` = reflected output.
2. Write a short ```python block to inspect coordinates and define `def solve(grid): ...`, then call `submit_program(solve)` so the harness tests `solve()` on all training pairs automatically.
"""


def build_prompt_for_spec(
    spec: VariantSpec,
    task: dict[str, Any],
    test_input: list[list[int]],
) -> str:
  """Builds the prompt according to the variant's ablation step (.0 to .5)."""
  ex_blocks = []
  for i, pair in enumerate(task["train"]):
    inp, out = pair["input"], pair["output"]
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])
    same_shape = (
        f"same shape {ih}x{iw}"
        if (ih, iw) == (oh, ow)
        else f"shape changes {ih}x{iw} -> {oh}x{ow}"
    )
    ex_blocks.append(
        f"### Training Example {i + 1} ({same_shape})\n"
        f"INPUT ({grid_meta(inp)}):\n{fmt_grid(inp)}\n\n"
        f"OUTPUT ({grid_meta(out)}):\n{fmt_grid(out)}"
    )
  train_section = "\n\n".join(ex_blocks)

  extra_sections = []
  if spec.use_prompt_tricks:
    extra_sections.append(build_host_task_summary(task, test_input))
    extra_sections.append(WORKED_EXAMPLE_1SHOT)

  role_rules = ""
  if spec.use_prompt_tricks:
    role_rules = (
        "\nCritical Reasoning Rules (lean2roles):\n"
        "- Explicitly assign a functional ROLE to every color/component before"
        " hypothesizing a rule (e.g., background, frame/border, divider,"
        " template/stamp, position marker, color indicator, legend/key).\n"
        "- NEVER dismiss any pixel or cell as 'random noise'. Every single"
        " non-background cell in an ARC puzzle has a deterministic purpose.\n"
        "- Carefully check the TEST-VS-TRAIN NOVELTY AUDIT above: if the test"
        " input has a different grid size, new colors, or more components,"
        " make sure your rule is parameterized rather than hardcoded to"
        " training constants.\n"
    )

  pacing_rules = ""
  if spec.use_token_pacing:
    pacing_rules = (
        f"\nToken Pacing Target:\n"
        f"- Aim to solve and verify the puzzle within the SOFT target of"
        f" {spec.soft_token_target:,} tokens (hard ceiling:"
        f" {spec.total_token_budget:,} tokens).\n"
        "- Keep each thinking turn focused and concise (<8k tokens) before"
        " running Python code to test your hypothesis on the training pairs.\n"
    )

  tournament_rules = ""
  if spec.multi_hypothesis_thinking:
    tournament_rules = (
        "\nDeep Multi-Hypothesis Deliberation (Internal Tournament):\n"
        "- In your thinking, explicitly formulate at least TWO distinct"
        " candidate transformation rules (Hypothesis A vs. Hypothesis B).\n"
        "- Mentally stress-test both hypotheses against edge cases, boundary"
        " conditions, and output dimensions before writing your Python"
        " verification script.\n"
    )

  if spec.use_tools:
    dsl_doc = (
        "  * `find_objects(grid, bg=None, diag=False)`: returns list of"
        " connected-component dicts `{'color': c, 'size': n, 'bbox':"
        " (r0,c0,r1,c1), 'cells': [(r,c),...], 'subgrid': np.ndarray}`\n"
        "  * `crop_bbox(grid, r0, c0, r1, c1)` and `most_common_color(grid)`\n"
        if spec.dsl_helpers
        else ""
    )
    tool_instructions = (
        "\nInteractive Python Environment & Verification Harness:\n"
        "- You have access to a persistent Python REPL across turns. Simply"
        " write a ```python ... ``` code block in your response and stop; the"
        " harness will execute it and return the stdout.\n"
        "- Pre-loaded variables & helpers in your Python session:\n"
        "  * `np` (numpy)\n"
        "  * `train`: list of `(input_grid_list, output_grid_list)` pairs\n"
        "  * `train_inputs`: list of 2D numpy arrays\n"
        "  * `train_outputs`: list of 2D numpy arrays\n"
        "  * `test_input`: 2D numpy array (and `test_input_list` as 2D list)\n"
        f"{dsl_doc}"
        "  * `submit_program(solve)`: tests your `def solve(grid):` function"
        " (which takes a 2D list or numpy array and returns a 2D list/array) on"
        " EVERY training example. If all training examples match 100%, it"
        " automatically computes `solve(test_input)` and records your final"
        " answer! If any training example fails, it shows the exact diff so you"
        " can fix your `solve` function.\n"
        "  * `submit_answer(grid)`: directly submits a 2D integer grid `[[...]]`"
        " for `test_input`.\n"
        "- Always prefer writing `def solve(grid): ...` and calling"
        " `submit_program(solve)` so your rule is 100% verified on all training"
        " examples!\n"
    )
    ending_instruction = (
        "Write a ```python ... ``` block to inspect the grids, test your"
        " transformation hypothesis, and call `submit_program(solve)`."
    )
  else:
    tool_instructions = ""
    if spec.concise_visible_cot:
      ending_instruction = (
          "Keep your written explanation concise (under 8 sentences summarizing"
          " the verified transformation rule and output dimensions), then end"
          " your response with ONLY the exact 2D JSON grid `[[...]]` for the"
          " TEST OUTPUT:"
      )
    else:
      ending_instruction = (
          "Use your internal thinking to thoroughly verify the pattern on all"
          " training examples and construct the exact output grid. In your"
          " visible response, state the verified rule briefly (1-4 sentences)"
          " and end with ONLY the exact 2D JSON grid `[[...]]` for the TEST"
          " OUTPUT:"
      )

  extra_text = ("\n\n".join(extra_sections) + "\n\n") if extra_sections else ""

  prompt = (
      "You are an expert solver for the ARC-AGI-2 (Abstraction and Reasoning"
      " Corpus) competition.\n"
      "Each puzzle consists of 2D grids of integers from 0 to 9 (representing"
      " colors, where 0 is typically background/black).\n"
      "All Training Examples below obey a single, exact, deterministic"
      " transformation rule that maps each INPUT grid to its OUTPUT grid.\n\n"
      "Instructions:\n"
      "1. Carefully inspect all Training Examples below. Pay close attention to"
      " grid dimensions (rows x cols), background color, foreground objects,"
      " connected components, symmetries/rotations/reflections, periodicity or"
      " repeating tiles, divider lines, bounding boxes, color mappings, and"
      " counts.\n"
      "2. Formulate a precise hypothesis for the transformation rule and"
      " explicitly verify that your rule reproduces EVERY Training Example's"
      " OUTPUT from its INPUT without any exceptions.\n"
      "3. Determine the exact output dimensions (number of rows and number of"
      " columns) for the TEST INPUT.\n"
      "4. Apply the verified transformation rule to the TEST INPUT.\n"
      f"{role_rules}{tournament_rules}{pacing_rules}{tool_instructions}\n"
      f"{extra_text}"
      f"{train_section}\n\n"
      f"### Test Puzzle\n"
      f"TEST INPUT ({grid_meta(test_input)}):\n{fmt_grid(test_input)}\n\n"
      f"{ending_instruction}"
  )
  return prompt


ARC_RGB_PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


def render_task_png_b64(
    task: dict[str, Any], test_input: list[list[int]], cell_px: int = 12
) -> str:
  """Renders train pairs and test_input as an RGB PNG (base64) with grid lines."""
  def _draw_grid(g: list[list[int]]) -> np.ndarray:
    h, w = len(g), len(g[0])
    img = np.full((h * cell_px + 1, w * cell_px + 1, 3), 48, dtype=np.uint8)
    for r in range(h):
      for c in range(w):
        rgb = ARC_RGB_PALETTE.get(int(g[r][c]), (0, 0, 0))
        img[
            r * cell_px + 1 : (r + 1) * cell_px,
            c * cell_px + 1 : (c + 1) * cell_px,
        ] = rgb
    return img

  row_canvases = []
  pad = 12
  for pair in task["train"]:
    in_img = _draw_grid(pair["input"])
    out_img = _draw_grid(pair["output"])
    rh = max(in_img.shape[0], out_img.shape[0])
    rw = in_img.shape[1] + pad + out_img.shape[1]
    row_c = np.full((rh, rw, 3), 28, dtype=np.uint8)
    row_c[: in_img.shape[0], : in_img.shape[1]] = in_img
    row_c[: out_img.shape[0], in_img.shape[1] + pad :] = out_img
    row_canvases.append(row_c)

  test_img = _draw_grid(test_input)
  row_canvases.append(test_img)

  total_h = sum(rc.shape[0] for rc in row_canvases) + pad * (
      len(row_canvases) + 1
  )
  total_w = max(rc.shape[1] for rc in row_canvases) + pad * 2
  canvas = np.full((total_h, total_w, 3), 28, dtype=np.uint8)
  y = pad
  for rc in row_canvases:
    canvas[y : y + rc.shape[0], pad : pad + rc.shape[1]] = rc
    y += rc.shape[0] + pad

  height, width, _ = canvas.shape
  raw_rows = b"".join(b"\x00" + canvas[r].tobytes() for r in range(height))

  def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

  ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
  png_bytes = (
      b"\x89PNG\r\n\x1a\n"
      + _png_chunk(b"IHDR", ihdr)
      + _png_chunk(b"IDAT", zlib.compress(raw_rows, 6))
      + _png_chunk(b"IEND", b"")
  )
  return base64.b64encode(png_bytes).decode("ascii")


def transpose_grid(g: list[list[int]]) -> list[list[int]]:
  return [list(row) for row in zip(*g)]


def transpose_task(
    task: dict[str, Any],
    test_input: list[list[int]],
    target: list[list[int]],
) -> tuple[dict[str, Any], list[list[int]], list[list[int]]]:
  t_task = {
      "train": [
          {
              "input": transpose_grid(p["input"]),
              "output": transpose_grid(p["output"]),
          }
          for p in task["train"]
      ],
      "test": task.get("test", []),
  }
  return t_task, transpose_grid(test_input), transpose_grid(target)


def dsl_most_common_color(grid: Any) -> int:
  arr = np.asarray(grid, dtype=int)
  vals, counts = np.unique(arr, return_counts=True)
  return int(vals[int(np.argmax(counts))])


def dsl_crop_bbox(
    grid: Any, r0: int, c0: int, r1: int, c1: int
) -> np.ndarray:
  arr = np.asarray(grid, dtype=int)
  return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def dsl_find_objects(
    grid: Any, bg: int | None = None, diag: bool = False
) -> list[dict[str, Any]]:
  arr = np.asarray(grid, dtype=int)
  if bg is None:
    bg = dsl_most_common_color(arr)
  h, w = arr.shape
  visited = np.zeros((h, w), dtype=bool)
  dirs = (
      [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
      if diag
      else [(-1, 0), (1, 0), (0, -1), (0, 1)]
  )
  objs = []
  for r in range(h):
    for c in range(w):
      color = int(arr[r, c])
      if color == bg or visited[r, c]:
        continue
      q = deque([(r, c)])
      visited[r, c] = True
      cells = []
      while q:
        cr, cc = q.popleft()
        cells.append((cr, cc))
        for dr, dc in dirs:
          nr, nc = cr + dr, cc + dc
          if (
              0 <= nr < h
              and 0 <= nc < w
              and not visited[nr, nc]
              and int(arr[nr, nc]) == color
          ):
            visited[nr, nc] = True
            q.append((nr, nc))
      rs = [p[0] for p in cells]
      cs = [p[1] for p in cells]
      r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
      sub = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
      for pr, pc in cells:
        sub[pr - r0, pc - c0] = color
      objs.append({
          "color": color,
          "size": len(cells),
          "bbox": (r0, c0, r1, c1),
          "shape": (r1 - r0 + 1, c1 - c0 + 1),
          "cells": cells,
          "subgrid": sub,
      })
  objs.sort(key=lambda x: (-x["size"], x["bbox"][0], x["bbox"][1]))
  return objs


def normalize_grid(val: Any) -> list[list[int]] | None:
  """Converts a numpy array or nested list into a validated 2D list[list[int]]."""
  if isinstance(val, np.ndarray):
    val = val.tolist()
  if not isinstance(val, (list, tuple)) or len(val) == 0:
    return None
  out: list[list[int]] = []
  w = None
  for r in val:
    if isinstance(r, np.ndarray):
      r = r.tolist()
    if not isinstance(r, (list, tuple)) or len(r) == 0:
      return None
    if w is None:
      w = len(r)
    elif len(r) != w:
      return None
    row_ints: list[int] = []
    for x in r:
      if isinstance(x, bool):
        return None
      if isinstance(x, (int, np.integer)):
        iv = int(x)
      elif isinstance(x, (float, np.floating)) and float(x).is_integer():
        iv = int(x)
      else:
        return None
      if not (0 <= iv <= 9):
        return None
      row_ints.append(iv)
    out.append(row_ints)
  return out


def parse_grid(text: str) -> list[list[int]] | None:
  """Scans backward from the end of `text` for the last valid 2D JSON grid [[...]]."""
  if not text:
    return None
  cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
  for candidate_text in (cleaned, text):
    for end in range(len(candidate_text) - 1, -1, -1):
      if candidate_text[end] == "]":
        depth = 0
        for start in range(end, -1, -1):
          if candidate_text[start] == "]":
            depth += 1
          elif candidate_text[start] == "[":
            depth -= 1
            if depth == 0:
              try:
                val = json.loads(candidate_text[start : end + 1])
                norm = normalize_grid(val)
                if norm is not None:
                  return norm
              except Exception:
                pass
              break
  return None


def extract_python_blocks(text: str) -> list[str]:
  """Extracts ```python ... ``` blocks from visible assistant text."""
  if not text:
    return []
  cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
  blocks = re.findall(r"```(?:python|py)\s*\n([\s\S]*?)```", cleaned)
  return [b.strip() for b in blocks if b.strip()]


def format_grid_diff(
    expected: list[list[int]], got: list[list[int]], rich: bool = True
) -> str:
  """Formats a diagnostic diff between expected and actual training grids."""
  eh, ew = len(expected), len(expected[0])
  gh, gw = len(got), len(got[0])
  if (eh, ew) != (gh, gw):
    return (
        f"Shape mismatch: expected {eh}x{ew}, but solve() returned {gh}x{gw}."
    )
  mismatches = []
  for r in range(eh):
    for c in range(ew):
      if expected[r][c] != got[r][c]:
        mismatches.append((r, c, expected[r][c], got[r][c]))
  if not mismatches:
    return "Exact match!"
  msg = [
      f"Cell mismatch on {len(mismatches)}/{eh*ew} cells (shape {eh}x{ew}"
      " matches)."
  ]
  if rich:
    sample_m = ", ".join(
        f"({r},{c}): exp={e} got={g}" for r, c, e, g in mismatches[:12]
    )
    if len(mismatches) > 12:
      sample_m += f", ... (+{len(mismatches) - 12} more)"
    msg.append(f"  First mismatches: {sample_m}")
    if eh <= 20 and ew <= 20:
      diff_rows = []
      for r in range(eh):
        row_cells = []
        for c in range(ew):
          if expected[r][c] == got[r][c]:
            row_cells.append(f" {got[r][c]} ")
          else:
            row_cells.append(f"{got[r][c]}!{expected[r][c]}")
        diff_rows.append(" ".join(row_cells))
      msg.append("  Grid diff (got!expected):\n    " + "\n    ".join(diff_rows))
  return "\n".join(msg)


class PythonToolSession:
  """Stateful per-attempt Python REPL + train-pair verification harness."""

  def __init__(
      self,
      task: dict[str, Any],
      test_input: list[list[int]],
      rich_diff: bool = True,
  ):
    self.task = task
    self.test_input = copy.deepcopy(test_input)
    self.rich_diff = rich_diff
    self.train_passed = False
    self.best_train_pairs_ratio = 0.0
    self.best_train_cell_ratio = 0.0
    self.submitted_grid: list[list[int]] | None = None
    self.candidate_test_grid: list[list[int]] | None = None
    self.submitted_code: str = ""
    self._out_buf = io.StringIO()

    train_pairs = [
        (copy.deepcopy(p["input"]), copy.deepcopy(p["output"]))
        for p in task["train"]
    ]
    self.globals: dict[str, Any] = {
        "__builtins__": builtins.__dict__,
        "np": np,
        "math": math,
        "copy": copy,
        "Counter": Counter,
        "deque": deque,
        "train": train_pairs,
        "train_inputs": [np.array(p["input"], dtype=int) for p in task["train"]],
        "train_outputs": [
            np.array(p["output"], dtype=int) for p in task["train"]
        ],
        "test_input": np.array(test_input, dtype=int),
        "test_input_list": copy.deepcopy(test_input),
        "find_objects": dsl_find_objects,
        "crop_bbox": dsl_crop_bbox,
        "most_common_color": dsl_most_common_color,
        "print": self._custom_print,
        "submit_program": self._submit_program,
        "submit_answer": self._submit_answer,
    }

  def _custom_print(self, *args: Any, sep: str = " ", end: str = "\n", **kw: Any) -> None:
    del kw
    s = sep.join(str(a) for a in args) + end
    if self._out_buf.tell() < 12000:
      self._out_buf.write(s[:4000])

  def _submit_answer(self, grid: Any) -> None:
    norm = normalize_grid(grid)
    if norm is None:
      self._custom_print(
          "ERROR in submit_answer(grid): grid must be a valid 2D rectangular"
          " grid of integers 0..9."
      )
      return
    self.submitted_grid = norm
    self.candidate_test_grid = norm
    self._custom_print(
        f"SUBMIT_ANSWER RECORDED: valid {len(norm)}x{len(norm[0])} grid."
    )

  def _verify_solve_fn(self, solve_fn: Any) -> bool:
    if not callable(solve_fn):
      self._custom_print("ERROR: argument to submit_program must be callable.")
      return False
    all_ok = True
    passed_pairs = 0
    cell_ratios: list[float] = []
    results_msg = []
    num_train = max(1, len(self.task["train"]))
    for idx, pair in enumerate(self.task["train"]):
      inp = copy.deepcopy(pair["input"])
      exp = pair["output"]
      try:
        # Try passing numpy array first; if TypeError/AttributeError, pass list
        try:
          raw_out = solve_fn(np.array(inp, dtype=int))
        except Exception:
          raw_out = solve_fn(copy.deepcopy(inp))
        norm_out = normalize_grid(raw_out)
        if norm_out is None:
          all_ok = False
          cell_ratios.append(0.0)
          results_msg.append(
              f"  - Train {idx+1}: FAILED (returned invalid grid: {type(raw_out)})"
          )
        elif norm_out != exp:
          all_ok = False
          eh, ew = len(exp), len(exp[0])
          if (len(norm_out), len(norm_out[0])) == (eh, ew):
            match_c = sum(
                1
                for r in range(eh)
                for c in range(ew)
                if exp[r][c] == norm_out[r][c]
            )
            cell_ratios.append(match_c / max(1, eh * ew))
          else:
            cell_ratios.append(0.0)
          diff_s = format_grid_diff(exp, norm_out, rich=self.rich_diff)
          results_msg.append(f"  - Train {idx+1}: FAILED -> {diff_s}")
        else:
          passed_pairs += 1
          cell_ratios.append(1.0)
          results_msg.append(
              f"  - Train {idx+1}: PASSED ({len(exp)}x{len(exp[0])})"
          )
      except Exception as e:
        all_ok = False
        cell_ratios.append(0.0)
        results_msg.append(f"  - Train {idx+1}: EXCEPTION -> {e}")

    pairs_ratio = passed_pairs / num_train
    mean_cell_ratio = (
        sum(cell_ratios) / len(cell_ratios) if cell_ratios else 0.0
    )

    # Also evaluate on test_input
    test_norm = None
    try:
      try:
        raw_t = solve_fn(np.array(self.test_input, dtype=int))
      except Exception:
        raw_t = solve_fn(copy.deepcopy(self.test_input))
      test_norm = normalize_grid(raw_t)
    except Exception as e:
      results_msg.append(f"  - Test Input execution raised exception: {e}")

    if test_norm is not None:
      # Only overwrite candidate_test_grid if this solve_fn scored >= previous best on train!
      if (pairs_ratio, mean_cell_ratio) >= (
          self.best_train_pairs_ratio,
          self.best_train_cell_ratio,
      ) or self.candidate_test_grid is None:
        self.best_train_pairs_ratio = pairs_ratio
        self.best_train_cell_ratio = mean_cell_ratio
        self.candidate_test_grid = test_norm

    if all_ok and test_norm is not None:
      self.train_passed = True
      self.best_train_pairs_ratio = 1.0
      self.best_train_cell_ratio = 1.0
      self.submitted_grid = test_norm
      self._custom_print(
          "=== TRAIN CHECK PASSED (ALL TRAINING EXAMPLES MATCH 100%) ===\n"
          + "\n".join(results_msg)
          + f"\nTest Output successfully generated: shape"
          f" {len(test_norm)}x{len(test_norm[0])}."
      )
      return True
    else:
      self._custom_print(
          f"=== TRAIN CHECK FAILED ({passed_pairs}/{num_train} pairs passed,"
          f" cell accuracy={mean_cell_ratio*100:.1f}%) ===\n"
          + "\n".join(results_msg)
          + "\nPlease inspect the failure diff above, fix `solve(grid)`, and"
          " call `submit_program(solve)` again."
      )
      return False

  def _submit_program(self, solve_fn: Any) -> bool:
    return self._verify_solve_fn(solve_fn)

  def run_code(self, code: str, timeout_s: float = 6.0) -> str:
    """Executes Python code in the persistent session with a timeout."""
    self._out_buf = io.StringIO()
    prev_submitted = self.submitted_grid
    err_holder: list[str] = []

    def _target() -> None:
      try:
        exec(code, self.globals)  # pylint: disable=exec-used
        # If the code defined `def solve` but forgot to call `submit_program(solve)`
        if (
            self.submitted_grid is prev_submitted
            and "def solve" in code
            and callable(self.globals.get("solve"))
        ):
          self._verify_solve_fn(self.globals["solve"])
        # Check if `answer` variable was set in globals
        if "answer" in self.globals and self.candidate_test_grid is None:
          norm_ans = normalize_grid(self.globals["answer"])
          if norm_ans is not None:
            self.candidate_test_grid = norm_ans
      except Exception:
        err_holder.append(traceback.format_exc(limit=4))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
      return f"TIMEOUT: Code execution exceeded {timeout_s}s limit."
    out_str = self._out_buf.getvalue()
    if err_holder:
      out_str += "\n" + "\n".join(err_holder)
    if self.train_passed and "def solve" in code:
      self.submitted_code = code
    return out_str.strip() or "(Executed cleanly with no stdout)"


class ModelSampler:
  """Unified sampler for Gemma-4 (OnePlatform/Ditto) and Qwen (Ditto/Vertex/OpenAI)."""

  def __init__(self, spec: VariantSpec, api_key: str = ""):
    self.spec = spec
    self.model_name = spec.model_name.removeprefix("models/")
    self.api_key = api_key
    cfgs = list(AI_STUDIO_DITTO_CONFIGS)
    if spec.preferred_ditto_cfg and spec.preferred_ditto_cfg in cfgs:
      cfgs.insert(0, spec.preferred_ditto_cfg)
    self.ditto_configs = cfgs
    self._tls = threading.local()
    self._lock = threading.Lock()
    self._counter = 0
    # Allow up to 64 simultaneous in-flight API requests across workers/attempts
    self._api_sem = threading.Semaphore(64)

  def _get_ditto_client(self, cfg: str) -> ditto.DittoClient:
    clients = getattr(self._tls, "ditto_clients", None)
    if clients is None:
      clients = {}
      self._tls.ditto_clients = clients
    if cfg not in clients:
      clients[cfg] = ditto.DittoClient(cfg, use_cache=False, rotate_key=True)
    return clients[cfg]

  def generate_contents(
      self,
      contents: list[dict[str, Any]],
      temperature: float,
      max_output_tokens: int,
      thinking_level: str,
  ) -> dict[str, Any]:
    """Generates a single turn from multi-turn `contents`."""
    if self.spec.backend_family == "qwen_openai":
      return self._generate_qwen_openai(
          contents, temperature, max_output_tokens
      )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{self.model_name}:generateContent"
    )
    cur_max_tokens = max_output_tokens

    with self._lock:
      idx = self._counter
      self._counter += 1

    backends: list[tuple[str, str]] = []
    for i in range(len(self.ditto_configs)):
      cfg = self.ditto_configs[(idx + i) % len(self.ditto_configs)]
      backends.append(("ditto", cfg))
    if self.api_key:
      backends.insert(2, ("direct", self.api_key))

    last_err = ""
    for attempt in range(5):
      # If multimodal inlineData fails repeatedly with 500/503, fall back to text-only parts
      active_contents = contents
      if attempt >= 2 and contents and "parts" in contents[0]:
        text_only_parts = [p for p in contents[0]["parts"] if "text" in p]
        if text_only_parts and len(text_only_parts) < len(contents[0]["parts"]):
          active_contents = [
              {"role": contents[0].get("role", "user"), "parts": text_only_parts}
          ] + contents[1:]

      body = {
          "contents": active_contents,
          "generationConfig": {
              "temperature": temperature,
              "topP": self.spec.top_p,
              "topK": self.spec.top_k,
              "maxOutputTokens": cur_max_tokens,
              "thinkingConfig": {
                  "thinkingLevel": thinking_level,
                  "includeThoughts": True,
              },
          },
      }
      body_json = json.dumps(body)
      btype, bval = backends[attempt % len(backends)]
      t0 = time.time()
      try:
        with self._api_sem:
          if btype == "ditto":
            dc = self._get_ditto_client(bval)
            resp = dc.post(
                url=url,
                body=body_json,
                response_type=ditto.DittoResponseType.JSON,
                deadline_secs=2400,
            )
            if isinstance(resp, dict) and "candidates" in resp:
              return self._parse_gemini_json(
                  resp, time.time() - t0, f"ditto:{bval}"
              )
            last_err = f"ditto:{bval} unexpected resp: {str(resp)[:200]}"
          else:
            gclient = genai.Client(api_key=bval)
            resp = gclient.models.generate_content(
                model=f"models/{self.model_name}",
                contents=active_contents,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=self.spec.top_p,
                    top_k=self.spec.top_k,
                    max_output_tokens=cur_max_tokens,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_level=thinking_level,
                        include_thoughts=True,
                    ),
                ),
            )
            thought_parts = []
            text_parts = []
            finish_reason = ""
            if resp.candidates:
              cand = resp.candidates[0]
              finish_reason = str(cand.finish_reason)
              if cand.content and cand.content.parts:
                for p in cand.content.parts:
                  if getattr(p, "thought", False):
                    if p.text:
                      thought_parts.append(p.text)
                  elif p.text:
                    text_parts.append(p.text)
            usage = resp.usage_metadata
            prompt_toks = getattr(usage, "prompt_token_count", 0) or 0
            cand_toks = getattr(usage, "candidates_token_count", 0) or 0
            think_toks = getattr(usage, "thoughts_token_count", 0) or 0
            tot_toks = getattr(usage, "total_token_count", 0) or (
                prompt_toks + cand_toks + think_toks
            )
            return {
                "thought_text": "".join(thought_parts),
                "text": "".join(text_parts),
                "prompt_tokens": prompt_toks,
                "answer_tokens": cand_toks,
                "thinking_tokens": think_toks,
                "total_output_tokens": cand_toks + think_toks,
                "total_tokens": tot_toks,
                "finish_reason": finish_reason,
                "latency_s": round(time.time() - t0, 2),
                "backend": "direct_api_key",
                "error": "",
            }
      except Exception as e:
        last_err = f"{btype}:{bval}: {e}"
      sleep_s = min(
          40.0, 2.0 * (1.3 ** min(attempt, 10)) + random.uniform(0.5, 3.5)
      )
      time.sleep(sleep_s)

    return {
        "thought_text": "",
        "text": "",
        "prompt_tokens": 0,
        "answer_tokens": 0,
        "thinking_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "finish_reason": "ERROR",
        "latency_s": 0.0,
        "backend": "failed",
        "error": last_err[:300],
    }

  def _parse_gemini_json(
      self, data: dict[str, Any], latency_s: float, backend: str
  ) -> dict[str, Any]:
    cands = data.get("candidates", [])
    thought_parts = []
    text_parts = []
    if cands:
      parts = cands[0].get("content", {}).get("parts", [])
      for p in parts:
        if p.get("thought", False) and "text" in p:
          thought_parts.append(p["text"])
        elif not p.get("thought", False) and "text" in p:
          text_parts.append(p["text"])
      finish_reason = cands[0].get("finishReason", "")
    else:
      finish_reason = "NO_CANDIDATES"
    usage = data.get("usageMetadata", {})
    prompt_toks = int(usage.get("promptTokenCount", 0) or 0)
    cand_toks = int(usage.get("candidatesTokenCount", 0) or 0)
    think_toks = int(usage.get("thoughtsTokenCount", 0) or 0)
    tot_toks = int(
        usage.get("totalTokenCount", 0)
        or (prompt_toks + cand_toks + think_toks)
    )
    return {
        "thought_text": "".join(thought_parts),
        "text": "".join(text_parts),
        "prompt_tokens": prompt_toks,
        "answer_tokens": cand_toks,
        "thinking_tokens": think_toks,
        "total_output_tokens": cand_toks + think_toks,
        "total_tokens": tot_toks,
        "finish_reason": finish_reason,
        "latency_s": round(latency_s, 2),
        "backend": backend,
        "error": "",
    }

  def _generate_qwen_openai(
      self,
      contents: list[dict[str, Any]],
      temperature: float,
      max_output_tokens: int,
  ) -> dict[str, Any]:
    """Calls Qwen via Ditto (Vertex Dedicated Endpoint / OpenRouter / Together)."""
    messages = []
    for c in contents:
      role = "assistant" if c.get("role") == "model" else "user"
      text = "".join(p.get("text", "") for p in c.get("parts", []))
      messages.append({"role": role, "content": text})

    m_lower = self.model_name.lower()
    if m_lower in (
        "qwen3.6-35b",
        "qwen3.6-35b-a3b",
        "qwen/qwen3.6-35b-a3b",
        "gdm/qwen3.6-35b-fp8",
    ):
      endpoints = [(
          "gemini_evals_together_dedicated",
          "https://api.together.xyz/v1/chat/completions",
          "gdm/qwen3.6-35b-fp8",
      )]
    elif m_lower in (
        "qwen3.8-2.4t",
        "qwen3.8-max",
        "qwen3.8-2.4t-a95b",
        "qwen/qwen3.8-2.4t-a95b",
    ):
      endpoints = [(
          "gemini_evals_together_tier1",
          "https://api.together.xyz/v1/chat/completions",
          "Qwen/Qwen3.8-2.4T-A95B",
      )]
    elif m_lower in (
        "qwen3.5-397b",
        "qwen3.5-397b-a17b",
        "qwen/qwen3.5-397b-a17b",
        "drogosand/qwen/qwen3.5-397b-a17b-fp8-b9f87d60",
    ):
      endpoints = [(
          "gemini_evals_together_tier1",
          "https://api.together.xyz/v1/chat/completions",
          "Qwen/Qwen3.5-397B-A17B",
      )]
    elif m_lower in ("qwen/qwen3.8-27b", "qwen3.8-27b", "drogosand/qwen3.8-27b"):
      endpoints = [(
          "gemini_evals_together_dedicated",
          "https://api.together.xyz/v1/chat/completions",
          "drogosand/qwen3.8-27b",
      )]
    else:
      endpoints = [
          (
              "gemini_evals_together_tier1",
              "https://api.together.xyz/v1/chat/completions",
              self.model_name,
          ),
          (
              "gemini_evals_together_dedicated",
              "https://api.together.xyz/v1/chat/completions",
              self.model_name,
          ),
      ]
    errs = []
    for cfg, url, m_id in endpoints:
      cur_max_toks = min(max_output_tokens, 65536)
      for retry_idx in range(4):
        t0 = time.time()
        try:
          with self._api_sem:
            dc = self._get_ditto_client(cfg)
            body = {
                "model": m_id,
                "temperature": temperature,
                "top_p": self.spec.top_p,
                "max_tokens": cur_max_toks,
                "messages": messages,
            }
            resp = dc.post(
                url=url,
                body=json.dumps(body),
                response_type=ditto.DittoResponseType.JSON,
                deadline_secs=2400,
            )
          if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
            ch = resp["choices"][0]
            msg = ch.get("message", {})
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            if not reasoning and "<think>" in content:
              m = re.search(r"<think>([\s\S]*?)</think>", content)
              if m:
                reasoning = m.group(1)
                content = re.sub(
                    r"<think>[\s\S]*?</think>", "", content
                ).strip()
            usage = resp.get("usage", {})
            p_toks = int(usage.get("prompt_tokens", 0) or 0)
            c_toks = int(usage.get("completion_tokens", 0) or 0)
            r_toks = int(usage.get("reasoning_tokens", 0) or 0)
            if r_toks == 0 and reasoning and c_toks > 0:
              tot_chars = max(1, len(reasoning) + len(content))
              r_toks = int(round(c_toks * (len(reasoning) / tot_chars)))
            return {
                "thought_text": reasoning,
                "text": content,
                "prompt_tokens": p_toks,
                "answer_tokens": max(0, c_toks - r_toks),
                "thinking_tokens": r_toks,
                "total_output_tokens": c_toks,
                "total_tokens": p_toks + c_toks,
                "finish_reason": str(ch.get("finish_reason", "STOP")).upper(),
                "latency_s": round(time.time() - t0, 2),
                "backend": f"ditto:{cfg}:{m_id}",
                "error": "",
            }
          errs.append(f"{cfg}({m_id}): {str(resp)[:200]}")
          break
        except Exception as e:
          e_str = str(e)
          if "maximum context length" in e_str.lower():
            m_ctx = re.search(
                r"maximum context length is (\d+).*?(\d+) in the messages",
                e_str,
                re.IGNORECASE | re.DOTALL,
            )
            if m_ctx:
              avail = int(m_ctx.group(1)) - int(m_ctx.group(2)) - 1024
              if avail >= 256 and avail < cur_max_toks:
                cur_max_toks = avail
                continue
          if any(
              code in e_str for code in ("429", "500", "502", "503", "504")
          ) and retry_idx < 3:
            time.sleep((2.5 ** retry_idx) + random.uniform(0.5, 2.0))
            continue
          errs.append(f"{cfg}({m_id}): {e_str}")
          break
    err_str = " | ".join(errs)
    tqdm.tqdm.write(f"[Qwen error] {err_str[:400]}")
    return {
        "thought_text": "",
        "text": "",
        "prompt_tokens": 0,
        "answer_tokens": 0,
        "thinking_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "finish_reason": "ERROR",
        "latency_s": 0.0,
        "backend": "failed",
        "error": err_str[:350],
    }


def run_attempt_with_spec(
    spec: VariantSpec,
    sampler: ModelSampler,
    task: dict[str, Any],
    test_input: list[list[int]],
    target: list[list[int]],
    temperature: float,
) -> tuple[list[list[int]] | None, dict[str, Any]]:
  """Executes a single attempt under the given VariantSpec (.0 through .5)."""
  prompt = build_prompt_for_spec(spec, task, test_input)
  user_parts: list[dict[str, Any]] = []
  if spec.use_images:
    try:
      png_b64 = render_task_png_b64(task, test_input)
      user_parts.append(
          {"inlineData": {"mimeType": "image/png", "data": png_b64}}
      )
    except Exception:
      pass
  user_parts.append({"text": prompt})
  contents: list[dict[str, Any]] = [{"role": "user", "parts": user_parts}]

  total_prompt_tokens = 0
  total_thinking_tokens = 0
  total_answer_tokens = 0
  total_output_tokens = 0
  total_latency_s = 0.0
  last_finish_reason = ""
  last_backend = ""
  last_error = ""

  all_thoughts: list[str] = []
  all_answers: list[str] = []
  tool_trace: list[dict[str, Any]] = []

  tool_session = (
      PythonToolSession(task, test_input, rich_diff=spec.rich_diff)
      if spec.use_tools
      else None
  )

  pred: list[list[int]] | None = None
  allow_multi_turn = (
      spec.use_tools
      or spec.mid_thought_continue
      or (spec.total_token_budget > spec.max_request_tokens)
  )
  max_turns = (
      spec.max_turns
      if spec.max_turns > 0
      else (5 if spec.use_tools else (3 if allow_multi_turn else 1))
  )
  next_turn_minimal_thinking = False

  for turn_idx in range(max_turns):
    remaining_budget = spec.total_token_budget - total_output_tokens
    if remaining_budget <= 1024:
      break

    if next_turn_minimal_thinking:
      turn_thinking_level = "MINIMAL"
      turn_cap = min(8192, remaining_budget)
      next_turn_minimal_thinking = False
    else:
      turn_thinking_level = spec.thinking_level
      turn_cap = min(
          spec.per_turn_max_tokens,
          spec.max_request_tokens,
          remaining_budget,
      )

    res = sampler.generate_contents(
        contents=contents,
        temperature=temperature,
        max_output_tokens=turn_cap,
        thinking_level=turn_thinking_level,
    )

    total_prompt_tokens += res["prompt_tokens"]
    total_thinking_tokens += res["thinking_tokens"]
    total_answer_tokens += res["answer_tokens"]
    total_output_tokens += res["total_output_tokens"]
    total_latency_s += res["latency_s"]
    last_finish_reason = res["finish_reason"]
    last_backend = res["backend"]
    if res["error"]:
      last_error = res["error"]

    if res["thought_text"]:
      all_thoughts.append(
          f"--- [Turn {turn_idx+1} Thought] ---\n{res['thought_text']}"
      )
    if res["text"]:
      all_answers.append(
          f"--- [Turn {turn_idx+1} Answer] ---\n{res['text']}"
      )

    if res["error"] and not res["text"] and not res["thought_text"]:
      break

    # Branch A: Tool-use harness (.3, .4, .5)
    if spec.use_tools and tool_session is not None:
      py_blocks = extract_python_blocks(res["text"])
      if py_blocks:
        combined_code = "\n\n".join(py_blocks)
        exec_out = tool_session.run_code(combined_code)
        tool_trace.append({
            "turn": turn_idx + 1,
            "code": combined_code,
            "output": exec_out,
            "train_passed": tool_session.train_passed,
            "cumulative_output_tokens": total_output_tokens,
        })
        if (
            tool_session.train_passed
            and tool_session.submitted_grid is not None
        ):
          pred = tool_session.submitted_grid
          break
        if (
            tool_session.submitted_grid is not None
            or tool_session.candidate_test_grid is not None
        ) and (spec.keep_unverified_grid or not spec.use_prompt_tricks):
          pred = (
              tool_session.submitted_grid
              if tool_session.submitted_grid is not None
              else tool_session.candidate_test_grid
          )

        budget_note = ""
        if spec.use_token_pacing:
          budget_note = (
              f"\n[Token Budget: {total_output_tokens:,} /"
              f" {spec.total_token_budget:,} used (soft target:"
              f" {spec.soft_token_target:,})]"
          )
        model_turn_text = res["text"]
        if not model_turn_text and res["thought_text"]:
          model_turn_text = f"Thought summary:\n{res['thought_text'][-4000:]}"
        contents.append({"role": "model", "parts": [{"text": model_turn_text}]})
        contents.append({
            "role": "user",
            "parts": [{
                "text": (
                    f"Python execution output:\n```\n{exec_out}\n```{budget_note}\n"
                    "Continue refining your `solve(grid)` function and call"
                    " `submit_program(solve)`, or output the final 2D JSON grid."
                )
            }],
        })
        continue
      else:
        direct_grid = parse_grid(res["text"])
        if direct_grid is not None:
          pred = direct_grid
          break
        if (
            allow_multi_turn
            and total_output_tokens < spec.total_token_budget - 4096
        ):
          # We already spent a full HIGH-thinking turn; switch next turn to MINIMAL
          # so it immediately emits the Python code block from its accumulated thoughts.
          next_turn_minimal_thinking = True
          partial = res["thought_text"][-8000:] if res["thought_text"] else ""
          budget_note = (
              f"\n[Token Budget: {total_output_tokens:,} /"
              f" {spec.total_token_budget:,} used]"
              if spec.use_token_pacing
              else ""
          )
          contents.append({
              "role": "model",
              "parts": [{"text": f"Partial analysis:\n{partial}\n{res['text']}"}],
          })
          contents.append({
              "role": "user",
              "parts": [{
                  "text": (
                      f"Your previous turn stopped before executing Python code.{budget_note} "
                      "Based on your analysis above, write a ```python ... ```"
                      " block right now defining `solve(grid)` and calling"
                      " `submit_program(solve)`."
                  )
              }],
          })
          continue
        break

    # Branch B: No-tools mode (.0, .1, .2)
    pred = parse_grid(res["text"])
    if pred is not None:
      break

    if not allow_multi_turn:
      if "MAX_TOKENS" in res.get("finish_reason", "") or res.get("error"):
        res2 = sampler.generate_contents(
            contents=contents,
            temperature=0.85,
            max_output_tokens=spec.max_request_tokens,
            thinking_level=spec.thinking_level,
        )
        total_prompt_tokens += res2["prompt_tokens"]
        total_thinking_tokens += res2["thinking_tokens"]
        total_answer_tokens += res2["answer_tokens"]
        total_output_tokens += res2["total_output_tokens"]
        total_latency_s += res2["latency_s"]
        if res2["thought_text"]:
          all_thoughts.append(f"--- [Retry Thought] ---\n{res2['thought_text']}")
        if res2["text"]:
          all_answers.append(f"--- [Retry Answer] ---\n{res2['text']}")
        pred2 = parse_grid(res2["text"])
        if pred2 is not None:
          pred = pred2
          last_finish_reason = res2["finish_reason"]
      break

    # Multi-turn continuation (.1, .2) across the 96k/128k/192k total budget
    if total_output_tokens < spec.total_token_budget - 4096:
      if turn_idx >= 1:
        next_turn_minimal_thinking = True
      partial_thought = (
          res["thought_text"][-8000:] if res["thought_text"] else ""
      )
      partial_ans = res["text"] or ""
      contents.append({
          "role": "model",
          "parts": [{
              "text": (
                  f"Reasoning so far:\n{partial_thought}\n{partial_ans}".strip()
              )
          }],
      })
      contents.append({
          "role": "user",
          "parts": [{
              "text": (
                  "Continue your analysis from where you left off, verify the"
                  " transformation rule on all training examples, and end your"
                  " response with ONLY the exact 2D JSON grid `[[...]]` for the"
                  " TEST OUTPUT:"
              )
          }],
      })

  # Session fallback for tool mode if `pred` is still None
  if pred is None and tool_session is not None:
    if tool_session.submitted_grid is not None:
      pred = tool_session.submitted_grid
    elif tool_session.candidate_test_grid is not None:
      pred = tool_session.candidate_test_grid

  # 4k Forced Final Answer turn (.2, .3, .4, .5) if still no grid
  if pred is None and spec.forced_final_turn and not last_error:
    combined_notes = "\n".join(all_thoughts + all_answers)[-10000:]
    forced_contents = [
        {"role": "user", "parts": [{"text": prompt}]},
        {
            "role": "model",
            "parts": [{"text": f"Analysis summary:\n{combined_notes}"}],
        },
        {
            "role": "user",
            "parts": [{
                "text": (
                    "Your thinking budget is complete. Based on your analysis"
                    " above, output ONLY the final 2D JSON integer grid"
                    " `[[...]]` for the TEST OUTPUT right now with no extra"
                    " prose:"
                )
            }],
        },
    ]
    f_res = sampler.generate_contents(
        contents=forced_contents,
        temperature=0.2,
        max_output_tokens=4096,
        thinking_level="MINIMAL",
    )
    total_prompt_tokens += f_res["prompt_tokens"]
    total_thinking_tokens += f_res["thinking_tokens"]
    total_answer_tokens += f_res["answer_tokens"]
    total_output_tokens += f_res["total_output_tokens"]
    total_latency_s += f_res["latency_s"]
    if f_res["text"]:
      all_answers.append(f"--- [Forced 4k Final] ---\n{f_res['text']}")
      pred = parse_grid(f_res["text"])

  # Last-chance check if grid appeared inside thought_text
  if pred is None and all_thoughts:
    pred = parse_grid("\n".join(all_thoughts))

  if pred is not None or total_output_tokens > 0:
    last_error = ""

  exact_match = pred == target
  train_passed = tool_session.train_passed if tool_session is not None else False
  best_train_pairs_ratio = (
      round(tool_session.best_train_pairs_ratio, 4)
      if tool_session is not None
      else 0.0
  )
  best_train_cell_ratio = (
      round(tool_session.best_train_cell_ratio, 4)
      if tool_session is not None
      else 0.0
  )
  info: dict[str, Any] = {
      "temperature": temperature,
      "parsed_ok": pred is not None,
      "train_passed": train_passed,
      "best_train_pairs_ratio": best_train_pairs_ratio,
      "best_train_cell_ratio": best_train_cell_ratio,
      "exact_match": exact_match,
      "pred_shape": [len(pred), len(pred[0])] if pred else None,
      "target_shape": [len(target), len(target[0])],
      "prompt_tokens": total_prompt_tokens,
      "thinking_tokens": total_thinking_tokens,
      "answer_tokens": total_answer_tokens,
      "total_output_tokens": total_output_tokens,
      "num_turns": len(all_thoughts) + (1 if not all_thoughts and all_answers else 0),
      "finish_reason": last_finish_reason,
      "latency_s": round(total_latency_s, 2),
      "backend": last_backend,
      "error": last_error,
  }
  if _SAVE_OUTPUTS.value:
    info["pred_grid"] = pred
    info["submitted_code"] = (
        tool_session.submitted_code if tool_session is not None else ""
    )
    info["tool_trace"] = tool_trace
    info["thought_text"] = "\n".join(all_thoughts)
    info["answer_text"] = "\n".join(all_answers)
  return pred, info


def select_and_rank_attempts(
    attempts: list[tuple[list[list[int]] | None, dict[str, Any]]],
    rank_enabled: bool,
) -> tuple[list[list[int]] | None, list[list[int]] | None, list[dict[str, Any]]]:
  """Selects top 2 predictions, ranking by train_passed, partial train score, and consensus."""
  infos = [a[1] for a in attempts]
  if not rank_enabled:
    p1 = attempts[0][0] if len(attempts) > 0 else None
    p2 = attempts[1][0] if len(attempts) > 1 else None
    return p1, p2, infos

  # Count consensus frequencies among valid grids
  grid_counts: dict[str, int] = {}
  grid_train_pass: dict[str, bool] = {}
  for pred, info in attempts:
    if pred is not None:
      key = json.dumps(pred)
      grid_counts[key] = grid_counts.get(key, 0) + 1
      if info.get("train_passed", False):
        grid_train_pass[key] = True

  # Sort attempts: (train_passed, best_train_pairs_ratio, consensus_count, best_train_cell_ratio, parsed_ok, -attempt_idx)
  indexed = list(enumerate(attempts))
  indexed.sort(
      key=lambda item: (
          bool(item[1][1].get("train_passed", False)),
          float(item[1][1].get("best_train_pairs_ratio", 0.0)),
          grid_counts.get(json.dumps(item[1][0]), 0)
          if item[1][0] is not None
          else 0,
          round(float(item[1][1].get("best_train_cell_ratio", 0.0)), 3),
          item[1][0] is not None,
          -item[0],
      ),
      reverse=True,
  )

  ranked_infos = [item[1][1] for item in indexed]
  distinct_preds: list[list[list[int]]] = []
  seen_keys: set[str] = set()
  for _, (pred, _) in indexed:
    if pred is not None:
      k = json.dumps(pred)
      if k not in seen_keys:
        seen_keys.add(k)
        distinct_preds.append(pred)

  pred1 = distinct_preds[0] if len(distinct_preds) >= 1 else None
  pred2 = (
      distinct_preds[1]
      if len(distinct_preds) >= 2
      else (distinct_preds[0] if len(distinct_preds) == 1 else None)
  )
  return pred1, pred2, ranked_infos


def load_all_tasks(
    limit_tasks: int = 0, task_ids_csv: str = ""
) -> list[tuple[str, dict[str, Any]]]:
  files = sorted(f for f in gfile.ListDir(ARC_EVAL_DIR) if f.endswith(".json"))
  wanted: set[str] = set()
  if _TASK_IDS_FILE.value:
    with gfile.GFile(_TASK_IDS_FILE.value, "r") as tf:
      obj = json.loads(tf.read())
    wanted.update(obj[_TASK_IDS_KEY.value] if isinstance(obj, dict) else obj)
  if task_ids_csv:
    wanted.update(t.strip() for t in task_ids_csv.split(",") if t.strip())
  if wanted:
    files = [f for f in files if f[:-5] in wanted]
  if limit_tasks > 0:
    files = files[:limit_tasks]
  tasks = []
  for fname in files:
    task_id = fname[:-5]
    fpath = os.path.join(ARC_EVAL_DIR, fname)
    with gfile.GFile(fpath, "r") as f:
      tasks.append((task_id, json.loads(f.read())))
  return tasks


def percentile(sorted_vals: list[float], p: float) -> float:
  if not sorted_vals:
    return 0.0
  k = (len(sorted_vals) - 1) * p
  f = math.floor(k)
  c = math.ceil(k)
  if f == c:
    return float(sorted_vals[int(k)])
  return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def compute_summary(
    variant: str,
    model_label: str,
    records: list[dict[str, Any]],
    total_expected_pairs: int,
    total_expected_tasks: int,
    elapsed_s: float,
) -> dict[str, Any]:
  """Computes Kaggle (pair-weighted) and ARC Prize (task-weighted) pass@1 & pass@2 and token stats."""
  by_task: dict[str, list[dict[str, Any]]] = {}
  for r in records:
    by_task.setdefault(r["task_id"], []).append(r)

  n_pairs = len(records)
  n_tasks = len(by_task)

  pair_pass1 = sum(1 for r in records if r["pass1"])
  pair_pass2 = sum(1 for r in records if r["pass2"])

  task_pass1_score = 0.0
  task_pass2_score = 0.0
  solved_tasks_pass1 = []
  solved_tasks_pass2 = []
  for tid, trecs in sorted(by_task.items()):
    tp1 = sum(1.0 for r in trecs if r["pass1"]) / len(trecs)
    tp2 = sum(1.0 for r in trecs if r["pass2"]) / len(trecs)
    task_pass1_score += tp1
    task_pass2_score += tp2
    if tp1 > 0:
      solved_tasks_pass1.append(f"{tid} ({tp1:.2f})" if tp1 < 1.0 else tid)
    if tp2 > 0:
      solved_tasks_pass2.append(f"{tid} ({tp2:.2f})" if tp2 < 1.0 else tid)

  all_attempts = []
  for r in records:
    all_attempts.extend(r.get("attempts", []))

  total_out_toks = sorted(
      float(a.get("total_output_tokens", 0)) for a in all_attempts
  )
  think_toks = sorted(float(a.get("thinking_tokens", 0)) for a in all_attempts)
  ans_toks = sorted(float(a.get("answer_tokens", 0)) for a in all_attempts)
  latencies = sorted(float(a.get("latency_s", 0.0)) for a in all_attempts)

  parsed_ok = sum(1 for a in all_attempts if a.get("parsed_ok", False))
  train_passed_cnt = sum(1 for a in all_attempts if a.get("train_passed", False))
  max_tok_trunc = sum(
      1
      for a in all_attempts
      if "MAX_TOKENS" in str(a.get("finish_reason", "")).upper()
      or "LENGTH" in str(a.get("finish_reason", "")).upper()
  )

  return {
      "variant": variant,
      "model": model_label,
      "completed_pairs": n_pairs,
      "total_expected_pairs": total_expected_pairs,
      "completed_tasks": n_tasks,
      "total_expected_tasks": total_expected_tasks,
      "elapsed_s": round(elapsed_s, 1),
      "kaggle_pair_weighted": {
          "pass1_count": pair_pass1,
          "pass2_count": pair_pass2,
          "total_pairs": n_pairs,
          "pass1_pct": round(100.0 * pair_pass1 / n_pairs, 2) if n_pairs else 0.0,
          "pass2_pct": round(100.0 * pair_pass2 / n_pairs, 2) if n_pairs else 0.0,
      },
      "arc_prize_task_weighted": {
          "pass1_score": round(task_pass1_score, 2),
          "pass2_score": round(task_pass2_score, 2),
          "total_tasks": n_tasks,
          "pass1_pct": (
              round(100.0 * task_pass1_score / n_tasks, 2) if n_tasks else 0.0
          ),
          "pass2_pct": (
              round(100.0 * task_pass2_score / n_tasks, 2) if n_tasks else 0.0
          ),
      },
      "token_stats": {
          "total_attempts": len(all_attempts),
          "parsed_grid_count": parsed_ok,
          "train_passed_count": train_passed_cnt,
          "max_tokens_truncated_count": max_tok_trunc,
          "mean_output_tokens": (
              round(statistics.mean(total_out_toks), 1)
              if total_out_toks
              else 0.0
          ),
          "median_output_tokens": round(percentile(total_out_toks, 0.5), 1),
          "p90_output_tokens": round(percentile(total_out_toks, 0.9), 1),
          "mean_thinking_tokens": (
              round(statistics.mean(think_toks), 1) if think_toks else 0.0
          ),
          "median_thinking_tokens": round(percentile(think_toks, 0.5), 1),
          "p90_thinking_tokens": round(percentile(think_toks, 0.9), 1),
          "mean_answer_tokens": (
              round(statistics.mean(ans_toks), 1) if ans_toks else 0.0
          ),
          "mean_latency_s": (
              round(statistics.mean(latencies), 1) if latencies else 0.0
          ),
      },
      "solved_tasks_pass1": solved_tasks_pass1,
      "solved_tasks_pass2": solved_tasks_pass2,
  }


def run_eval_variant(variant: str) -> None:
  if variant == "probe_qwen":
    spec = get_variant_spec("qwen_2_1")
    sampler = ModelSampler(spec, api_key=_API_KEY.value)
    res = sampler.generate_contents(
        [{"role": "user", "parts": [{"text": "Reply with [[1, 2], [3, 4]]"}]}],
        temperature=1.0,
        max_output_tokens=1024,
        thinking_level="HIGH",
    )
    print("Probe Qwen result:", json.dumps(res, indent=2), flush=True)
    return

  os.makedirs(_OUT_DIR.value, exist_ok=True)
  spec = get_variant_spec(variant)
  variant = spec.name
  jsonl_path = os.path.join(_OUT_DIR.value, f"{variant}.jsonl")
  summary_path = os.path.join(_OUT_DIR.value, f"{variant}_summary.json")

  model_label = (
      f"{spec.model_name} (req_max={spec.max_request_tokens},"
      f" budget={spec.total_token_budget}, turns={spec.max_turns},"
      f" att={spec.num_attempts}, tools={spec.use_tools},"
      f" diff={spec.rich_diff}, keep_unver={spec.keep_unverified_grid},"
      f" tricks={spec.use_prompt_tricks}, hybrid={spec.hybrid_prompts},"
      f" mht={spec.multi_hypothesis_thinking}, img={spec.use_images},"
      f" transp={spec.transpose_aug}, rank={spec.rank_and_retry3})"
  )
  sampler = ModelSampler(spec, api_key=_API_KEY.value)

  tasks = load_all_tasks(_LIMIT_TASKS.value, _TASK_IDS.value)
  all_work_items: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  for task_id, task in tasks:
    for pair_idx, test_pair in enumerate(task["test"]):
      all_work_items.append((task_id, pair_idx, task, test_pair))

  total_expected_tasks = len(tasks)
  total_expected_pairs = len(all_work_items)

  completed_records: dict[tuple[str, int], dict[str, Any]] = {}

  def refresh_completed_from_disk() -> None:
    if os.path.exists(jsonl_path):
      with open(jsonl_path, "r") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            rec = json.loads(line)
            attempts = rec.get("attempts", [])
            valid_atts = [
                a
                for a in attempts
                if a.get("total_output_tokens", 0) > 0
                and (not a.get("error") or a.get("parsed_ok"))
            ]
            if len(attempts) >= 2 and len(valid_atts) >= 2:
              completed_records[(rec["task_id"], rec["pair_idx"])] = rec
          except Exception:
            pass

  refresh_completed_from_disk()

  pending_items = [
      item
      for item in all_work_items
      if (item[0], item[1]) not in completed_records
  ]
  if _REVERSE.value:
    pending_items.reverse()

  print(
      f"[{variant}] Model: {model_label} | Total tasks: {total_expected_tasks}"
      f" ({total_expected_pairs} test pairs) | Already completed:"
      f" {len(completed_records)} | Pending: {len(pending_items)}"
      f" | reverse={_REVERSE.value}",
      flush=True,
  )

  t_start = time.time()
  write_lock = threading.Lock()

  def process_pair(
      item: tuple[str, int, dict[str, Any], dict[str, Any]]
  ) -> dict[str, Any]:
    task_id, pair_idx, task, test_pair = item
    with write_lock:
      refresh_completed_from_disk()
      if (task_id, pair_idx) in completed_records:
        return completed_records[(task_id, pair_idx)]

    test_in = test_pair["input"]
    target = test_pair["output"]

    num_parallel_attempts = max(
        spec.num_attempts, 3 if spec.rank_and_retry3 else 2
    )
    attempt_specs = []
    for idx in range(num_parallel_attempts):
      if spec.hybrid_prompts:
        # Attempts 1..N-1 use clean 2.3 tool prompt; final attempt(s) use 2.4 role+novelty prompt
        use_tricks_here = idx >= 2 or (num_parallel_attempts == 2 and idx == 1)
        att_spec = dataclasses.replace(
            spec,
            use_prompt_tricks=use_tricks_here,
            rich_diff=True,
            keep_unverified_grid=True,
        )
      else:
        att_spec = spec
      attempt_specs.append(att_spec)

    def _run_indexed_attempt(
        idx: int,
    ) -> tuple[list[list[int]] | None, dict[str, Any]]:
      use_transp = spec.transpose_aug and (idx == num_parallel_attempts - 1)
      if use_transp:
        t_task, t_in, t_target = transpose_task(task, test_in, target)
        t_pred, t_info = run_attempt_with_spec(
            attempt_specs[idx],
            sampler,
            t_task,
            t_in,
            t_target,
            spec.temp_att1 if idx == 0 else spec.temp_att2,
        )
        orig_pred = transpose_grid(t_pred) if t_pred is not None else None
        t_info["transposed_attempt"] = True
        t_info["exact_match"] = orig_pred == target
        t_info["pred_shape"] = (
            [len(orig_pred), len(orig_pred[0])] if orig_pred else None
        )
        t_info["target_shape"] = [len(target), len(target[0])]
        if _SAVE_OUTPUTS.value:
          t_info["pred_grid"] = orig_pred
        return orig_pred, t_info
      return run_attempt_with_spec(
          attempt_specs[idx],
          sampler,
          task,
          test_in,
          target,
          spec.temp_att1 if idx == 0 else spec.temp_att2,
      )

    with ThreadPoolExecutor(max_workers=num_parallel_attempts) as subpool:
      futs = [
          subpool.submit(_run_indexed_attempt, idx)
          for idx in range(num_parallel_attempts)
      ]
      all_atts = [f.result() for f in futs]

    if (
        spec.rank_and_retry3
        or spec.hybrid_prompts
        or spec.transpose_aug
        or num_parallel_attempts > 2
    ):
      raw_attempts = all_atts
    else:
      raw_attempts = all_atts[:2]

    pred1, pred2, attempts_out = select_and_rank_attempts(
        raw_attempts,
        rank_enabled=(
            spec.rank_and_retry3
            or spec.hybrid_prompts
            or spec.transpose_aug
            or num_parallel_attempts > 2
        ),
    )

    pass1 = pred1 == target
    pass2 = (pred1 == target) or (pred2 == target)
    rec: dict[str, Any] = {
        "task_id": task_id,
        "pair_idx": pair_idx,
        "num_task_pairs": len(task["test"]),
        "pass1": pass1,
        "pass2": pass2,
        "attempts": attempts_out,
    }
    if _SAVE_OUTPUTS.value:
      rec["target_grid"] = target
      rec["selected_pred1"] = pred1
      rec["selected_pred2"] = pred2

    valid_att_count = sum(
        1
        for a in attempts_out
        if a.get("total_output_tokens", 0) > 0
        and (not a.get("error") or a.get("parsed_ok"))
    )
    if valid_att_count < 2:
      tqdm.tqdm.write(
          f"[{variant}] WARNING: {task_id}[{pair_idx}] had <2 valid"
          " attempts due to server 503s; skipping save so it can be retried."
      )
      return rec

    with write_lock:
      refresh_completed_from_disk()
      if (task_id, pair_idx) not in completed_records:
        completed_records[(task_id, pair_idx)] = rec
        with open(jsonl_path, "a") as wf:
          wf.write(json.dumps(rec) + "\n")
      summary = compute_summary(
          variant=variant,
          model_label=model_label,
          records=list(completed_records.values()),
          total_expected_pairs=total_expected_pairs,
          total_expected_tasks=total_expected_tasks,
          elapsed_s=time.time() - t_start,
      )
      with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
      kp = summary["kaggle_pair_weighted"]
      ap = summary["arc_prize_task_weighted"]
      ts = summary["token_stats"]
      tok_list = ",".join(str(a["total_output_tokens"]) for a in attempts_out)
      tp_list = ",".join(
          "T" if a.get("train_passed") else "F" for a in attempts_out
      )
      tqdm.tqdm.write(
          f"[{variant}] {len(completed_records)}/{total_expected_pairs}"
          f" | {task_id}[{pair_idx}] p@1={pass1} p@2={pass2}"
          f" (toks={tok_list} train_pass={tp_list})"
          f" | Kaggle p@1={kp['pass1_pct']:.1f}% p@2={kp['pass2_pct']:.1f}%"
          f" | ARC p@1={ap['pass1_pct']:.1f}% p@2={ap['pass2_pct']:.1f}%"
          f" | mean_tok={ts['mean_output_tokens']}"
      )
    return rec

  pbar = tqdm.tqdm(
      total=total_expected_pairs,
      initial=len(completed_records),
      desc=variant,
      unit="pair",
      dynamic_ncols=True,
  )

  def process_pair_with_pbar(
      item: tuple[str, int, dict[str, Any], dict[str, Any]]
  ) -> dict[str, Any]:
    task_id, pair_idx, _, _ = item
    was_already_done = False
    with write_lock:
      if (task_id, pair_idx) in completed_records:
        was_already_done = True
    rec = process_pair(item)
    with write_lock:
      if not was_already_done:
        pbar.update(1)
      summary = compute_summary(
          variant=variant,
          model_label=model_label,
          records=list(completed_records.values()),
          total_expected_pairs=total_expected_pairs,
          total_expected_tasks=total_expected_tasks,
          elapsed_s=time.time() - t_start,
      )
      kp = summary["kaggle_pair_weighted"]
      ap = summary["arc_prize_task_weighted"]
      ts = summary["token_stats"]
      pbar.set_postfix(
          k_p1=f"{kp['pass1_pct']:.1f}%",
          k_p2=f"{kp['pass2_pct']:.1f}%",
          arc_p1=f"{ap['pass1_pct']:.1f}%",
          tok=f"{ts['mean_output_tokens']:.0f}",
      )
    return rec

  with ThreadPoolExecutor(max_workers=_WORKERS.value) as pool:
    futs = [pool.submit(process_pair_with_pbar, item) for item in pending_items]
    for fut in as_completed(futs):
      fut.result()
  pbar.close()

  with write_lock:
    refresh_completed_from_disk()
    final_summary = compute_summary(
        variant=variant,
        model_label=model_label,
        records=list(completed_records.values()),
        total_expected_pairs=total_expected_pairs,
        total_expected_tasks=total_expected_tasks,
        elapsed_s=time.time() - t_start,
    )
    with open(summary_path, "w") as sf:
      json.dump(final_summary, sf, indent=2)
  print(f"\n=== FINISHED {variant} ===", flush=True)
  print(json.dumps(final_summary, indent=2), flush=True)


_EXEC_SCRIPT = flags.DEFINE_string(
    "exec_script",
    "",
    "If set, execute the specified Python script inside the Blaze/Ditto environment.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _EXEC_SCRIPT.value:
    with open(_EXEC_SCRIPT.value, "r", encoding="utf-8") as f:
      code = f.read()
    exec(compile(code, _EXEC_SCRIPT.value, "exec"), {"__name__": "__main__"})  # pylint: disable=exec-used
    return
  run_eval_variant(_VARIANT.value)


if __name__ == "__main__":
  app.run(main)

