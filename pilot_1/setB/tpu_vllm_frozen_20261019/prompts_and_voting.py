"""Prompts, Qwen raw transcript builder, TTA transforms, and candidate voting for ARC-AGI-2."""

import copy
import json
import re
from typing import Any
import numpy as np
from PIL import Image

from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm_frozen_20261019 import sandbox

Grid = list[list[int]]

ARC_RGB_PALETTE = {
    0: (0, 0, 0),        # black
    1: (0, 116, 217),    # blue
    2: (255, 65, 54),    # red
    3: (46, 204, 64),    # green
    4: (255, 220, 0),    # yellow
    5: (170, 170, 170),  # grey
    6: (240, 18, 190),   # magenta
    7: (255, 133, 27),   # orange
    8: (127, 219, 255),  # azure
    9: (135, 12, 37),    # maroon
}

COLOR_LEGEND_STR = (
    "0:black, 1:blue, 2:red, 3:green, 4:yellow, "
    "5:grey, 6:magenta, 7:orange, 8:azure, 9:maroon"
)


def render_task_pil_image(
    task: dict[str, Any], test_input: Grid, cell_px: int = 14
) -> Image.Image:
  """Renders all training (input -> output) pairs and test_input as a single RGB PIL Image with grid lines."""
  def _draw_grid(g: Grid) -> np.ndarray:
    h = len(g)
    w = len(g[0]) if h else 0
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

  return Image.fromarray(canvas, mode="RGB")


# ==============================================================================
# 1. Prompt Templates & Qwen3.8 Raw Chat-Template Builder
# ==============================================================================

NO_TOOLS_SYSTEM = (
    "You are an expert solver for ARC-AGI-2 (Abstraction and Reasoning Corpus) puzzles.\n"
    "Each puzzle consists of 2D grids of integers 0..9 representing colors.\n"
    "All training pairs obey a single exact deterministic rule mapping each input grid to its output grid.\n"
    "Infer the exact rule, verify it against every training pair, and output the predicted 2D JSON grid [[...]] for the test input."
)

TOOLS_SYSTEM_PROGRAM = """You are solving an ARC (Abstraction and Reasoning Corpus) puzzle using an interactive Python session (`run_python`) and a verified program submission tool (`submit_program`).

Rules & Workflow:
- Grids are 2D arrays of integers 0..9 (colors; 0 is usually background).
- Preloaded in your persistent Python session:
  * `train`: list of `{"input": [[...]], "output": [[...]]}` dicts
  * `train_inputs`, `train_outputs`: lists of 2D numpy arrays
  * `test_input`: 2D list `[[...]]` (and `test_input_arr` as 2D numpy array)
  * `np`, `scipy.ndimage`, `Counter`, `deque`
  * Helpers: `objects(g, connectivity=4, bg=None, multicolour=False)` (or `find_objects`), `bbox(g, colour=None)`, `crop(g, box=None, bg=0)`, `bg_colour(g)`, `d8(g)`, and `check(solve)`
- Inspect the grids with Python code instead of counting rows/columns by hand.
- Keep each thinking turn focused and concise: form a concrete hypothesis, then immediately run a short script with `run_python` to inspect or test it.
- Write `def solve(grid):` taking a 2D list/array and returning a 2D list/array.
- When ready, call `submit_program` (or call `submit_program(solve)` inside `run_python`). The harness runs `solve(grid)` on every training input:
  * If all training pairs match 100%, `solve(test_input)` is automatically recorded as your final answer!
  * If any training pair fails, the exact failure summary is returned so you can fix `solve(grid)` and resubmit.

# Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "run_python", "description": "Execute Python code in the persistent stateful session (`train`, `test_input`, `np`, `scipy.ndimage`, `objects`, `check`, `submit_program` preloaded). Prints to stdout are returned.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute."}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "submit_program", "description": "Submit a self-contained Python `def solve(grid):` implementation. The harness tests `solve(grid)` on every training pair first; only if all training pairs match 100% does it record `solve(test_input)` as your final answer.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code defining `def solve(grid): ...`"}}, "required": ["code"]}}}
</tools>

For each function call, return a function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
<function=FUNCTION_NAME>
<parameter=code>
YOUR_PYTHON_CODE
</parameter>
</function>
</tool_call>"""

TOOLS_SYSTEM_QWEN_CODER_NEXT = """You are solving an ARC (Abstraction and Reasoning Corpus) puzzle using an interactive Python session (`run_python`) and a verified program submission tool (`submit_program`).

Rules & Workflow:
- Grids are 2D arrays of integers 0..9 (colors; 0 is usually background).
- Preloaded in your persistent Python session:
  * `train`: list of `{"input": [[...]], "output": [[...]]}` dicts
  * `train_inputs`, `train_outputs`: lists of 2D numpy arrays
  * `test_input`: 2D list `[[...]]` (and `test_input_arr` as 2D numpy array)
  * `np`, `scipy.ndimage`, `Counter`, `deque`
  * Helpers: `objects(g, connectivity=4, bg=None, multicolour=False)` (or `find_objects`), `bbox(g, colour=None)`, `crop(g, box=None, bg=0)`, `bg_colour(g)`, `d8(g)`, and `check(solve)`
- Inspect the grids with Python code instead of counting rows/columns by hand.
- Every tool call requires TWO parameters in order:
  1. `<parameter=reasoning>`: Write 3-5 concise bullet points analyzing the previous output/visual diff, formulating the exact 2D geometric/topological invariant across ALL training examples, and stating what this script will test or fix.
  2. `<parameter=code>`: Write short, clean, executable Python code (<50 lines). NEVER write long inline `#` comment blocks inside `code` — put all thinking in `<parameter=reasoning>`.
- Write `def solve(grid):` as a PURE function of its `grid` argument (`solve` MUST NOT reference `train` or `test_input`). Whenever you define `def solve(grid):` in `run_python`, `check(solve)` automatically runs on all training pairs and prints a visual coordinate diff (`got>exp`) so you can fix any off-by-one or boundary bugs.
- When `solve(grid)` passes all training pairs, call `submit_program` (and verify that your rule did not rely on a coincidence in `train`, such as sorting by row when the true rule is connectivity/color matching).

# Tools

You have access to the following functions:

<tools>
<function>
<name>run_python</name>
<description>Execute Python code in the persistent stateful session (`train`, `test_input`, `np`, `scipy.ndimage`, `objects`, `check`, `submit_program` preloaded). Prints to stdout are returned. Defining `def solve(grid):` automatically runs `check(solve)` with a visual diff.</description>
<parameters>
<parameter>
<name>reasoning</name>
<type>string</type>
<description>REQUIRED before code: 3-5 bullet points covering (1) what the previous output/diff showed, (2) your exact 2D transformation hypothesis across all training pairs, and (3) what this code tests.</description>
</parameter>
<parameter>
<name>code</name>
<type>string</type>
<description>Concise executable Python code (<50 lines; no long inline comment blocks).</description>
</parameter>
<required>["reasoning", "code"]</required>
</parameters>
</function>
<function>
<name>submit_program</name>
<description>Submit a self-contained pure Python `def solve(grid):` implementation (`solve` must only inspect `grid` and never reference `train` or `test_input`). The harness tests `solve(grid)` on every training pair first; only if all training pairs match 100% does it record `solve(test_input)` as your final answer.</description>
<parameters>
<parameter>
<name>reasoning</name>
<type>string</type>
<description>REQUIRED: Explain why `solve(grid)` is 100% general for `test_input` (no hardcoded training shapes/colors/coordinates).</description>
</parameter>
<parameter>
<name>code</name>
<type>string</type>
<description>Python code defining `def solve(grid): ...`</description>
</parameter>
<required>["reasoning", "code"]</required>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=run_python>
<parameter=reasoning>
- Observation: ...
- Hypothesis: ...
- Next check: ...
</parameter>
<parameter=code>
# concise executable Python code
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Always include `<parameter=reasoning>` BEFORE `<parameter=code>`
- Keep `<parameter=code>` concise (<50 lines of executable Python) and NEVER repeat `#` comments in a loop
</IMPORTANT>"""

THINK_NUDGE = (
    "\n\nWait — I have been reasoning for a long time without running any code."
    " I should stop here and test my current hypothesis in Python now, then"
    " continue from what the output shows.\n</think>\n\n"
    "<tool_call>\n<function=run_python>\n<parameter=code>\n"
)

CODER_NEXT_THINK_NUDGE = (
    "\n\n<tool_call>\n<function=run_python>\n<parameter=reasoning>\n"
    "- Testing current hypothesis with concise executable Python code.\n"
    "</parameter>\n<parameter=code>\n"
)

FORCE_SUBMIT_PROMPT = (
    "\n</think>\n\n<tool_call>\n<function=submit_program>\n<parameter=code>\n"
)

CODER_NEXT_FORCE_SUBMIT_PROMPT = (
    "\n\n<tool_call>\n<function=submit_program>\n<parameter=reasoning>\n"
    "- Submitting final general `def solve(grid):` implementation.\n"
    "</parameter>\n<parameter=code>\n"
)

# P2 (commit pacing): used instead of THINK_NUDGE once the attempt is past
# `--commit_after_tokens` output tokens without ever defining `def solve`.
COMMIT_THINK_NUDGE = (
    "\n\nWait — I have used a large part of my budget without writing any"
    " `def solve(grid)`. I should commit now: write my current best"
    " hypothesis as `def solve(grid):` (even if it is incomplete), let the"
    " harness check it against every training pair, and refine from the"
    " diff instead of analysing further.\n</think>\n\n"
    "<tool_call>\n<function=run_python>\n<parameter=code>\n"
)


def format_commit_note(used: int) -> str:
  """Harness-voice note appended to tool responses while a commit is due (P2)."""
  return (
      f"\n[harness: {used // 1000}k output tokens used and no `def solve(grid)`"
      " defined yet. In your next code cell, define `def solve(grid):`"
      " implementing your current best hypothesis (even if partial) — it is"
      " auto-checked against every training pair and the per-cell diff will"
      " tell you exactly what to fix.]"
  )


def fmt_grid_compact(g: Grid) -> str:
  return "\n".join("[" + ", ".join(str(v) for v in row) + "]" for row in g)


def grid_meta_line(g: Grid) -> str:
  h = len(g)
  w = len(g[0]) if h else 0
  counts: dict[int, int] = {}
  for r in g:
    for v in r:
      counts[v] = counts.get(v, 0) + 1
  hist = " ".join(
      f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
  )
  return f"{h}x{w} (colors: {hist})"


def build_user_prompt(
    task: dict[str, Any],
    test_input: Grid,
    use_tools: bool,
    soft_target: int,
    hard_budget: int,
    use_images: bool = False,
    multi_hypothesis: bool = False,
) -> str:
  """Builds the user message with training pairs, optional visual diagram, and best-practice guidance."""
  parts: list[str] = []
  if use_images:
    parts.append(
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "The attached color image renders every Training Example (left = Input,"
        " right = Output) stacked vertically, with the Test Input at the"
        f" bottom. Color palette: ({COLOR_LEGEND_STR}).\n"
        "Visual & Structural Inspection Best Practices:\n"
        "- First inspect the visual diagram to spot global 2D geometry,"
        " symmetries, rotations/reflections, divider lines, enclosing frames,"
        " repeating tiles, trajectories, and object roles (background,"
        " template/stamp, marker/key).\n"
        "- Never dismiss any non-background pixel as random noise; every cell"
        " has a deterministic role."
    )
  if multi_hypothesis:
    parts.append(
        "Multi-Hypothesis Deliberation:\n"
        "- Whenever the visual pattern could be explained in more than one way,"
        " briefly formulate two competing candidate rules (Hypothesis A vs."
        " Hypothesis B) and use `run_python` to test both on `train` before"
        " committing."
    )
  for i, pair in enumerate(task["train"], 1):
    inp, out = pair["input"], pair["output"]
    parts.append(
        f"Training Example {i}:\n"
        f"Input ({grid_meta_line(inp)}):\n{fmt_grid_compact(inp)}\n"
        f"Output ({grid_meta_line(out)}):\n{fmt_grid_compact(out)}"
    )
  parts.append(
      f"Test Input ({grid_meta_line(test_input)}):\n{fmt_grid_compact(test_input)}"
  )
  if use_tools:
    parts.append(
        f"Use `run_python` (with `<parameter=reasoning>` then `<parameter=code>`)"
        f" to inspect the grids and test `def solve(grid):` on `train`,"
        f" then call `submit_program` with `def solve(grid):`."
        f" [Token budget: aim to finish within {soft_target // 1000}k tokens"
        f" (hard limit {hard_budget // 1000}k)]"
    )
  else:
    parts.append(
        "Reason step by step to find the exact rule that maps every Training"
        " Input to its Output. Then apply the rule to Test Input and output"
        " ONLY the final 2D JSON grid [[...]] at the very end of your response."
    )
  return "\n\n".join(parts)


def build_initial_transcript(
    task: dict[str, Any],
    test_input: Grid,
    use_tools: bool,
    soft_target: int,
    hard_budget: int,
    use_images: bool = False,
    multi_hypothesis: bool = False,
    coder_next_format: bool = False,
) -> str:
  """Builds the raw Qwen chat transcript optimized for either Qwen3-Coder-Next or Qwen3.8 thinking models."""
  if not use_tools:
    sys_msg = NO_TOOLS_SYSTEM
  elif coder_next_format:
    sys_msg = TOOLS_SYSTEM_QWEN_CODER_NEXT
  else:
    sys_msg = TOOLS_SYSTEM_PROGRAM
  user_msg = build_user_prompt(
      task=task,
      test_input=test_input,
      use_tools=use_tools,
      soft_target=soft_target,
      hard_budget=hard_budget,
      use_images=use_images,
      multi_hypothesis=multi_hypothesis,
  )
  assistant_prefix = (
      "<|im_start|>assistant\n"
      if coder_next_format
      else "<|im_start|>assistant\n<think>\n"
  )
  return (
      f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
      f"<|im_start|>user\n{user_msg}<|im_end|>\n"
      f"{assistant_prefix}"
  )


def format_budget_note(used: int, soft: int, hard: int) -> str:
  note = (
      f"\n[budget: {used // 1000}k / {hard // 1000}k output tokens used"
      f" (soft target {soft // 1000}k)]"
  )
  if used >= int(hard * 0.75):
    note += (
        " WARNING: >75% of token budget used! Finalize `def solve(grid):` and"
        " call `submit_program` immediately."
    )
  elif used >= soft:
    note += (
        f" Note: past the {soft // 1000}k soft target — submit your rule as"
        " soon as it fits all training pairs."
    )
  return note


# ==============================================================================
# 2. Tool Call & Grid Extraction Parsers
# ==============================================================================

_XML_FUNC_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([a-zA-Z0-9_]+)>\s*([\s\S]*?)(?:</function>\s*(?:</tool_call>)?|(?=<function=)|$)"
)
_XML_PARAM_REASONING_RE = re.compile(
    r"<parameter=reasoning>\n?([\s\S]*?)\n?(?:</parameter>|(?=<parameter=code>)|$)"
)
_XML_PARAM_CODE_RE = re.compile(
    r"<parameter=code>\n?([\s\S]*?)\n?(?:</parameter>|$)"
)


def strip_excessive_comments(code: str, max_consecutive_comments: int = 5) -> tuple[str, int]:
  """Strips degenerate consecutive `#` comment loops from code.

  Returns `(cleaned_code, num_removed_comment_lines)`.
  """
  lines = code.splitlines()
  cleaned: list[str] = []
  consec_comments = 0
  removed = 0
  for ln in lines:
    stripped = ln.strip()
    if stripped.startswith("#"):
      consec_comments += 1
      if consec_comments <= max_consecutive_comments:
        cleaned.append(ln)
      else:
        removed += 1
    elif not stripped and consec_comments > max_consecutive_comments:
      removed += 1
    else:
      consec_comments = 0
      cleaned.append(ln)
  return "\n".join(cleaned).strip(), removed


def sanitize_turn_for_transcript(
    text: str, finish_reason: str
) -> tuple[str, bool]:
  """Sanitizes a turn output before appending to transcript to prevent comment-loop KV-cache poisoning.

  Returns `(sanitized_text, had_comment_loop_or_truncation)`.
  """
  if not text:
    return text, False

  had_issue = False
  out_text = text

  # If <parameter=code> is present, strip any >5 consecutive `#` comment loops
  if "<parameter=code>" in out_text:
    pre_code, post_code = out_text.split("<parameter=code>", 1)
    if "</parameter>" in post_code:
      code_body, tail = post_code.split("</parameter>", 1)
      cleaned_code, removed = strip_excessive_comments(code_body)
      if removed >= 5:
        had_issue = True
      out_text = f"{pre_code}<parameter=code>\n{cleaned_code}\n</parameter>{tail}"
    else:
      # Truncated inside <parameter=code> (e.g. finish_reason == "length")
      had_issue = True
      cleaned_code, _ = strip_excessive_comments(post_code, max_consecutive_comments=3)
      # Drop potentially incomplete last line if cut by length
      c_lines = cleaned_code.splitlines()
      if finish_reason == "length" and len(c_lines) > 1:
        c_lines = c_lines[:-1]
      cleaned_code = "\n".join(c_lines).strip()
      if not cleaned_code:
        cleaned_code = "pass"
      out_text = (
          f"{pre_code}<parameter=code>\n{cleaned_code}\n"
          "</parameter>\n</function>\n</tool_call>"
      )
  elif finish_reason == "length" and "<parameter=reasoning>" in out_text and "<parameter=code>" not in out_text:
    had_issue = True
    r_lines = out_text.splitlines()
    if len(r_lines) > 12:
      out_text = "\n".join(r_lines[:12])

  return out_text, had_issue


def split_turn_thinking_and_tool(text: str) -> tuple[str, str, str]:
  """Splits a single turn output into `(thought_text, tool_call_text, visible_answer_text)`.

  Handles:
  1. `<think>...</think>` models (Qwen3 / Qwen3.8)
  2. `Qwen3-Coder-Next` pre-tool-call natural-language reasoning AND `<parameter=reasoning>...</parameter>` inside tool calls!
  """
  if not text:
    return "", "", ""
  thought_parts: list[str] = []
  rem = text
  if "</think>" in rem:
    think_seg, rem = rem.split("</think>", 1)
    cleaned_think = think_seg.replace("<think>", "").strip()
    if cleaned_think:
      thought_parts.append(cleaned_think)
  elif rem.lstrip().startswith("<think>"):
    rem = rem.replace("<think>", "", 1)

  tc_idx = rem.find("<tool_call>")
  if tc_idx < 0:
    tc_idx = rem.find("<function=")

  if tc_idx >= 0:
    pre_tc = rem[:tc_idx].strip()
    if pre_tc:
      thought_parts.append(pre_tc)
    tool_raw = rem[tc_idx:].strip()
    # Also extract any `<parameter=reasoning>...</parameter>` inside the tool call as thinking!
    for rm in _XML_PARAM_REASONING_RE.finditer(tool_raw):
      r_body = rm.group(1).strip()
      if r_body:
        thought_parts.append(r_body)
    # Remove `<parameter=reasoning>...</parameter>` from `tool_str` so `tool_use_tokens` counts only the code + XML wrapper
    tool_code_only = _XML_PARAM_REASONING_RE.sub("", tool_raw).strip()
    return "\n\n".join(thought_parts), tool_code_only, ""

  # No tool call in `rem`
  if "</think>" in text:
    return "\n\n".join(thought_parts), "", rem.strip()
  stripped = rem.strip()
  if stripped.startswith("CONFIRMED") or stripped.startswith("[["):
    return "\n\n".join(thought_parts), "", stripped
  if stripped:
    thought_parts.append(stripped)
  return "\n\n".join(thought_parts), "", ""


def parse_tool_calls(text: str) -> list[tuple[str, str]]:
  """Extracts `(tool_name, code)` from visible text (supporting optional `<parameter=reasoning>` before `<parameter=code>`)."""
  after = text.split("</think>", 1)[1] if "</think>" in text else text
  calls: list[tuple[str, str]] = []
  for fm in _XML_FUNC_RE.finditer(after):
    fn_name = fm.group(1).strip()
    fn_body = fm.group(2)
    cm = _XML_PARAM_CODE_RE.search(fn_body)
    if cm:
      code_body, _ = strip_excessive_comments(cm.group(1).strip())
      if code_body:
        calls.append((fn_name, code_body))
  if calls:
    return calls

  # Fallback 1: JSON inside <tool_call>
  for tc_m in re.finditer(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", after):
    inner = tc_m.group(1).strip()
    if inner.startswith("{"):
      try:
        obj = json.loads(inner)
        fn_name = obj.get("name", "run_python")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
          args = json.loads(args)
        if isinstance(args, dict) and isinstance(args.get("code"), str):
          calls.append((fn_name, args["code"].strip()))
      except Exception:
        pass
  if calls:
    return calls

  # Fallback 2: Markdown ```python ... ``` blocks
  for md_m in re.finditer(r"```(?:python|py)\s*\n([\s\S]*?)```", after):
    code = md_m.group(1).strip()
    if code:
      calls.append(("run_python", code))
  return calls


def parse_last_grid(text: str) -> Grid | None:
  """Scans backward from the end of `text` for the last valid 2D JSON grid `[[...]]`."""
  if not text:
    return None
  after = text.split("</think>", 1)[1] if "</think>" in text else ""
  for candidate in (after, text):
    for end in range(len(candidate) - 1, -1, -1):
      if candidate[end] == "]":
        depth = 0
        for start in range(end, -1, -1):
          if candidate[start] == "]":
            depth += 1
          elif candidate[start] == "[":
            depth -= 1
            if depth == 0:
              sub = candidate[start : end + 1]
              if "\n" in sub or "[" in sub[1:]:
                try:
                  val = json.loads(sub)
                  norm = sandbox.normalize_grid(val)
                  if norm is not None:
                    return norm
                except Exception:
                  pass
              break
  return None


# ==============================================================================
# 3. Dihedral (D8) & Color-Permutation Test-Time Augmentation (TTA)
# ==============================================================================


class TtaTransform:
  """Invertible dihedral + color permutation transform for an ARC task."""

  def __init__(self, attempt_idx: int, enable_tta: bool):
    self.attempt_idx = attempt_idx
    self.enable_tta = enable_tta
    # D8 index: 0=id, 1=transpose, 2=rot90, 3=rot180, 4=rot270, 5=fliplr, 6=flipud, 7=rot90+fliplr
    self.d8_idx = (attempt_idx % 8) if enable_tta else 0

  def apply_grid(self, g: Grid) -> Grid:
    if not self.enable_tta or self.d8_idx == 0:
      return copy.deepcopy(g)
    arr = np.asarray(g, dtype=int)
    if self.d8_idx == 1:
      out = arr.T
    elif self.d8_idx == 2:
      out = np.rot90(arr, 1)
    elif self.d8_idx == 3:
      out = np.rot90(arr, 2)
    elif self.d8_idx == 4:
      out = np.rot90(arr, 3)
    elif self.d8_idx == 5:
      out = np.fliplr(arr)
    elif self.d8_idx == 6:
      out = np.flipud(arr)
    else:
      out = np.rot90(np.fliplr(arr), 1)
    return out.tolist()

  def invert_grid(self, g: Grid | None) -> Grid | None:
    if g is None:
      return None
    if not self.enable_tta or self.d8_idx == 0:
      return copy.deepcopy(g)
    arr = np.asarray(g, dtype=int)
    if self.d8_idx == 1:
      out = arr.T
    elif self.d8_idx == 2:
      out = np.rot90(arr, -1)
    elif self.d8_idx == 3:
      out = np.rot90(arr, -2)
    elif self.d8_idx == 4:
      out = np.rot90(arr, -3)
    elif self.d8_idx == 5:
      out = np.fliplr(arr)
    elif self.d8_idx == 6:
      out = np.flipud(arr)
    else:
      out = np.fliplr(np.rot90(arr, -1))
    return out.tolist()

  def transform_task(
      self, task: dict[str, Any], test_input: Grid
  ) -> tuple[dict[str, Any], Grid]:
    if not self.enable_tta or self.d8_idx == 0:
      return copy.deepcopy(task), copy.deepcopy(test_input)
    t_task: dict[str, Any] = {
        "train": [
            {
                "input": self.apply_grid(p["input"]),
                "output": self.apply_grid(p["output"]),
            }
            for p in task["train"]
        ],
        "test": [
            {"input": self.apply_grid(p["input"])}
            for p in task.get("test", [])
        ],
    }
    return t_task, self.apply_grid(test_input)


# ==============================================================================
# 4. Candidate Ranking, Deduplication & Majority Voting Across Attempts
# ==============================================================================


def _primary_candidate(att: dict[str, Any], grid_override: Grid | None = None) -> list[dict[str, Any]]:
  """One candidate per attempt: its final (primary) grid tagged with attempt-level train stats."""
  g = grid_override if grid_override is not None else att.get("pred_grid")
  if g is None:
    return []
  return [{
      "grid": g,
      "source": "primary",
      "train_pass": att.get("train_passed"),
      "train_frac": att.get("best_train_frac", 0.0),
      "cell_frac": att.get("best_cell_frac", 0.0),
      "suspect": False,
      "is_primary": True,
  }]


def select_top_two_predictions(
    attempt_records: list[dict[str, Any]],
    fallback_grid: Grid,
    primary_only: bool = False,
) -> tuple[Grid, Grid, dict[str, Any]]:
  """Pools all primary & workspace candidates across attempts and selects distinct `(attempt_1, attempt_2)`.

  Ranking hierarchy (combining collaborator `selection.py` + Best-of-N majority voting):
    Tier 0: `train_pass is True` and `not suspect`
    Tier 1: `train_pass is True` (even if audit flagged `suspect`) or `is_primary` with `train_pass is None`
    Tier 2: Partial train scorers sorted by `train_frac`, `cell_frac`, and vote count.
  Within each tier, distinct grids are ranked by:
    1. Number of independent attempts whose primary/verified candidate produced this exact grid (`verified_votes`, then `total_votes`)
    2. Best `train_frac` and `cell_frac`
    3. Primary candidate over workspace salvage alternative.

  If `primary_only` is True ("primary-only voting"), each attempt contributes exactly ONE
  candidate (its final grid, tagged with the attempt's `train_passed` / best train score);
  workspace-salvage, intermediate `check()` grids and cross-pair pseudo-attempts
  (attempt_idx >= 100) are ignored.
  """
  by_key: dict[str, dict[str, Any]] = {}

  for att_idx, att in enumerate(attempt_records):
    seen_in_attempt: set[str] = set()
    if primary_only:
      if int(att.get("attempt_idx", 0) or 0) >= 100:
        continue  # cross-pair pseudo-attempt
      cands = _primary_candidate(att)
    else:
      cands = att.get("candidates", [])
      if not cands and att.get("pred_grid") is not None:
        cands = _primary_candidate(att)

    for c in cands:
      g = c.get("grid")
      if g is None:
        continue
      k = sandbox.grid_key(g)
      if not k:
        continue
      tp = c.get("train_pass")
      tf = float(c.get("train_frac") or 0.0)
      cf = float(c.get("cell_frac") or 0.0)
      suspect = bool(c.get("suspect", False))
      is_prim = bool(c.get("is_primary", False))

      if tp is True and not suspect:
        tier = 0
      elif (tp is True and suspect) or (is_prim and tp is None):
        tier = 1
      else:
        tier = 2

      if k not in by_key:
        by_key[k] = {
            "grid": g,
            "best_tier": tier,
            "verified_votes": 0,
            "primary_votes": 0,
            "attempt_Set": set(),
            "best_train_frac": tf,
            "best_cell_frac": cf,
            "has_primary": is_prim,
            "first_attempt_idx": att_idx,
            "sources": [c.get("source", "")],
        }
      entry = by_key[k]
      entry["best_tier"] = min(entry["best_tier"], tier)
      entry["best_train_frac"] = max(entry["best_train_frac"], tf)
      entry["best_cell_frac"] = max(entry["best_cell_frac"], cf)
      entry["has_primary"] = entry["has_primary"] or is_prim

      if k not in seen_in_attempt:
        seen_in_attempt.add(k)
        entry["attempt_Set"].add(att_idx)
        if tp is True and not suspect:
          entry["verified_votes"] += 1
        if is_prim:
          entry["primary_votes"] += 1

  if not by_key:
    fb = copy.deepcopy(fallback_grid)
    return fb, fb, {"num_unique_candidates": 0, "top1_tier": 3, "top1_votes": 0}

  ranked = sorted(
      by_key.values(),
      key=lambda e: (
          -e["best_tier"],  # Tier 0 > Tier 1 > Tier 2
          e["verified_votes"],  # Most train-verified votes across attempts
          e["primary_votes"],  # Most primary attempt votes
          len(e["attempt_Set"]),  # Total distinct attempts producing this grid
          e["best_train_frac"],  # Highest fraction of training pairs passed
          e["best_cell_frac"],  # Highest training cell accuracy
          1 if e["has_primary"] else 0,
          -e["first_attempt_idx"],
      ),
      reverse=True,
  )

  att1 = ranked[0]["grid"]
  att2 = ranked[1]["grid"] if len(ranked) >= 2 else att1
  meta = {
      "num_unique_candidates": len(ranked),
      "top1_tier": ranked[0]["best_tier"],
      "top1_verified_votes": ranked[0]["verified_votes"],
      "top1_total_votes": len(ranked[0]["attempt_Set"]),
      "top1_train_frac": round(ranked[0]["best_train_frac"], 4),
      "top2_tier": ranked[1]["best_tier"] if len(ranked) >= 2 else None,
      "top2_total_votes": (
          len(ranked[1]["attempt_Set"]) if len(ranked) >= 2 else 0
      ),
  }
  return att1, att2, meta


# ==============================================================================
# 5. Selection-policy grid: {pooled, primary-only} x {latest (legacy audit), first-verified}
# ==============================================================================

SELECTION_POLICIES = (
    "pooled_latest",   # legacy pipeline: all candidates, destructive audit
    "pooled_first",    # all candidates, non-destructive audit (freeze first verified)
    "primary_latest",  # primary-only voting, destructive audit
    "primary_first",   # primary-only voting + non-destructive audit
)


def legacy_audit_view(att: dict[str, Any]) -> dict[str, Any]:
  """Returns the attempt as it would look under the legacy (destructive) audit.

  With `freeze_first_verified`, a later verified resubmission is stored with
  `wants_primary=True, is_primary=False`. The legacy sandbox would have made the
  LAST `wants_primary` candidate the primary (and the attempt's `pred_grid`).
  Attempts recorded without freezing (or with no override) are returned as-is.

  Args:
    att: compact attempt record (with `candidates`).

  Returns:
    A shallow copy with `pred_grid` / `candidates[*].is_primary` rewritten.
  """
  cands = att.get("candidates") or []
  if not att.get("frozen_overrides") or not cands:
    return att
  last_idx = None
  for i, c in enumerate(cands):
    if c.get("wants_primary"):
      last_idx = i
  if last_idx is None:
    return att
  view = dict(att)
  view["pred_grid"] = cands[last_idx]["grid"]
  view["candidates"] = [
      dict(c, is_primary=(i == last_idx)) for i, c in enumerate(cands)
  ]
  return view


def score_selection_policies(
    attempt_records: list[dict[str, Any]],
    fallback_grid: Grid,
    target: Grid | None = None,
) -> dict[str, dict[str, Any]]:
  """Scores all four selection policies on the same attempt records.

  Args:
    attempt_records: attempt dicts (may include cross-pair pseudo-attempts).
    fallback_grid: grid to submit if no candidate exists (the test input).
    target: optional ground-truth grid; if given, pass flags are added.

  Returns:
    {policy_name: {"attempt_1", "attempt_2", "meta", ["pass_at_1", "pass_at_2"]}}
  """
  legacy = [legacy_audit_view(a) for a in attempt_records]
  configs = {
      "pooled_latest": (legacy, False),
      "pooled_first": (attempt_records, False),
      "primary_latest": (legacy, True),
      "primary_first": (attempt_records, True),
  }
  out: dict[str, dict[str, Any]] = {}
  for name, (recs, primary_only) in configs.items():
    a1, a2, meta = select_top_two_predictions(
        recs, fallback_grid=fallback_grid, primary_only=primary_only
    )
    entry: dict[str, Any] = {"attempt_1": a1, "attempt_2": a2, "meta": meta}
    if target is not None:
      entry["pass_at_1"] = a1 == target
      entry["pass_at_2"] = (a1 == target) or (a2 == target)
    out[name] = entry
  return out
