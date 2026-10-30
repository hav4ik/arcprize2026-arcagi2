"""ARC-AGI-2 Batched TPU vLLM Evaluation Runner for Qwen3.8-27B-FP8 (Experiments 1, 2, 3, 4).

Supports:
- Experiment 1 (`--mode=no_tools --num_attempts=2 --total_token_budget=48000`):
  Single-turn `xhigh` reasoning across all 167 test pairs (120 tasks).
- Experiment 2 (`--mode=tools --num_attempts=2 --total_token_budget=48000 --turn_think_cap=8000`):
  Multi-turn interactive Python REPL (`run_python` + `submit_program`) with
  append-only prefix-cached transcripts, 8k per-turn thinking cap + nudge,
  live training-pair verification, and zero-token workspace candidate salvage.
- Experiment 3 (`--mode=tools --num_attempts=8 --total_token_budget=96000 --turn_think_cap=8000`):
  Best-of-8 scaling with train-pass filtering, majority voting across attempts,
  and automatic early stopping when consensus is reached.
- Experiment 4 (`--mode=tools --num_attempts=8 --total_token_budget=96000 --enable_tta=true`):
  Adds Dihedral (D8) test-time augmentation across attempts before voting.

Saves incremental results to CNS (`--output_dir`):
- `predictions.jsonl` (or `predictions_shard_XXXofYYY.jsonl`)
- `summary.json` (live Kaggle pair-weighted & ARC Prize task-weighted pass@1 / pass@2 + tok/s)
- `submission.json` (official Kaggle submission format)
"""

from collections.abc import Sequence
from concurrent import futures
import copy
import json
import os
import shutil
import subprocess
import time
from typing import Any

from absl import flags
import tqdm
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import prompts_and_voting
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import sandbox
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "CNS path to Qwen3.8-27B-FP8 model weights.",
)
flags.DEFINE_string(
    "cns_data_path",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data",
    "CNS path to ARC-AGI-2 dataset.",
)
flags.DEFINE_string(
    "output_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_qwen_tpu_runs/exp1_no_tools",
    "CNS directory to save predictions.jsonl, summary.json, and submission.json.",
)
flags.DEFINE_string(
    "exp_name",
    "exp1_no_tools_48k",
    "Experiment identifier recorded in summary.json.",
)
flags.DEFINE_enum(
    "mode",
    "tools",
    ["no_tools", "tools"],
    "Evaluation mode: 'no_tools' (Exp 1) or 'tools' (Exp 2, 3, 4).",
)
flags.DEFINE_integer(
    "num_attempts",
    2,
    "Number of attempts per test pair (2 for Exp 1/2, 8 for Exp 3/4).",
)
flags.DEFINE_integer(
    "total_token_budget",
    48000,
    "Hard output token cap per attempt (48000 for Exp 1/2, 96000 for Exp 3/4).",
)
flags.DEFINE_integer(
    "soft_token_target",
    24000,
    "Soft output token target for budget pacing notes (24000 for 48k, 48000 for 96k).",
)
flags.DEFINE_integer(
    "turn_think_cap",
    8000,
    "Max output tokens per generation step in 'tools' mode before injecting think nudge.",
)
flags.DEFINE_integer(
    "max_turns",
    16,
    "Maximum generation/tool steps per attempt in 'tools' mode.",
)
flags.DEFINE_boolean(
    "enable_tta",
    False,
    "If True, applies Dihedral D8 Test-Time Augmentation across attempts (Exp 4).",
)
flags.DEFINE_boolean(
    "use_images",
    False,
    "If True, renders each ARC task as a color RGB PIL image and passes it via multi_modal_data alongside <|vision_start|><|image_pad|><|vision_end|>.",
)
flags.DEFINE_boolean(
    "multi_hypothesis",
    False,
    "If True, adds Multi-Hypothesis Deliberation instructions to the user prompt.",
)
flags.DEFINE_boolean(
    "hybrid_modality",
    False,
    "If True, even attempts use Text+Tools and odd attempts use Image+Tools.",
)
flags.DEFINE_boolean(
    "audit_on_submit",
    False,
    "If True, performs a 1-turn post-submit generalization audit on solve(test_input) before finishing.",
)
flags.DEFINE_boolean(
    "shard_by_task",
    False,
    "If True, assigns all test pairs of the same task_id to the same shard and shares verified solve(grid) candidates across sibling test pairs.",
)
flags.DEFINE_integer(
    "early_stop_verified_votes",
    3,
    "When num_attempts > 2, stop launching remaining waves for a pair once this many attempts pass train_check and agree on the exact same test grid.",
)
flags.DEFINE_boolean(
    "freeze_first_verified",
    False,
    "Non-destructive audit: once an attempt has a verified (train-pass & audit-clean) program, later verified resubmissions (e.g. after the post-submit audit) are recorded as extra candidates but never replace that attempt's primary answer.",
)
flags.DEFINE_boolean(
    "keep_candidates",
    True,
    "If True, keep the full per-attempt candidate list (grids + sources) in predictions.jsonl so selection policies can be re-scored offline.",
)
flags.DEFINE_boolean(
    "save_transcripts",
    True,
    "If True, write the full raw transcript of every attempt to transcripts[_shard_XXX].jsonl in output_dir.",
)
flags.DEFINE_enum(
    "submission_policy",
    "primary_first",
    list(prompts_and_voting.SELECTION_POLICIES),
    "Selection policy used for submission.json. The JSONL/summary report all four policies; `pass_at_1/2` keep the legacy `pooled_latest` policy for comparability with earlier runs.",
)
flags.DEFINE_integer(
    "pair_batch_size",
    16,
    "Number of test pairs processed concurrently per batch wave on a single TPU worker.",
)
flags.DEFINE_integer(
    "attempts_per_wave",
    2,
    "Number of attempts per test pair launched in each wave (allows early stopping when num_attempts=8).",
)
flags.DEFINE_float("temperature", 1.0, "Sampling temperature.")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling top_p.")
flags.DEFINE_integer("top_k", 20, "Top-k sampling.")
flags.DEFINE_float(
    "repetition_penalty",
    1.0,
    "Repetition penalty for vLLM SamplingParams (use 1.05 for 4-bit AWQ Qwen3-Coder-Next to prevent comment loops).",
)
flags.DEFINE_integer(
    "tensor_parallel_size",
    4,
    "Tensor parallel size across TPU chips (4 for ghostfish=2x2x1).",
)
flags.DEFINE_integer(
    "max_model_len",
    65536,
    "Maximum context length for vLLM on TPU (65536 for 48k budget, 114688 for 96k budget).",
)
flags.DEFINE_integer(
    "max_num_batched_tokens",
    8192,
    "Maximum batched tokens per iteration.",
)
flags.DEFINE_integer(
    "max_num_seqs",
    32,
    "Maximum concurrent sequences in vLLM.",
)
flags.DEFINE_float(
    "gpu_memory_utilization",
    0.90,
    "Fraction of TPU HBM reserved for model weights and KV cache.",
)
flags.DEFINE_integer(
    "limit_tasks",
    0,
    "If > 0, evaluate only the first N tasks (for quick smoke tests).",
)
flags.DEFINE_integer(
    "num_shards",
    1,
    "Total number of parallel TPU worker shards.",
)
flags.DEFINE_integer(
    "shard_idx",
    0,
    "0-based shard index for this worker.",
)
flags.DEFINE_enum(
    "chat_format",
    "auto",
    ["auto", "qwen_coder_next", "qwen3_think"],
    "Chat/tool prompt format: 'qwen_coder_next' (non-<think> agentic XML tool format from Qwen3-Coder-Next chat_template.jinja), 'qwen3_think' (Qwen3/3.8 <think>...</think> format), or 'auto' (infer from model_path).",
)
# ---- Pilot flags (ARC-AGI-2 pilots P1/P2/P3; see pilots_plan in the report) ----
flags.DEFINE_boolean(
    "auto_check_on_define",
    True,
    "P1: if True, any `run_python` turn that (re)defines `solve` without calling `check`/`submit_program` gets an automatic harness verification (structured diff on every training pair) appended to the tool response. Set False to reproduce the Exp 2/7B baseline behaviour.",
)
flags.DEFINE_integer(
    "commit_after_tokens",
    0,
    "P2: if > 0, once an attempt has generated this many output tokens without ever defining `def solve(grid)`, every subsequent tool response carries a commit note and the think-nudge asks for a `def solve` with the current best hypothesis. 0 disables.",
)
flags.DEFINE_string(
    "task_ids",
    "",
    "Optional comma-separated list of task ids to evaluate (e.g. the frozen DEV split). Empty = all tasks.",
)
flags.DEFINE_string(
    "task_ids_file",
    "",
    "Optional JSON file (CNS or local) containing either a list of task ids or a dict with key --task_ids_key.",
)
flags.DEFINE_string(
    "task_ids_key", "dev", "Key inside --task_ids_file when it is a dict."
)
flags.DEFINE_integer(
    "max_repair_rounds",
    2,
    "Axis 1: Number of explicit post-submission repair rounds allowed when a submitted solve(grid) fails demonstrations.",
)
flags.DEFINE_boolean(
    "structured_feedback",
    True,
    "Axis 1: If True, include expected & predicted grids and zero-indexed mismatch coordinates in execution feedback.",
)
flags.DEFINE_boolean(
    "diverse_hypotheses",
    False,
    "Axis 2: If True, inject a distinct hypothesis lens into each attempt prompt so N=4 attempts explore distinct rule families.",
)
flags.DEFINE_boolean(
    "structured_summary",
    False,
    "Axis 3: If True, include automated structural summary (dimensions, color histograms, 4-conn components, and input->output diff maps) in the prompt.",
)
flags.DEFINE_boolean(
    "enable_color_tta",
    False,
    "Axis 3: If True, apply invertible non-background color permutations across attempts alongside TTA.",
)


def is_coder_next_format() -> bool:
  """Returns True if using Qwen3-Coder-Next chat/tool format."""
  if FLAGS.chat_format == "qwen_coder_next":
    return True
  if FLAGS.chat_format == "qwen3_think":
    return False
  mp = FLAGS.model_path.lower()
  return "coder-next" in mp or "coder_next" in mp


class AttemptState:
  """Tracks one multi-turn attempt inside a batched vLLM execution wave."""

  def __init__(
      self,
      task_id: str,
      pair_idx: int,
      attempt_idx: int,
      raw_task: dict[str, Any],
      raw_test_input: list[list[int]],
      target_output: list[list[int]],
      use_tools: bool,
      enable_tta: bool,
      use_images: bool,
      multi_hypothesis: bool,
      soft_target: int,
      hard_budget: int,
      coder_next_format: bool = False,
  ):
    self.task_id = task_id
    self.pair_idx = pair_idx
    self.attempt_idx = attempt_idx
    self.raw_task = raw_task
    self.raw_test_input = raw_test_input
    self.target_output = target_output
    self.use_tools = use_tools
    self.use_images = use_images
    self.multi_hypothesis = multi_hypothesis
    self.soft_target = soft_target
    self.hard_budget = hard_budget
    self.coder_next_format = coder_next_format

    self.tta = prompts_and_voting.TtaTransform(
        attempt_idx=attempt_idx,
        enable_tta=enable_tta,
        enable_color_tta=FLAGS.enable_color_tta,
    )
    self.tta_task, self.tta_test_input = self.tta.transform_task(
        raw_task, raw_test_input
    )
    self.pil_image = (
        prompts_and_voting.render_task_pil_image(
            self.tta_task, self.tta_test_input
        )
        if use_images
        else None
    )
    self.sandbox = sandbox.ArcPythonSandbox(
        task=self.tta_task,
        test_input=self.tta_test_input,
        freeze_first_verified=FLAGS.freeze_first_verified,
        auto_check=FLAGS.auto_check_on_define,
        structured_feedback=FLAGS.structured_feedback,
    )
    self.transcript = prompts_and_voting.build_initial_transcript(
        task=self.tta_task,
        test_input=self.tta_test_input,
        use_tools=use_tools,
        soft_target=soft_target,
        hard_budget=hard_budget,
        use_images=use_images,
        multi_hypothesis=multi_hypothesis,
        coder_next_format=coder_next_format,
        structured_summary=FLAGS.structured_summary,
        diverse_attempt_idx=(
            attempt_idx if FLAGS.diverse_hypotheses else None
        ),
    )
    self.done = False
    self.turns = 0
    self.tool_calls = 0
    self.output_tokens = 0
    self.thinking_tokens = 0
    self.tool_use_tokens = 0
    self.answer_tokens = 0
    self.thought_text = ""
    self.visible_text = ""
    self.turn_logs: list[dict[str, Any]] = []
    self.prompt_tokens = 0
    self.max_prompt_tokens = 0
    self.eos_continues = 0
    self.nudge_pending = False
    self.force_submit_done = False
    self.forced_repair_rounds = 0
    self.post_submit_audited = False
    self.last_turn_sanitized = False
    self.commit_notes = 0  # P2: number of commit notes / commit nudges injected
    self.think_nudges = 0
    self.ctx_cutoff = False  # attempt ended because max_model_len headroom ran out
    self.start_time = time.time()
    self.elapsed_s = 0.0

  def as_vllm_input(self) -> Any:
    """Formats the current transcript (and optional PIL image) for `llm.generate()`."""
    if self.use_images and self.pil_image is not None:
      return {
          "prompt": self.transcript,
          "multi_modal_data": {"image": self.pil_image},
      }
    return self.transcript

  def finalize_record(self) -> dict[str, Any]:
    """Harvests all workspace candidates, inverts TTA if enabled, and returns attempt summary."""
    raw_cands = self.sandbox.harvest_workspace_candidates()
    canonical_cands: list[dict[str, Any]] = []
    for c in raw_cands:
      inv_g = self.tta.invert_grid(c["grid"])
      if inv_g is not None:
        c_copy = dict(c)
        c_copy["grid"] = inv_g
        canonical_cands.append(c_copy)

    # Evaluate `solve(grid)` on all sibling test pairs of the same task before clearing ns
    cross_pair_cands: dict[int, dict[str, Any]] = {}
    solve_fn = self.sandbox.ns.get("solve")
    all_tests = self.raw_task.get("test", [])
    if callable(solve_fn) and len(all_tests) > 1:
      for other_pidx, other_tpair in enumerate(all_tests):
        if other_pidx == self.pair_idx:
          continue
        try:
          tta_other_in = self.tta.apply_grid(other_tpair["input"])
          raw_other_out = solve_fn(copy.deepcopy(tta_other_in))
          coerced_other = sandbox.coerce_grid(raw_other_out)
          if coerced_other is not None:
            audit_ok, _ = sandbox.audit_test_grid(
                self.tta_task, tta_other_in, coerced_other
            )
            inv_other = self.tta.invert_grid(coerced_other)
            if inv_other is not None:
              cross_pair_cands[other_pidx] = {
                  "grid": inv_other,
                  "source": (
                      f"cross_pair_from_p{self.pair_idx}_att{self.attempt_idx}"
                  ),
                  "train_pass": bool(self.sandbox.train_passed),
                  "train_frac": round(self.sandbox.best_train_frac, 4),
                  "cell_frac": round(self.sandbox.best_cell_frac, 4),
                  "suspect": not audit_ok,
                  "is_primary": bool(self.sandbox.train_passed and audit_ok),
              }
        except Exception:  # pylint: disable=broad-exception-caught
          pass

    pred_grid: list[list[int]] | None = None
    if self.sandbox.primary_grid is not None:
      pred_grid = self.tta.invert_grid(self.sandbox.primary_grid)
    elif canonical_cands:
      pred_grid = canonical_cands[0]["grid"]

    exact_match = pred_grid is not None and pred_grid == self.target_output
    shape_match = (
        pred_grid is not None
        and len(pred_grid) == len(self.target_output)
        and len(pred_grid[0]) == len(self.target_output[0])
    )
    cell_acc = 0.0
    if shape_match and pred_grid is not None:
      h, w = len(self.target_output), len(self.target_output[0])
      cell_acc = sum(
          1
          for r in range(h)
          for c in range(w)
          if pred_grid[r][c] == self.target_output[r][c]
      ) / max(1, h * w)

    self.sandbox.ns.clear()
    self.pil_image = None
    first_v = self.tta.invert_grid(self.sandbox.first_verified_grid)
    last_v = self.tta.invert_grid(self.sandbox.last_verified_grid)
    return {
        "attempt_idx": self.attempt_idx,
        "tta_d8_idx": self.tta.d8_idx,
        "exact_match": exact_match,
        "shape_match": shape_match,
        "cell_accuracy": round(cell_acc, 4),
        "parsed_ok": pred_grid is not None,
        "train_passed": bool(self.sandbox.train_passed),
        "best_train_frac": round(self.sandbox.best_train_frac, 4),
        "best_cell_frac": round(self.sandbox.best_cell_frac, 4),
        "turns": self.turns,
        "tool_calls": self.tool_calls,
        "prompt_tokens": self.prompt_tokens,
        "output_tokens": self.output_tokens,
        "thinking_tokens": self.thinking_tokens,
        "tool_use_tokens": self.tool_use_tokens,
        "answer_tokens": self.answer_tokens,
        "thought_text": self.thought_text,
        "visible_text": self.visible_text,
        "turn_logs": self.turn_logs,
        "pred_grid": pred_grid,
        "candidates": canonical_cands,
        "cross_pair_cands": cross_pair_cands,
        # Audit / non-destructive bookkeeping (for offline policy re-scoring)
        "audited": bool(self.post_submit_audited),
        "num_verified_submissions": self.sandbox.num_verified_submissions,
        "frozen_overrides": self.sandbox.frozen_overrides,
        "first_verified_grid": first_v,
        "last_verified_grid": last_v,
        "first_verified_exact": (
            first_v is not None and first_v == self.target_output
        ),
        "last_verified_exact": (
            last_v is not None and last_v == self.target_output
        ),
        # Pilot diagnostics (P1/P2/P3 & Axis 1-3)
        "ever_defined_solve": bool(self.sandbox.ever_defined_solve),
        "num_verifications": int(self.sandbox.num_verifications),
        "num_auto_checks": int(self.sandbox.num_auto_checks),
        "repair_rounds_used": int(self.sandbox.repair_rounds_used),
        "repaired_to_pass": bool(self.sandbox.repaired_to_pass),
        "forced_repair_rounds": int(self.forced_repair_rounds),
        "num_candidates": len(self.sandbox.candidates),
        "commit_notes": int(self.commit_notes),
        "think_nudges": int(self.think_nudges),
        "ctx_cutoff": bool(self.ctx_cutoff),
        "max_prompt_tokens": int(self.max_prompt_tokens),
        "elapsed_s": round(time.time() - self.start_time, 1),
        "transcript": self.transcript if FLAGS.save_transcripts else None,
    }


def _commit_due(s: "AttemptState") -> bool:
  """P2: True when the attempt should be pushed to commit a `def solve(grid)`."""
  return (
      FLAGS.commit_after_tokens > 0
      and s.output_tokens >= FLAGS.commit_after_tokens
      and not s.sandbox.ever_defined_solve
      and not s.sandbox.train_passed
  )


def run_attempt_wave_batched(
    llm: Any,
    vllm_mod: Any,
    states: list[AttemptState],
    wave_desc: str,
) -> tuple[int, float]:
  """Runs a batch of AttemptStates through lockstep vLLM generation + parallel Python execution.

  Returns `(total_generated_tokens_in_wave, generation_wall_seconds)`.
  """
  if not states:
    return 0, 0.0

  total_wave_gen_toks = 0
  total_wave_gen_s = 0.0
  tokenizer = llm.get_tokenizer()

  def _count_prompt_tokens(st: AttemptState) -> int:
    try:
      toks = len(tokenizer.encode(st.transcript))
    except Exception:  # pylint: disable=broad-exception-caught
      toks = int(len(st.transcript) // 2.0)
    return max(
        st.prompt_tokens + 256,
        toks + (2048 if st.use_images else 128),
    )

  def _count_sub_tokens(txt: str) -> int:
    if not txt:
      return 0
    try:
      return len(tokenizer.encode(txt))
    except Exception:  # pylint: disable=broad-exception-caught
      return max(1, len(txt) // 3)

  def _record_turn_stats(
      st: AttemptState,
      raw_text: str,
      n_tok: int,
      finish_reason: str,
      is_forced_tool: bool = False,
      forced_fn_name: str = "",
  ) -> None:
    if is_forced_tool:
      think_t, vis_t = "", ""
      think_toks, tool_toks, ans_toks = 0, n_tok, 0
    else:
      think_t, tool_t, vis_t = prompts_and_voting.split_turn_thinking_and_tool(
          raw_text
      )
      if tool_t:
        tool_toks = min(n_tok, _count_sub_tokens(tool_t))
        think_toks = max(0, n_tok - tool_toks)
        ans_toks = 0
      elif vis_t:
        ans_toks = min(n_tok, _count_sub_tokens(vis_t))
        think_toks = max(0, n_tok - ans_toks)
        tool_toks = 0
      else:
        think_toks = n_tok
        tool_toks = 0
        ans_toks = 0

    st.thinking_tokens += think_toks
    st.tool_use_tokens += tool_toks
    st.answer_tokens += ans_toks
    if think_t:
      st.thought_text = (
          f"{st.thought_text}\n\n[Turn {st.turns}]\n{think_t}"
          if st.thought_text
          else f"[Turn {st.turns}]\n{think_t}"
      )
    if vis_t:
      st.visible_text = (
          f"{st.visible_text}\n\n{vis_t}" if st.visible_text else vis_t
      )
    st.turn_logs.append({
        "turn": st.turns,
        "output_tokens": n_tok,
        "thinking_tokens": think_toks,
        "tool_use_tokens": tool_toks,
        "answer_tokens": ans_toks,
        "finish_reason": finish_reason,
        "forced_fn": forced_fn_name or None,
    })

  # Case A: Experiment 1 (`no_tools`) — single batched `llm.generate()` call
  if FLAGS.mode == "no_tools":
    prompts = [s.as_vllm_input() for s in states]
    params_list = [
        vllm_mod.SamplingParams(
            temperature=FLAGS.temperature,
            top_p=FLAGS.top_p,
            top_k=FLAGS.top_k,
            repetition_penalty=FLAGS.repetition_penalty,
            max_tokens=max(
                1024,
                min(
                    FLAGS.total_token_budget,
                    FLAGS.max_model_len - _count_prompt_tokens(s) - 512,
                ),
            ),
        )
        for s in states
    ]
    t0 = time.perf_counter()
    try:
      outputs = llm.generate(prompts, params_list, use_tqdm=True)
    except Exception as mm_exc:  # pylint: disable=broad-exception-caught
      print(
          f"[WARN] Multimodal generate failed ({mm_exc}); falling back to"
          " text-only.",
          flush=True,
      )
      for s in states:
        s.use_images = False
        s.transcript = s.transcript.replace(
            "<|vision_start|><|image_pad|><|vision_end|>\n", ""
        )
      outputs = llm.generate(
          [s.transcript for s in states], params_list, use_tqdm=True
      )
    gen_s = time.perf_counter() - t0
    total_wave_gen_s += gen_s

    for s, out in zip(states, outputs):
      gen_out = out.outputs[0]
      text = gen_out.text
      n_tok = len(gen_out.token_ids)
      s.prompt_tokens = len(out.prompt_token_ids)
      s.output_tokens += n_tok
      s.turns = 1
      total_wave_gen_toks += n_tok
      s.transcript += text
      if not s.transcript.endswith("<|im_end|>\n"):
        s.transcript += "<|im_end|>\n"
      _record_turn_stats(s, text, n_tok, str(gen_out.finish_reason))
      parsed = prompts_and_voting.parse_last_grid(text)
      if parsed is not None:
        s.sandbox._record_candidate(  # pylint: disable=protected-access
            grid=parsed,
            source="no_tools_json",
            train_pass=None,
            train_frac=None,
            cell_frac=0.0,
            suspect=False,
            is_primary=True,
        )
      s.done = True
    return total_wave_gen_toks, total_wave_gen_s

  # Case B: Experiments 2, 3, 4 (`tools` mode) — multi-turn lockstep loop
  total_max_steps = FLAGS.max_turns + max(0, FLAGS.max_repair_rounds)
  turn_pbar = tqdm.tqdm(
      total=total_max_steps,
      desc=f"  {wave_desc} turns",
      leave=False,
      dynamic_ncols=True,
  )
  # Reserve ~1500 tokens per repair round so forced submission happens early enough to allow up to max_repair_rounds repairs
  repair_reserve = max(1536, 1536 * (1 + max(0, FLAGS.max_repair_rounds)))
  for turn_idx in range(total_max_steps):
    active = [s for s in states if not s.done]
    if not active:
      break

    prompts: list[Any] = []
    params_list: list[Any] = []
    for s in active:
      # Exact tokenizer check so prompt_tokens + max_tokens never exceeds max_model_len
      est_prompt_toks = _count_prompt_tokens(s)
      s.max_prompt_tokens = max(s.max_prompt_tokens, est_prompt_toks)
      ctx_headroom = FLAGS.max_model_len - est_prompt_toks - 512
      if ctx_headroom <= 256:
        s.ctx_cutoff = True
        s.done = True
        continue

      rem = min(s.hard_budget - s.output_tokens, ctx_headroom)
      if rem <= 256:
        s.done = True
        continue

      # If we already did a forced submit and it failed demonstrations, allow up to `FLAGS.max_repair_rounds` repair rounds!
      if s.force_submit_done:
        if (
            not s.sandbox.train_passed
            and s.forced_repair_rounds < FLAGS.max_repair_rounds
            and rem >= 512
        ):
          s.forced_repair_rounds += 1
          step_max = min(2048, rem)
          prompts.append(s.as_vllm_input())
          params_list.append(
              vllm_mod.SamplingParams(
                  temperature=0.4,
                  top_p=FLAGS.top_p,
                  top_k=FLAGS.top_k,
                  repetition_penalty=FLAGS.repetition_penalty,
                  max_tokens=step_max,
              )
          )
        else:
          s.done = True
        continue

      is_main_turns_exhausted = (
          turn_idx >= FLAGS.max_turns - 1 and not s.sandbox.train_passed
      )
      if (
          rem <= (repair_reserve if not s.sandbox.ever_defined_solve else 1536)
          or is_main_turns_exhausted
      ):
        # Out of main exploration budget or on final main turn -> force a `submit_program` call
        if ctx_headroom >= 640:
          s.force_submit_done = True
          s.transcript += (
              prompts_and_voting.CODER_NEXT_FORCE_SUBMIT_PROMPT
              if s.coder_next_format
              else prompts_and_voting.FORCE_SUBMIT_PROMPT
          )
          prompts.append(s.as_vllm_input())
          params_list.append(
              vllm_mod.SamplingParams(
                  temperature=0.2,
                  top_p=FLAGS.top_p,
                  top_k=FLAGS.top_k,
                  repetition_penalty=FLAGS.repetition_penalty,
                  max_tokens=min(1536, rem - 64),
                  stop=["</parameter>", "</tool_call>"],
              )
          )
        else:
          s.done = True
        continue

      if s.nudge_pending:
        # Turn just got nudged into `<tool_call><function=run_python>...<parameter=code>\n`
        step_max = min(2048, rem)
        prompts.append(s.as_vllm_input())
        params_list.append(
            vllm_mod.SamplingParams(
                temperature=FLAGS.temperature,
                top_p=FLAGS.top_p,
                top_k=FLAGS.top_k,
                repetition_penalty=FLAGS.repetition_penalty,
                max_tokens=step_max,
                stop=["</parameter>"],
            )
        )
      else:
        step_max = min(FLAGS.turn_think_cap, rem)
        prompts.append(s.as_vllm_input())
        params_list.append(
            vllm_mod.SamplingParams(
                temperature=FLAGS.temperature,
                top_p=FLAGS.top_p,
                top_k=FLAGS.top_k,
                repetition_penalty=FLAGS.repetition_penalty,
                max_tokens=step_max,
            )
        )

    # Re-filter `active` in case some states hit `s.done = True` above
    active = [s for s in active if not s.done]
    if not active:
      break

    t_gen0 = time.perf_counter()
    try:
      outputs = llm.generate(prompts, params_list, use_tqdm=False)
    except Exception as mm_exc:  # pylint: disable=broad-exception-caught
      print(
          f"[WARN] Batch generate raised ({mm_exc}); retrying text-only with"
          " clamped max_tokens.",
          flush=True,
      )
      safe_active: list[AttemptState] = []
      safe_prompts: list[str] = []
      safe_params: list[Any] = []
      for s, p in zip(active, params_list):
        s.use_images = False
        s.transcript = s.transcript.replace(
            "<|vision_start|><|image_pad|><|vision_end|>\n", ""
        )
        if len(s.transcript) // 3 < FLAGS.max_model_len - 2048:
          p.max_tokens = min(p.max_tokens, 1024)
          safe_active.append(s)
          safe_prompts.append(s.transcript)
          safe_params.append(p)
        else:
          s.done = True
      active = safe_active
      if not active:
        break
      outputs = llm.generate(safe_prompts, safe_params, use_tqdm=False)
    gen_s = time.perf_counter() - t_gen0
    total_wave_gen_s += gen_s

    # Collect sandbox executions to run in parallel threads
    to_exec: list[tuple[AttemptState, list[tuple[str, str]]]] = []

    for s, out in zip(active, outputs):
      gen_out = out.outputs[0]
      text = gen_out.text
      n_tok = len(gen_out.token_ids)
      finish_reason = str(gen_out.finish_reason)
      s.prompt_tokens = len(out.prompt_token_ids)
      s.output_tokens += n_tok
      s.turns += 1
      s.last_turn_sanitized = False
      total_wave_gen_toks += n_tok

      # 1. Handle initial forced submit completion (when forced_repair_rounds == 0)
      if s.force_submit_done and s.forced_repair_rounds == 0:
        _record_turn_stats(
            s,
            text,
            n_tok,
            finish_reason,
            is_forced_tool=True,
            forced_fn_name="submit_program",
        )
        code, _ = prompts_and_voting.strip_excessive_comments(
            text.split("</parameter>")[0].strip()
        )
        s.transcript += (
            f"{code}\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        )
        if code:
          to_exec.append((s, [("submit_program", code)]))
          # If max_repair_rounds == 0, finish immediately; otherwise _run_one will check if train_passed and keep alive for repair if needed!
          if FLAGS.max_repair_rounds <= 0:
            s.done = True
        else:
          s.done = True
        continue

      # 2. Handle completion after `THINK_NUDGE` (where `<parameter=code>\n` was prefilled)
      if s.nudge_pending:
        s.nudge_pending = False
        _record_turn_stats(
            s,
            text,
            n_tok,
            finish_reason,
            is_forced_tool=True,
            forced_fn_name="run_python",
        )
        code, _ = prompts_and_voting.strip_excessive_comments(
            text.split("</parameter>")[0].strip()
        )
        s.transcript += (
            f"{code}\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        )
        if code:
          to_exec.append((s, [("run_python", code)]))
        else:
          s.done = True
        continue

      # 3. Standard generation step (sanitize comment loops / truncated tool calls before appending to transcript)
      if s.coder_next_format:
        sanitized_text, had_loop_or_trunc = (
            prompts_and_voting.sanitize_turn_for_transcript(text, finish_reason)
        )
        s.last_turn_sanitized = had_loop_or_trunc
      else:
        sanitized_text = text

      _record_turn_stats(s, sanitized_text, n_tok, finish_reason)
      s.transcript += sanitized_text
      calls = prompts_and_voting.parse_tool_calls(sanitized_text)
      if calls:
        if not s.transcript.endswith("<|im_end|>\n"):
          s.transcript += "<|im_end|>\n"
        to_exec.append((s, calls))
        continue

      # 4. No tool calls emitted yet: check if cut due to `turn_think_cap`
      if finish_reason == "length" and (
          s.coder_next_format or "</think>" not in text
      ):
        if s.output_tokens + 1024 < s.hard_budget:
          # Inject THINK_NUDGE in model's voice and force a `run_python` call next step!
          s.think_nudges += 1
          commit_due = _commit_due(s)
          if s.coder_next_format:
            nudge = prompts_and_voting.CODER_NEXT_THINK_NUDGE
          elif commit_due:
            nudge = prompts_and_voting.COMMIT_THINK_NUDGE
            s.commit_notes += 1
          else:
            nudge = prompts_and_voting.THINK_NUDGE
          s.transcript += nudge
          s.nudge_pending = True
          continue
        else:
          s.done = True
          continue

      # 5. If post-submit audit was already triggered and model finished without calling tools (e.g. CONFIRMED), finish!
      if s.post_submit_audited and s.sandbox.train_passed:
        s.done = True
        continue

      # 5b. Model stopped without a tool call: check if it wrote a JSON grid
      parsed = prompts_and_voting.parse_last_grid(text)
      if parsed is not None:
        s.sandbox._record_candidate(  # pylint: disable=protected-access
            grid=parsed,
            source="visible_json",
            train_pass=None,
            train_frac=None,
            cell_frac=0.0,
            suspect=False,
            is_primary=(s.sandbox.primary_grid is None),
        )
        s.done = True
        continue

      # 6. Premature EOS or text-only reply without tool call -> continue up to 3 times
      if s.eos_continues < 3 and s.output_tokens + 1024 < s.hard_budget:
        s.eos_continues += 1
        if s.coder_next_format or "</think>" in text:
          if not s.transcript.endswith("<|im_end|>\n"):
            s.transcript += "<|im_end|>\n"
          bnote = prompts_and_voting.format_budget_note(
              s.output_tokens, s.soft_target, s.hard_budget
          )
          asst_prefix = (
              "<|im_start|>assistant\n"
              if s.coder_next_format
              else "<|im_start|>assistant\n<think>\n"
          )
          s.transcript += (
              "<|im_start|>user\nPlease call `run_python` (with"
              " `<parameter=reasoning>` then `<parameter=code>`) to test your"
              " rule or `submit_program` with `def solve(grid):`."
              f"{bnote}<|im_end|>\n{asst_prefix}"
          )
        continue
      s.done = True

    # Execute all tool calls in parallel across CPU threads
    if to_exec:

      def _run_one(item: tuple[AttemptState, list[tuple[str, str]]]) -> None:
        st, call_list = item
        outs: list[str] = []
        if st.last_turn_sanitized:
          outs.append(
              "[NOTE: Excessive inline `#` comments or truncated lines were"
              " stripped from your previous tool call. Put all reasoning in"
              " `<parameter=reasoning>` and keep `<parameter=code>` under 50"
              " lines of executable Python.]"
          )
        for fn_name, code_str in call_list:
          st.tool_calls += 1
          res_str = st.sandbox.execute_turn(fn_name, code_str)
          outs.append(res_str)
          if st.sandbox.train_passed:
            if (
                FLAGS.audit_on_submit
                and not st.post_submit_audited
                and st.output_tokens + 2048 < st.hard_budget
            ):
              st.post_submit_audited = True
              confirm_instr = (
                  "reply `CONFIRMED` (no tool call needed)"
                  if st.coder_next_format
                  else "reply `CONFIRMED` after `</think>` (no tool call needed)"
              )
              outs.append(
                  "AUDIT CHECK BEFORE LOCK-IN: Inspect the `solve(test_input)`"
                  " grid shown above against `test_input`:\n1. Did your"
                  " `solve(grid)` hardcode any color, row/col coordinate, or"
                  " grid size from `train`?\n2. Did your rule rely on a"
                  " coincidence in `train` (e.g., sorting objects by `min_row`"
                  " when the true ordering is spatial connectivity/chain"
                  " matching, or assuming fixed offsets)?\n3. Are all colors"
                  " and objects in `test_input` properly handled?\nIf"
                  f" `solve(test_input)` is 100% general and correct, {confirm_instr}."
                  " If you spot any issue on `test_input`, fix `def"
                  " solve(grid):` and call `submit_program` again."
              )
              break
            st.done = True
            break
        if not st.done:
          if (
              st.force_submit_done
              and st.forced_repair_rounds >= FLAGS.max_repair_rounds
          ):
            st.done = True
            return
          combined = "\n---\n".join(outs)
          bnote = prompts_and_voting.format_budget_note(
              st.output_tokens, st.soft_target, st.hard_budget
          )
          if _commit_due(st):
            bnote += prompts_and_voting.format_commit_note(st.output_tokens)
            st.commit_notes += 1
          if st.force_submit_done and not st.sandbox.train_passed:
            bnote += (
                f"\n[REPAIR ROUND {st.forced_repair_rounds + 1}/{FLAGS.max_repair_rounds}:"
                " Your submitted `solve(grid)` did not pass all demonstrations."
                " Reconsider both your implementation and your underlying rule"
                " using the exact demonstration diffs above, and immediately"
                " call `submit_program` with a revised `def solve(grid):`.]"
            )
          asst_prefix = (
              "<|im_start|>assistant\n"
              if st.coder_next_format
              else "<|im_start|>assistant\n<think>\n"
          )
          st.transcript += (
              f"<|im_start|>user\n<tool_response>\n{combined}{bnote}\n"
              f"</tool_response><|im_end|>\n{asst_prefix}"
          )

      with futures.ThreadPoolExecutor(
          max_workers=min(32, len(to_exec))
      ) as pool:
        list(pool.map(_run_one, to_exec))

    rem_active = sum(1 for s in states if not s.done)
    passed_so_far = sum(1 for s in states if s.sandbox.train_passed)
    tok_per_s = total_wave_gen_toks / max(total_wave_gen_s, 1e-6)
    turn_pbar.set_postfix(
        active=f"{rem_active}/{len(states)}",
        train_ok=passed_so_far,
        tok_s=f"{tok_per_s:.0f}",
    )
    turn_pbar.update(1)
    if rem_active == 0:
      break

  turn_pbar.close()
  return total_wave_gen_toks, total_wave_gen_s


def compute_summary(
    rows: list[dict[str, Any]],
    total_gen_tokens: int,
    total_gen_seconds: float,
    wall_elapsed_s: float,
    num_tpus: int,
    device_kind: str,
) -> dict[str, Any]:
  """Computes live Kaggle (167 pairs) & ARC Prize (120 tasks) pass@1, pass@2, oracle, and tok/s metrics."""
  n = len(rows)
  if n == 0:
    return {"pairs_completed": 0}

  p1_pairs = sum(1 for r in rows if r["pass_at_1"])
  p2_pairs = sum(1 for r in rows if r["pass_at_2"])
  oracle_pairs = sum(
      1
      for r in rows
      if r["pass_at_2"]
      or any(
          bool(
              a.get("exact_match")
              or a.get("first_verified_exact")
              or a.get("last_verified_exact")
          )
          for a in r.get("attempts", [])
      )
      or any(
          bool(pv.get("pass_at_2") or pv.get("pass_at_1"))
          for pv in (r.get("policies") or {}).values()
      )
  )
  demo_perfect_pairs = sum(
      1
      for r in rows
      if any(bool(a.get("train_passed")) for a in r.get("attempts", []))
  )

  by_task: dict[str, list[dict[str, Any]]] = {}
  for r in rows:
    by_task.setdefault(r["task_id"], []).append(r)
  num_tasks = len(by_task)
  arc_prize_p1 = sum(
      sum(1.0 for r in trs if r["pass_at_1"]) / len(trs)
      for trs in by_task.values()
  )
  arc_prize_p2 = sum(
      sum(1.0 for r in trs if r["pass_at_2"]) / len(trs)
      for trs in by_task.values()
  )

  all_atts = [a for r in rows for a in r.get("attempts", [])]
  n_att = max(1, len(all_atts))
  sum_att_toks = sum(a.get("output_tokens", 0) for a in all_atts)
  sum_think_toks = sum(a.get("thinking_tokens", 0) for a in all_atts)
  sum_tool_toks = sum(a.get("tool_use_tokens", 0) for a in all_atts)
  sum_ans_toks = sum(a.get("answer_tokens", 0) for a in all_atts)

  # Per-policy metrics (2x2: {pooled, primary-only} x {latest, first-verified})
  policy_metrics: dict[str, Any] = {}
  for pol in prompts_and_voting.SELECTION_POLICIES:
    rows_p = [r for r in rows if pol in (r.get("policies") or {})]
    if not rows_p:
      continue
    pp1 = sum(1 for r in rows_p if r["policies"][pol].get("pass_at_1"))
    pp2 = sum(1 for r in rows_p if r["policies"][pol].get("pass_at_2"))
    by_t: dict[str, list[dict[str, Any]]] = {}
    for r in rows_p:
      by_t.setdefault(r["task_id"], []).append(r["policies"][pol])
    tp2 = sum(
        sum(1.0 for e in v if e.get("pass_at_2")) / len(v) for v in by_t.values()
    )
    policy_metrics[pol] = {
        "pairs": len(rows_p),
        "kaggle_pass_at_1_pct": round(100.0 * pp1 / len(rows_p), 2),
        "kaggle_pass_at_2_pct": round(100.0 * pp2 / len(rows_p), 2),
        "arc_prize_pass_at_2_pct": round(100.0 * tp2 / max(1, len(by_t)), 2),
    }
  audit_stats = {
      "attempts_audited": sum(1 for a in all_atts if a.get("audited")),
      "attempts_with_frozen_override": sum(
          1 for a in all_atts if a.get("frozen_overrides", 0) > 0
      ),
      "first_verified_exact": sum(
          1 for a in all_atts if a.get("first_verified_exact")
      ),
      "last_verified_exact": sum(
          1 for a in all_atts if a.get("last_verified_exact")
      ),
      "override_first_right_last_wrong": sum(
          1
          for a in all_atts
          if a.get("frozen_overrides", 0) > 0
          and a.get("first_verified_exact")
          and not a.get("last_verified_exact")
      ),
      "override_first_wrong_last_right": sum(
          1
          for a in all_atts
          if a.get("frozen_overrides", 0) > 0
          and not a.get("first_verified_exact")
          and a.get("last_verified_exact")
      ),
  }

  return {
      "exp_name": FLAGS.exp_name,
      "policy_metrics": policy_metrics,
      "audit_stats": audit_stats,
      "freeze_first_verified": FLAGS.freeze_first_verified,
      "audit_on_submit": FLAGS.audit_on_submit,
      "submission_policy": FLAGS.submission_policy,
      "pilot_config": {
          "auto_check_on_define": FLAGS.auto_check_on_define,
          "commit_after_tokens": FLAGS.commit_after_tokens,
          "max_turns": FLAGS.max_turns,
          "max_repair_rounds": FLAGS.max_repair_rounds,
          "structured_feedback": FLAGS.structured_feedback,
          "diverse_hypotheses": FLAGS.diverse_hypotheses,
          "structured_summary": FLAGS.structured_summary,
          "enable_color_tta": FLAGS.enable_color_tta,
          "turn_think_cap": FLAGS.turn_think_cap,
          "max_model_len": FLAGS.max_model_len,
          "task_ids_file": FLAGS.task_ids_file,
          "task_ids_key": FLAGS.task_ids_key,
          "n_task_ids": len([t for t in FLAGS.task_ids.split(",") if t.strip()]),
      },
      "pilot_diagnostics": {
          "attempts_with_any_verification_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("num_verifications", 0) > 0) / n_att, 2
          ),
          "attempts_ever_defined_solve_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("ever_defined_solve")) / n_att, 2
          ),
          "mean_verifications_per_attempt": round(
              sum(a.get("num_verifications", 0) for a in all_atts) / n_att, 2
          ),
          "mean_auto_checks_per_attempt": round(
              sum(a.get("num_auto_checks", 0) for a in all_atts) / n_att, 2
          ),
          "mean_repair_rounds_per_attempt": round(
              sum(a.get("repair_rounds_used", 0) for a in all_atts) / n_att, 2
          ),
          "attempts_repaired_to_pass_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("repaired_to_pass")) / n_att, 2
          ),
          "attempts_with_commit_note_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("commit_notes", 0) > 0) / n_att, 2
          ),
          "mean_think_nudges_per_attempt": round(
              sum(a.get("think_nudges", 0) for a in all_atts) / n_att, 2
          ),
          "attempts_hit_max_turns_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("turns", 0) >= FLAGS.max_turns) / n_att, 2
          ),
          "attempts_ctx_cutoff_pct": round(
              100.0 * sum(1 for a in all_atts if a.get("ctx_cutoff")) / n_att, 2
          ),
          "mean_max_prompt_tokens": round(
              sum(a.get("max_prompt_tokens", 0) for a in all_atts) / n_att, 1
          ),
          "mean_elapsed_s_per_attempt": round(
              sum(a.get("elapsed_s", 0.0) for a in all_atts) / n_att, 1
          ),
      },
      "mode": FLAGS.mode,
      "chat_format": (
          "qwen_coder_next" if is_coder_next_format() else "qwen3_think"
      ),
      "model_path": FLAGS.model_path,
      "tpu_hardware": f"{num_tpus}x {device_kind} (tp={FLAGS.tensor_parallel_size})",
      "num_attempts_config": FLAGS.num_attempts,
      "enable_tta": FLAGS.enable_tta,
      "use_images": FLAGS.use_images,
      "multi_hypothesis": FLAGS.multi_hypothesis,
      "total_token_budget": FLAGS.total_token_budget,
      "soft_token_target": FLAGS.soft_token_target,
      "turn_think_cap": FLAGS.turn_think_cap if FLAGS.mode == "tools" else None,
      "pairs_completed": n,
      "tasks_completed": num_tasks,
      "kaggle_pass_at_1_pairs": p1_pairs,
      "kaggle_pass_at_1_pct": round(100.0 * p1_pairs / n, 2),
      "kaggle_pass_at_2_pairs": p2_pairs,
      "kaggle_pass_at_2_pct": round(100.0 * p2_pairs / n, 2),
      "oracle_any_candidate_pairs": oracle_pairs,
      "oracle_any_candidate_pct": round(100.0 * oracle_pairs / n, 2),
      "demo_perfect_pair_pct": round(100.0 * demo_perfect_pairs / n, 2),
      "arc_prize_pass_at_1_score": round(arc_prize_p1, 2),
      "arc_prize_pass_at_1_pct": round(
          100.0 * arc_prize_p1 / max(1, num_tasks), 2
      ),
      "arc_prize_pass_at_2_score": round(arc_prize_p2, 2),
      "arc_prize_pass_at_2_pct": round(
          100.0 * arc_prize_p2 / max(1, num_tasks), 2
      ),
      "train_passed_attempt_pct": round(
          100.0 * sum(1 for a in all_atts if a.get("train_passed")) / n_att, 2
      ),
      "mean_attempts_run_per_pair": round(len(all_atts) / n, 2),
      "mean_turns_per_attempt": round(
          sum(a.get("turns", 0) for a in all_atts) / n_att, 2
      ),
      "mean_tool_calls_per_attempt": round(
          sum(a.get("tool_calls", 0) for a in all_atts) / n_att, 2
      ),
      "mean_output_tokens_per_attempt": round(sum_att_toks / n_att, 1),
      "mean_thinking_tokens_per_attempt": round(sum_think_toks / n_att, 1),
      "mean_tool_use_tokens_per_attempt": round(sum_tool_toks / n_att, 1),
      "mean_answer_tokens_per_attempt": round(sum_ans_toks / n_att, 1),
      "total_output_tokens": sum_att_toks,
      "total_thinking_tokens": sum_think_toks,
      "total_tool_use_tokens": sum_tool_toks,
      "total_answer_tokens": sum_ans_toks,
      "vllm_decode_tokens_per_sec": round(
          total_gen_tokens / max(total_gen_seconds, 1e-6), 2
      ),
      "end_to_end_tokens_per_sec": round(
          total_gen_tokens / max(wall_elapsed_s, 1e-6), 2
      ),
      "wall_elapsed_s": round(wall_elapsed_s, 1),
      "solved_tasks_pass_at_1": sorted(
          tid
          for tid, trs in by_task.items()
          if all(r["pass_at_1"] for r in trs)
      ),
      "solved_tasks_pass_at_2": sorted(
          tid
          for tid, trs in by_task.items()
          if all(r["pass_at_2"] for r in trs)
      ),
      "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
  }


def load_completed_rows(
    output_dir: str,
) -> dict[tuple[str, int], dict[str, Any]]:
  completed: dict[tuple[str, int], dict[str, Any]] = {}
  if not gfile.Exists(output_dir):
    return completed
  try:
    fnames = sorted(gfile.ListDir(output_dir))
  except Exception:
    return completed
  for fn in fnames:
    if fn.startswith("predictions") and fn.endswith(".jsonl"):
      fpath = os.path.join(output_dir, fn)
      try:
        with gfile.Open(fpath, "r") as f:
          for line in f:
            line = line.strip()
            if not line:
              continue
            row = json.loads(line)
            completed[(row["task_id"], int(row["pair_idx"]))] = row
      except Exception:
        pass
  return completed


def save_kaggle_submission(
    output_dir: str,
    evalset: dict[str, dict[str, Any]],
    completed_map: dict[tuple[str, int], dict[str, Any]],
) -> None:
  """Writes official Kaggle `submission.json` covering all tasks in `evalset`."""
  sub: dict[str, list[dict[str, Any]]] = {}
  for tid, task in sorted(evalset.items()):
    preds_for_task: list[dict[str, Any]] = []
    for pidx, tpair in enumerate(task["test"]):
      row = completed_map.get((tid, pidx))
      if row is not None:
        a1 = (
            row.get("submission_attempt_1")
            or row.get("attempt_1")
            or copy.deepcopy(tpair["input"])
        )
        a2 = row.get("submission_attempt_2") or row.get("attempt_2") or a1
      else:
        a1 = copy.deepcopy(tpair["input"])
        a2 = copy.deepcopy(tpair["input"])
      preds_for_task.append({"attempt_1": a1, "attempt_2": a2})
    sub[tid] = preds_for_task

  sub_path = os.path.join(output_dir, "submission.json")
  with gfile.Open(sub_path, "w") as f:
    json.dump(sub, f)


def main(argv: Sequence[str]) -> None:
  del argv
  # 1. Copy ARC-AGI-2 evaluation data to local /tmp before initializing JAX/PJRT
  local_data = f"/tmp/arc_agi_2_shard_{FLAGS.shard_idx}"
  if os.path.exists(local_data):
    shutil.rmtree(local_data)
  os.makedirs(local_data, exist_ok=True)
  t0 = time.time()
  r = subprocess.run(
      [
          "fileutil",
          "cp",
          "-R",
          "-parallelism=32",
          f"{FLAGS.cns_data_path}/evaluation",
          f"{local_data}/evaluation",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  print(
      f"Data copy rc={r.returncode} ({time.time()-t0:.1f}s)",
      flush=True,
  )
  evalset = {
      f[:-5]: json.load(open(f"{local_data}/evaluation/{f}"))
      for f in sorted(os.listdir(f"{local_data}/evaluation"))
      if f.endswith(".json")
  }
  if FLAGS.limit_tasks > 0:
    evalset = dict(list(evalset.items())[: FLAGS.limit_tasks])
  wanted_ids: list[str] = []
  if FLAGS.task_ids_file:
    with gfile.Open(FLAGS.task_ids_file, "r") as tf:
      obj = json.load(tf)
    wanted_ids = list(obj[FLAGS.task_ids_key] if isinstance(obj, dict) else obj)
  if FLAGS.task_ids:
    wanted_ids += [t.strip() for t in FLAGS.task_ids.split(",") if t.strip()]
  if wanted_ids:
    unknown = sorted(set(wanted_ids) - set(evalset))
    if unknown:
      raise ValueError(f"Unknown task ids requested: {unknown}")
    keep = set(wanted_ids)
    evalset = {k: v for k, v in evalset.items() if k in keep}
    print(
        f"Restricting evaluation to {len(evalset)} tasks from"
        f" --task_ids/--task_ids_file ({sum(len(t['test']) for t in evalset.values())} test pairs)",
        flush=True,
    )

  all_pairs: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  shard_pairs: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  for t_idx, (tid, task) in enumerate(sorted(evalset.items())):
    for pidx, tpair in enumerate(task["test"]):
      item = (tid, pidx, task, tpair)
      all_pairs.append(item)
      if FLAGS.shard_by_task and (t_idx % FLAGS.num_shards == FLAGS.shard_idx):
        shard_pairs.append(item)

  if not FLAGS.shard_by_task:
    shard_pairs = [
        item
        for idx, item in enumerate(all_pairs)
        if idx % FLAGS.num_shards == FLAGS.shard_idx
    ]

  gfile.MakeDirs(FLAGS.output_dir)
  shard_suffix = (
      f"_shard_{FLAGS.shard_idx:03d}of{FLAGS.num_shards:03d}"
      if FLAGS.num_shards > 1
      else ""
  )
  jsonl_path = os.path.join(
      FLAGS.output_dir, f"predictions{shard_suffix}.jsonl"
  )
  transcripts_path = os.path.join(
      FLAGS.output_dir, f"transcripts{shard_suffix}.jsonl"
  )
  summary_path = os.path.join(FLAGS.output_dir, "summary.json")

  completed_map = load_completed_rows(FLAGS.output_dir)
  todo_pairs = [
      item for item in shard_pairs if (item[0], item[1]) not in completed_map
  ]
  coder_next = is_coder_next_format()
  print(
      f"[{FLAGS.exp_name}] Mode={FLAGS.mode} | ChatFmt={'qwen_coder_next' if coder_next else 'qwen3_think'}"
      f" | Attempts={FLAGS.num_attempts} |"
      f" Budget={FLAGS.total_token_budget} | TTA={FLAGS.enable_tta} |"
      f" Images={FLAGS.use_images} | MultiHyp={FLAGS.multi_hypothesis} |"
      f" ShardByTask={FLAGS.shard_by_task} |"
      f" Shard={FLAGS.shard_idx}/{FLAGS.num_shards} | Total pairs={len(all_pairs)}"
      f" | Shard pairs={len(shard_pairs)} | Remaining={len(todo_pairs)}",
      flush=True,
  )

  if not todo_pairs:
    print("All pairs for this shard are already completed!", flush=True)
    return

  # 2. Initialize vLLM on TPU
  import jax  # pylint: disable=g-import-not-at-top
  import vllm  # pylint: disable=g-import-not-at-top

  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  tp_size = (
      FLAGS.tensor_parallel_size if FLAGS.tensor_parallel_size > 0 else num_tpus
  )
  t_init0 = time.time()
  extra_llm_kwargs: dict[str, Any] = {}
  if FLAGS.use_images or FLAGS.hybrid_modality:
    extra_llm_kwargs["limit_mm_per_prompt"] = {"image": 1, "video": 0}
    extra_llm_kwargs["mm_processor_kwargs"] = {
        "size": {"longest_edge": 1003520, "shortest_edge": 3136}
    }
  llm = vllm.LLM(
      model=FLAGS.model_path,
      tensor_parallel_size=tp_size,
      max_model_len=FLAGS.max_model_len,
      max_num_batched_tokens=FLAGS.max_num_batched_tokens,
      max_num_seqs=FLAGS.max_num_seqs,
      gpu_memory_utilization=FLAGS.gpu_memory_utilization,
      enable_prefix_caching=True,
      load_format="tpu_streaming_loader",
      **extra_llm_kwargs,
  )
  print(
      f"vLLM TPU engine initialized in {time.time() - t_init0:.1f}s on"
      f" {num_tpus}x {device_kind} (tp={tp_size})",
      flush=True,
  )

  # 3. Process `todo_pairs` in batches of `FLAGS.pair_batch_size` with tqdm progress bar
  t_eval_start = time.time()
  cum_gen_tokens = 0
  cum_gen_seconds = 0.0

  pbar = tqdm.tqdm(
      total=len(todo_pairs),
      desc=f"{FLAGS.exp_name} (shard {FLAGS.shard_idx})",
      dynamic_ncols=True,
  )

  # Group into batches while keeping sibling test pairs of the same task_id together
  batches: list[list[tuple[str, int, dict[str, Any], dict[str, Any]]]] = []
  if FLAGS.shard_by_task:
    cur_batch: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    idx = 0
    while idx < len(todo_pairs):
      tid0 = todo_pairs[idx][0]
      task_group: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
      while idx < len(todo_pairs) and todo_pairs[idx][0] == tid0:
        task_group.append(todo_pairs[idx])
        idx += 1
      if (
          cur_batch
          and len(cur_batch) + len(task_group) > FLAGS.pair_batch_size
      ):
        batches.append(cur_batch)
        cur_batch = []
      cur_batch.extend(task_group)
    if cur_batch:
      batches.append(cur_batch)
  else:
    for b_start in range(0, len(todo_pairs), FLAGS.pair_batch_size):
      batches.append(todo_pairs[b_start : b_start + FLAGS.pair_batch_size])

  for batch_idx, batch_items in enumerate(batches):
    attempts_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {
        (tid, pidx): [] for tid, pidx, _, _ in batch_items
    }

    # Run attempts in waves of `attempts_per_wave` (e.g. 2 at a time) so Best-of-8
    # can early-stop for any pair that already has `early_stop_verified_votes` agreeing train-passed programs!
    wave_step = max(1, FLAGS.attempts_per_wave)
    for att_start in range(0, FLAGS.num_attempts, wave_step):
      att_end = min(FLAGS.num_attempts, att_start + wave_step)
      wave_states: list[AttemptState] = []

      for tid, pidx, task, tpair in batch_items:
        existing_atts = attempts_by_pair[(tid, pidx)]
        if existing_atts and FLAGS.num_attempts > 2:
          _, _, meta_so_far = prompts_and_voting.select_top_two_predictions(
              existing_atts, fallback_grid=tpair["input"]
          )
          if (
              meta_so_far.get("top1_verified_votes", 0)
              >= FLAGS.early_stop_verified_votes
          ):
            continue

        for a_idx in range(att_start, att_end):
          att_use_images = (
              (a_idx % 2 == 1) if FLAGS.hybrid_modality else FLAGS.use_images
          )
          wave_states.append(
              AttemptState(
                  task_id=tid,
                  pair_idx=pidx,
                  attempt_idx=a_idx,
                  raw_task=task,
                  raw_test_input=tpair["input"],
                  target_output=tpair["output"],
                  use_tools=(FLAGS.mode == "tools"),
                  enable_tta=FLAGS.enable_tta,
                  use_images=att_use_images,
                  multi_hypothesis=FLAGS.multi_hypothesis,
                  soft_target=FLAGS.soft_token_target,
                  hard_budget=FLAGS.total_token_budget,
                  coder_next_format=coder_next,
              )
          )

      if not wave_states:
        break

      w_desc = f"batch {batch_idx + 1}/{len(batches)} att {att_start+1}..{att_end}"
      w_toks, w_secs = run_attempt_wave_batched(
          llm=llm,
          vllm_mod=vllm,
          states=wave_states,
          wave_desc=w_desc,
      )
      cum_gen_tokens += w_toks
      cum_gen_seconds += w_secs

      for s in wave_states:
        attempts_by_pair[(s.task_id, s.pair_idx)].append(s.finalize_record())
      del wave_states
      import gc  # pylint: disable=g-import-not-at-top
      gc.collect()

    # Finalize predictions for each pair in `batch_items` and write to CNS
    transcript_rows: list[dict[str, Any]] = []
    with gfile.Open(jsonl_path, "a") as jf:
      for tid, pidx, task, tpair in batch_items:
        del task
        att_recs = list(attempts_by_pair[(tid, pidx)])
        # Share verified/partial `solve(grid)` candidates from sibling test pairs of the same task_id
        for (other_tid, other_pidx), other_atts in attempts_by_pair.items():
          if other_tid == tid and other_pidx != pidx:
            for o_att in other_atts:
              cp_cand = o_att.get("cross_pair_cands", {}).get(pidx)
              if cp_cand is not None:
                att_recs.append({
                    "attempt_idx": 100 + other_pidx * 10 + o_att["attempt_idx"],
                    "pred_grid": cp_cand["grid"],
                    "train_passed": cp_cand["train_pass"],
                    "best_train_frac": cp_cand["train_frac"],
                    "best_cell_frac": cp_cand["cell_frac"],
                    "candidates": [cp_cand],
                })

        target_grid = tpair["output"]
        # Score every selection policy on the SAME attempts (paired comparison).
        policies = prompts_and_voting.score_selection_policies(
            att_recs, fallback_grid=tpair["input"], target=target_grid
        )
        # `pass_at_1/2` and `attempt_1/2` keep the legacy policy (pooled candidates,
        # destructive audit) so headline numbers stay comparable with Exp 1-6.
        legacy = policies["pooled_latest"]
        att1_grid, att2_grid, vote_meta = (
            legacy["attempt_1"], legacy["attempt_2"], legacy["meta"]
        )
        pass_1 = bool(legacy["pass_at_1"])
        pass_2 = bool(legacy["pass_at_2"])
        chosen = policies[FLAGS.submission_policy]

        compact_atts = []
        for a in attempts_by_pair[(tid, pidx)]:
          drop = {"cross_pair_cands", "transcript"}
          if not FLAGS.keep_candidates:
            drop.add("candidates")
          compact_atts.append({k: v for k, v in a.items() if k not in drop})
          if FLAGS.save_transcripts and a.get("transcript"):
            transcript_rows.append({
                "task_id": tid,
                "pair_idx": pidx,
                "attempt_idx": a["attempt_idx"],
                "exact_match": a.get("exact_match"),
                "train_passed": a.get("train_passed"),
                "output_tokens": a.get("output_tokens"),
                "thinking_tokens": a.get("thinking_tokens"),
                "tool_use_tokens": a.get("tool_use_tokens"),
                "answer_tokens": a.get("answer_tokens"),
                "turn_logs": a.get("turn_logs"),
                "thought_text": a.get("thought_text"),
                "visible_text": a.get("visible_text"),
                "pred_grid": a.get("pred_grid"),
                "transcript": a["transcript"],
            })
        row = {
            "task_id": tid,
            "pair_idx": pidx,
            "pass_at_1": pass_1,
            "pass_at_2": pass_2,
            "attempt_1": att1_grid,
            "attempt_2": att2_grid,
            "vote_meta": vote_meta,
            "submission_policy": FLAGS.submission_policy,
            "submission_attempt_1": chosen["attempt_1"],
            "submission_attempt_2": chosen["attempt_2"],
            "policies": {
                k: {
                    "pass_at_1": bool(v["pass_at_1"]),
                    "pass_at_2": bool(v["pass_at_2"]),
                    "meta": v["meta"],
                }
                for k, v in policies.items()
            },
            "attempts": compact_atts,
        }
        jf.write(json.dumps(row) + "\n")
    if transcript_rows:
      try:
        with gfile.Open(transcripts_path, "a") as tf:
          for trow in transcript_rows:
            tf.write(json.dumps(trow) + "\n")
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[WARN] Failed to write transcripts: {exc}", flush=True)

    # Reload across all shards & update live `summary.json` and `submission.json`
    completed_map = load_completed_rows(FLAGS.output_dir)
    all_rows = [
        completed_map[k] for k in sorted(completed_map.keys())
    ]
    summary = compute_summary(
        rows=all_rows,
        total_gen_tokens=cum_gen_tokens,
        total_gen_seconds=cum_gen_seconds,
        wall_elapsed_s=time.time() - t_eval_start,
        num_tpus=num_tpus,
        device_kind=device_kind,
    )
    if shard_suffix:
      shard_summary_path = os.path.join(
          FLAGS.output_dir, f"summary{shard_suffix}.json"
      )
      try:
        with gfile.Open(shard_summary_path, "w") as sf:
          json.dump(summary, sf, indent=2)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    try:
      with gfile.Open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
      save_kaggle_submission(FLAGS.output_dir, evalset, completed_map)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[WARN] Non-fatal CNS summary lock contention: {exc}", flush=True)

    pbar.update(len(batch_items))
    pbar.set_postfix(
        done=f"{summary['pairs_completed']}/{len(all_pairs)}",
        p1=f"{summary['kaggle_pass_at_1_pct']:.1f}%",
        p2=f"{summary['kaggle_pass_at_2_pct']:.1f}%",
        arc_p2=f"{summary['arc_prize_pass_at_2_pct']:.1f}%",
        tok_s=f"{summary['vllm_decode_tokens_per_sec']:.0f}",
    )

  pbar.close()
  completed_map = load_completed_rows(FLAGS.output_dir)
  all_rows = [completed_map[k] for k in sorted(completed_map.keys())]
  final_summary = compute_summary(
      rows=all_rows,
      total_gen_tokens=cum_gen_tokens,
      total_gen_seconds=cum_gen_seconds,
      wall_elapsed_s=time.time() - t_eval_start,
      num_tpus=num_tpus,
      device_kind=device_kind,
  )
  print("=" * 80, flush=True)
  print(f"FINAL SUMMARY ({FLAGS.exp_name}):", flush=True)
  print(json.dumps(final_summary, indent=2), flush=True)
  print("=" * 80, flush=True)


if __name__ == "__main__":
  app_runner.run(main)
