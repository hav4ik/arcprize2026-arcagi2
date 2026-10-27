"""Generates Pass@1 DeepSeek-V4.1-Flash-1017 SFT Rollouts on Top-10K ARC Candidates (`v8` Best Setting).

Uses the best-scoring Pass@1 model & harness configuration (`v8`, `http://xid/304081446`,
59.28% single-attempt Pass@1 / 61.68% ranked Pass@1 / 70.66% Pass@2 on ARC-AGI-2 Eval):
1. Full-Precision `BF16 -> FP32` Native JAX/Pallas TPU Megakernel (`NativeDeepSeekV41FullPrecisionLLM`)
   with lossless speculative rejection + residual distribution Gumbel-max sampling
   (`temperature=1.0, top_k=20, top_p=0.95`).
2. Collaborator Vu (`havu`) `E030` `sub80s40` prompt & budget configuration:
   - `ARC_PRIZE_TEMPLATE` + `HAVU_LEAN2SUB_SUFFIX`
   - `HAVU_EFFORT_HEADER_MINIMAL` with `reasoning_effort=0` (no conflicting numeric effort header)
   - `HAVU_TOOLS_SPEC` (`run_python` and `submit_answer`)
   - `max_turns=40`, `max_turn_tokens=64000`, `max_reasoning_tokens=0`,
     `total_token_budget=80000`, `soft_token_target=40000`, `max_model_len=65536`,
     `compact_at_tokens=60000`, `turns_note_threshold=5`
   - `E043` `--token-exact` (`token_exact=True`) and `--rescue-code` (`rescue_code=True`)
3. Isolated persistent child-process IPython sandbox (`HavuSubprocessSession`, `style="ipython"`)
   with 3-tier session fallback (`answer` -> `solve(test_input)` -> `salvage_answer`) and
   in-session 3-level verification (`train` + `test` + `extra_verification_pairs`).
4. Outputs in the exact 24-column `E031` Parquet + JSONL rollout format consumed by
   `//experimental/users/arazdai/kaggle/sft:audit_and_prepare_data` and `data_pipeline.py`
   for Qwen3.8 SFT training (`train_sft.py` / `train_full_sft.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
import copy
import dataclasses
import hashlib
import json
import os
import time
from typing import Any

# Start the pre-TPU Zygote fork server inside `deepseek_v41_arc2_eval` BEFORE JAX/libtpu init.
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_arc2_eval as ds_eval

from absl import flags
import jax
import tqdm
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_tpu_vllm_traces as ds_traces
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_encoding
from google3.pyglib import gfile

try:
  import pyarrow as pa  # pylint: disable=g-import-not-at-top
  import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

  _HAS_PYARROW = True
except ImportError:
  pa = None
  pq = None
  _HAS_PYARROW = False

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "eval_contamination_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "Path to official ARC-AGI-2 120 evaluation tasks for contamination filtering.",
)
flags.DEFINE_string(
    "batch_label",
    "top10k_v8_pass1",
    "Batch label written to the E031 `batch` column.",
)
flags.DEFINE_integer(
    "parquet_flush_every_batches",
    5,
    "Write atomic Parquet snapshots every N waves (and at shard completion).",
)

# Exact 24 columns from Vu's E031 Parquet files (`e031_arc2_train.parquet`, `e031_nvarc_b0.parquet`)
E031_CORE_COLUMNS: tuple[str, ...] = (
    "task_id",
    "batch",
    "source",
    "parent",
    "pair",
    "attempt",
    "correct",
    "train_pass",
    "train_frac",
    "answer_source",
    "stop_reason",
    "error",
    "pred",
    "turns",
    "tool_calls",
    "prompt_tokens",
    "cache_hit_tokens",
    "completion_tokens",
    "seconds",
    "est_cost_usd",
    "model",
    "provider",
    "messages",
    "events",
)

# Qwen3.8 / ChatML markup tokens that must never leak into assistant reasoning or content
LEAK_MARKERS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
)

KNOWN_CONTAMINATED_TASKS: frozenset[str] = frozenset({"2753e76c", "81c0276b"})


@dataclasses.dataclass
class RolloutAttemptState:
  """Lightweight per-attempt state tracker for Top-10K rollout waves."""

  task_id: str
  pair_idx: int
  attempt_idx: int
  soft_target: int
  hard_budget: int
  transcript: str = ""
  done: bool = False
  turns: int = 0
  tool_calls: int = 0
  output_tokens: int = 0
  thinking_tokens: int = 0
  tool_use_tokens: int = 0
  answer_tokens: int = 0
  thought_text: str = ""
  visible_text: str = ""
  turn_logs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  prompt_tokens: int = 0
  max_prompt_tokens: int = 0
  ctx_cutoff: bool = False


def get_parent_puzzle_id(task_id: str, parent_col: str | None = None) -> str:
  """Extracts the canonical parent puzzle ID matching `sft/data_pipeline.py`."""
  if parent_col and isinstance(parent_col, str) and parent_col.strip():
    return parent_col.strip().split("_")[0]
  return str(task_id).strip().split("_")[0]


def load_eval_task_ids(eval_dir: str) -> set[str]:
  """Loads the set of 120 official ARC-AGI-2 evaluation task IDs."""
  eval_ids: set[str] = set()
  if eval_dir and gfile.Exists(eval_dir):
    try:
      for fname in gfile.ListDir(eval_dir):
        if fname.endswith(".json"):
          eval_ids.add(fname[:-5])
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return eval_ids


def is_task_contaminated(
    task_id: str, parent_id: str | None, eval_task_ids: set[str]
) -> tuple[bool, str]:
  """Checks if a task or its parent overlaps with ARC-AGI-2 evaluation tasks or known derivatives."""
  base_id = get_parent_puzzle_id(task_id, parent_id)
  if task_id in KNOWN_CONTAMINATED_TASKS or base_id in KNOWN_CONTAMINATED_TASKS:
    return True, f"known_d8_color_derivative({base_id})"
  if task_id in eval_task_ids or base_id in eval_task_ids:
    return True, f"direct_eval_task_id({base_id})"
  return False, ""


def sanitize_sft_text(text: str | None) -> str:
  """Removes any accidental Qwen ChatML/tool markup markers so `audit_rollout_messages` never trips."""
  if not text:
    return ""
  cleaned = text
  for marker in LEAK_MARKERS:
    if marker in cleaned:
      cleaned = cleaned.replace(marker, "")
  return cleaned


def audit_rollout_messages(
    messages: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> list[str]:
  """Exact trajectory audit from `google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py`."""
  bad_reasons: list[str] = []
  ev_idx = 0
  for msg in messages:
    if msg.get("role") != "assistant":
      continue
    ev = events[ev_idx] if ev_idx < len(events) else {}
    ev_idx += 1
    if ev.get("finish_reason") == "length":
      bad_reasons.append("length_stop")

    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    for marker in LEAK_MARKERS:
      if marker in reasoning or marker in content:
        bad_reasons.append(f"markup_leak:{marker}")

    for tc in msg.get("tool_calls") or []:
      raw_args = tc.get("function", {}).get("arguments") or ""
      try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict) or "code" not in args:
          bad_reasons.append("missing_code_param")
        elif "<parameter=" in str(args["code"]) or "</function>" in str(
            args["code"]
        ):
          bad_reasons.append("xml_in_code")
      except Exception:  # pylint: disable=broad-exception-caught
        bad_reasons.append("bad_tool_json")
    if not (msg.get("tool_calls") or content.strip()):
      bad_reasons.append("empty_assistant_turn")
  return bad_reasons


def run_3level_session_check(
    session: ds_eval.HavuSubprocessSession,
    pred: list[list[int]] | None,
    code: str | None,
    extra_pairs: list[dict[str, Any]] | None = None,
    per_call_timeout: float = 2.5,
) -> dict[str, Any]:
  """Evaluates `pred` and `code` inside the live `HavuSubprocessSession` on Level 1 (`train`) and Level 3 (`extra_pairs`).

  Extends `HavuSubprocessSession.run_train_check` (`havu/evals/train_check.py`) so that the
  winning verification path (`expr`, `cell`, or `fn` that reproduces `pred` on `test_input`)
  is tested on both `session.train` (Level 1) and `extra_pairs` (Level 3 held-out generator pairs)
  within the live IPython session where all helper functions and imports reside.
  """
  t0 = time.monotonic()
  extra_list = extra_pairs or []
  n_train = len(session.train)
  n_extra = len(extra_list)

  if pred is None or not ds_eval.is_valid_havu_grid(pred):
    return {
        "train_pass": None,
        "train_frac": None,
        "level3_pass": None if n_extra > 0 else True,
        "level3_frac": None if n_extra > 0 else 1.0,
        "train_trace": "no_answer",
        "train_paths": [],
        "verified_code": code,
        "train_check_seconds": 0.0,
    }

  train_outs = [ds_eval._grid_hash(p["output"]) for p in session.train]  # pylint: disable=protected-access
  extra_outs = [ds_eval._grid_hash(p["output"]) for p in extra_list]  # pylint: disable=protected-access
  ph = ds_eval._grid_hash(pred)  # pylint: disable=protected-access

  # Layout of `inputs`: [train_0 .. train_{n_train-1}, extra_0 .. extra_{n_extra-1}, test_input]
  inputs = (
      [p["input"] for p in session.train]
      + [p["input"] for p in extra_list]
      + [session.test_input]
  )
  cells = [c for name, c in session.executed if name == "run_python"]

  if code is None:
    for name, c in reversed(session.executed):
      if name == "submit_answer":
        code = c
        break

  body = r"""
def _alarm(s, f): raise TimeoutError()
def _valid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30
            and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))
def _hash(v):
    if hasattr(v, 'tolist'): v = v.tolist()
    v = json.loads(json.dumps(v, default=int))
    return hashlib.sha1(json.dumps(v, separators=(',', ':')).encode()).hexdigest()[:16] if _valid(v) else 'INVALID'
def _eval_with(fn, sec):
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return _hash(fn())
    except BaseException as e:
        return 'ERR:' + type(e).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

inps = _payload['inputs']
code = _payload['code']
cells = _payload['cells']
ph = _payload['ph']
sec = _payload['sec']
_out = {}
old_h = signal.signal(signal.SIGALRM, _alarm)
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if code is not None:
            try:
                c, mode = compile(code, '<tc>', 'eval'), 'eval'
            except SyntaxError:
                try:
                    c, mode = compile(code, '<tc>', 'exec'), 'exec'
                except SyntaxError:
                    c, mode = None, None
            if c is not None and 'test_input' in c.co_names:
                preds = []
                # Evaluate test_input (inps[-1]) first so non-matching expressions short-circuit fast
                for g in [inps[-1]] + inps[:-1]:
                    def _run(g=g):
                        env = {**_g, 'test_input': copy.deepcopy(g)}
                        if mode == 'eval':
                            return eval(c, env)
                        exec(c, env)
                        return env['answer']
                    preds.append(_eval_with(_run, sec))
                    if len(preds) == 1 and preds[0] != ph:
                        break
                if preds and preds[0] == ph:
                    _out['expr'] = preds[1:] + preds[:1]

        expr_ok = _out.get('expr', [None])[-1] == ph
        if not expr_ok:
            vars_matching = []
            for k, v in list(_g.items()):
                if k.startswith('_') or isinstance(v, (types.FunctionType, types.ModuleType, type)):
                    continue
                if isinstance(v, (list, _np.ndarray)):
                    try:
                        if _hash(v) == ph:
                            vars_matching.append(k)
                    except Exception:
                        pass
            for var in sorted(vars_matching, key=lambda x: x != 'answer')[:3]:
                for src in reversed(cells):
                    try:
                        tree = ast.parse(src)
                        stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                    except SyntaxError:
                        stores, loads = set(), set()
                    if var in stores:
                        if 'test_input' in loads:
                            c = compile(src, '<tc_cell>', 'exec')
                            preds = []
                            for g in [inps[-1]] + inps[:-1]:
                                def _run_c(g=g, var=var, c=c):
                                    env = {**_g, 'test_input': copy.deepcopy(g)}
                                    env.pop(var, None)
                                    exec(c, env)
                                    return env[var]
                                preds.append(_eval_with(_run_c, sec))
                                if len(preds) == 1 and preds[0] != ph:
                                    break
                            if preds and preds[0] == ph:
                                _out['cell'] = {'var': var, 'src': src, 'preds': preds[1:] + preds[:1]}
                        break
                if 'cell' in _out:
                    break

            fns = [k for k, v in _g.items() if isinstance(v, types.FunctionType) and not k.startswith('_')]
            fn_hits = {}
            for name in fns[-25:]:
                fn = _g[name]
                preds = []
                for g in [inps[-1]] + inps[:-1]:
                    h = _eval_with(lambda g=g, fn=fn: fn(copy.deepcopy(g)), sec)
                    if h.startswith('ERR:'):
                        h2 = _eval_with(lambda g=g, fn=fn: fn(_np.array(g)), sec)
                        if not h2.startswith('ERR:'):
                            h = h2
                    preds.append(h)
                    if len(preds) == 1 and h != ph:
                        break
                if preds and preds[0] == ph:
                    fn_hits[name] = preds[1:] + preds[:1]
            if fn_hits:
                _out['fns'] = fn_hits
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old_h)
"""
  total_pairs = n_train + n_extra + 1
  data = session._run_marked(  # pylint: disable=protected-access
      body,
      {
          "inputs": inputs,
          "code": code,
          "cells": cells,
          "ph": ph,
          "sec": per_call_timeout,
      },
      timeout=max(25.0, per_call_timeout * total_pairs * 2.0),
  )
  elapsed = round(time.monotonic() - t0, 2)
  if not isinstance(data, dict):
    return {
        "train_pass": None,
        "train_frac": None,
        "level3_pass": None if n_extra > 0 else True,
        "level3_frac": None if n_extra > 0 else 1.0,
        "train_trace": "check_error",
        "train_paths": [],
        "verified_code": code,
        "train_check_seconds": elapsed,
    }

  def _score_train_and_extra(preds_list: list[str]) -> tuple[int, int]:
    t_k = sum(a == b for a, b in zip(preds_list[:n_train], train_outs))
    e_k = sum(
        a == b
        for a, b in zip(preds_list[n_train : n_train + n_extra], extra_outs)
    )
    return t_k, e_k

  # Candidate tuple: (train_k, extra_k, priority, trace_name, verified_code_str)
  paths: list[tuple[int, int, int, str, str | None]] = []
  if "expr" in data and data["expr"][-1] == ph:
    tk, ek = _score_train_and_extra(data["expr"])
    paths.append((tk, ek, 3, "expr", code))
  if "cell" in data and data["cell"]["preds"][-1] == ph:
    tk, ek = _score_train_and_extra(data["cell"]["preds"])
    paths.append((tk, ek, 2, "cell", data["cell"].get("src") or code))
  if "fns" in data:
    best_fn_tuple: tuple[int, int, str] | None = None
    for fn_name, fn_preds in data["fns"].items():
      tk, ek = _score_train_and_extra(fn_preds)
      cand = (tk, ek, fn_name)
      if best_fn_tuple is None or cand > best_fn_tuple:
        best_fn_tuple = cand
    if best_fn_tuple is not None:
      paths.append((
          best_fn_tuple[0],
          best_fn_tuple[1],
          1,
          "fn",
          f"{best_fn_tuple[2]}(test_input)",
      ))

  if paths:
    tk, ek, _, trace, v_code = max(paths)
    return {
        "train_pass": tk == n_train,
        "train_frac": round(tk / max(1, n_train), 4),
        "level3_pass": (ek == n_extra) if n_extra > 0 else True,
        "level3_frac": round(ek / max(1, n_extra), 4) if n_extra > 0 else 1.0,
        "train_trace": trace,
        "train_paths": [p[3] for p in paths],
        "verified_code": v_code,
        "train_check_seconds": elapsed,
    }
  return {
      "train_pass": None,
      "train_frac": None,
      "level3_pass": None if n_extra > 0 else True,
      "level3_frac": None if n_extra > 0 else 1.0,
      "train_trace": "untraceable",
      "train_paths": [],
      "verified_code": code,
      "train_check_seconds": elapsed,
  }


class Top10kRolloutRuntime(ds_eval.HavuAttemptRuntime):
  """Extends `HavuAttemptRuntime` with uncompacted SFT messages, per-turn events, and 3-level metadata."""

  def __init__(
      self,
      state: RolloutAttemptState,
      messages: list[dict[str, Any]],
      reasoning_effort: int,
      session: ds_eval.HavuSubprocessSession,
      base_seed: int,
      candidate_meta: dict[str, Any],
      extra_verification_pairs: list[dict[str, Any]],
  ):
    super().__init__(
        state=state,  # type: ignore[arg-type]
        messages=messages,
        reasoning_effort=reasoning_effort,
        session=session,
        base_seed=base_seed,
    )
    # Full uncompacted OpenAI messages list for SFT (`encode_havu_messages_with_fit` never mutates this)
    self.uncompacted_messages: list[dict[str, Any]] = copy.deepcopy(messages)
    # 1-to-1 per-assistant-turn event dicts for `sft/data_pipeline.py:audit_rollout_messages`
    self.events: list[dict[str, Any]] = []
    self.candidate_meta = candidate_meta
    self.extra_verification_pairs = extra_verification_pairs
    self.started_at = time.perf_counter()
    self.elapsed_seconds = 0.0
    self.error_text: str | None = None


def run_top10k_rollout_wave_batched(
    llm: Any,
    vllm_mod: Any,
    runtimes: list[Top10kRolloutRuntime],
    wave_desc: str,
) -> tuple[int, float]:
  """Runs a wave of `Top10kRolloutRuntime` trajectories with `v8` settings and records uncompacted SFT messages."""
  if not runtimes:
    return 0, 0.0
  if hasattr(llm, "clear_prefix_cache"):
    llm.clear_prefix_cache()

  total_wave_gen_toks = 0
  total_wave_gen_s = 0.0
  tokenizer = llm.get_tokenizer()
  bos_id = int(getattr(getattr(llm, "cfg", None), "bos_token_id", 0))
  eos_id = int(getattr(getattr(llm, "cfg", None), "eos_token_id", 1))

  def _count_sub_tokens(txt: str) -> int:
    if not txt:
      return 0
    try:
      ids = tokenizer.encode(txt)
      if ids and ids[0] == bos_id:
        ids = ids[1:]
      return len(ids)
    except Exception:  # pylint: disable=broad-exception-caught
      return max(1, len(txt) // 3)

  for rt in runtimes:
    rt.started_at = time.perf_counter()

  turn_pbar = tqdm.tqdm(
      total=FLAGS.max_turns,
      desc=f"  {wave_desc} turns",
      leave=False,
      dynamic_ncols=True,
  )

  for turn_idx in range(FLAGS.max_turns):
    active_rts = [rt for rt in runtimes if not rt.state.done]
    if not active_rts:
      break

    prompts: list[str | list[int]] = []
    params_list: list[Any] = []
    sched_rts: list[Top10kRolloutRuntime] = []

    for rt in active_rts:
      st = rt.state
      rem_output = st.hard_budget - st.output_tokens
      rem_reasoning = (
          FLAGS.max_reasoning_tokens - st.thinking_tokens
          if FLAGS.max_reasoning_tokens > 0
          else rem_output
      )
      remaining_budget = min(rem_output, rem_reasoning)
      if remaining_budget <= 0:
        rt.stop_reason = (
            "reasoning_budget" if rem_reasoning <= 0 else "output_budget"
        )
        st.done = True
        rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)
        continue

      prompt_str, did_compact = ds_eval.encode_havu_messages_with_fit(
          messages=rt.messages,
          reasoning_effort=rt.reasoning_effort,
          tokenizer=tokenizer,
          max_seq_len=FLAGS.max_model_len,
          compact_at_tokens=FLAGS.compact_at_tokens,
          min_turn_headroom=min(8192, max(2048, remaining_budget)),
          drop_prior_thinking=FLAGS.drop_prior_turn_thinking,
      )
      st.transcript = prompt_str
      full_ids = tokenizer.encode(prompt_str)
      if not full_ids or full_ids[0] != bos_id:
        full_ids = [bos_id] + full_ids

      prompt_item: str | list[int] = full_ids
      if (
          FLAGS.token_exact
          and not did_compact
          and rt.exact_token_ids is not None
          and not rt.exact_resync
          and 0 < rt.exact_upto_msg_count <= len(rt.messages)
      ):
        upto_str = ds4_encoding.encode_messages(
            rt.messages[: rt.exact_upto_msg_count],
            thinking_mode="thinking",
            drop_thinking=False,
            reasoning_effort=rt.reasoning_effort,
        )
        if prompt_str.startswith(upto_str):
          upto_ids = tokenizer.encode(upto_str)
          if not upto_ids or upto_ids[0] != bos_id:
            upto_ids = [bos_id] + upto_ids
          if full_ids[: len(upto_ids)] == upto_ids:
            if upto_ids != rt.exact_token_ids:
              rt.token_exact_breaks_saved += 1
            prompt_item = list(rt.exact_token_ids) + full_ids[len(upto_ids) :]
          else:
            delta_ids = tokenizer.encode(prompt_str[len(upto_str) :])
            if delta_ids and delta_ids[0] == bos_id:
              delta_ids = delta_ids[1:]
            prompt_item = list(rt.exact_token_ids) + delta_ids

      p_toks = (
          len(prompt_item)
          if isinstance(prompt_item, list)
          else len(full_ids)
      )
      st.max_prompt_tokens = max(st.max_prompt_tokens, p_toks)
      ctx_headroom = FLAGS.max_model_len - p_toks - 128
      if ctx_headroom <= 256:
        st.ctx_cutoff = True
        rt.stop_reason = "ctx_cutoff"
        st.done = True
        rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)
        continue

      turn_cap = min(FLAGS.max_turn_tokens, remaining_budget, ctx_headroom)
      turn_seed = int((rt.base_seed + (turn_idx + 1) * 104729) & 0x7FFFFFFF)
      sched_rts.append(rt)
      prompts.append(prompt_item)
      params_list.append(
          vllm_mod.SamplingParams(
              temperature=FLAGS.temperature,
              top_p=FLAGS.top_p,
              top_k=FLAGS.top_k,
              repetition_penalty=FLAGS.repetition_penalty,
              max_tokens=turn_cap,
              stop=ds_eval.DS41_STOP_STRINGS,
              seed=turn_seed,
          )
      )

    if not sched_rts:
      break

    t_gen0 = time.perf_counter()
    outputs = llm.generate(prompts, params_list, use_tqdm=False)
    gen_s = time.perf_counter() - t_gen0
    total_wave_gen_s += gen_s

    to_exec: list[tuple[Top10kRolloutRuntime, list[dict[str, Any]], str]] = []

    for rt, out, sp_used in zip(sched_rts, outputs, params_list):
      st = rt.state
      gen_out = out.outputs[0]
      raw_text = gen_out.text
      n_tok = len(gen_out.token_ids)
      finish_reason = str(gen_out.finish_reason)
      st.prompt_tokens = len(out.prompt_token_ids)
      st.output_tokens += n_tok
      st.turns += 1
      total_wave_gen_toks += n_tok

      reasoning_content, visible_content, openai_calls, is_canonical = (
          ds_eval.parse_ds41_assistant_turn(
              raw_text, st.turns, finish_reason=finish_reason
          )
      )

      think_toks = min(n_tok, _count_sub_tokens(reasoning_content))
      if FLAGS.max_reasoning_tokens > 0:
        think_toks = min(
            think_toks, max(0, FLAGS.max_reasoning_tokens - st.thinking_tokens)
        )
      if openai_calls:
        tool_toks = max(0, n_tok - think_toks)
        ans_toks = 0
      else:
        tool_toks = 0
        ans_toks = max(0, n_tok - think_toks)

      st.thinking_tokens += think_toks
      st.tool_use_tokens += tool_toks
      st.answer_tokens += ans_toks
      if reasoning_content:
        st.thought_text = (
            f"{st.thought_text}\n\n[Turn {st.turns}]\n{reasoning_content.strip()}"
            if st.thought_text
            else f"[Turn {st.turns}]\n{reasoning_content.strip()}"
        )
      if visible_content:
        st.visible_text = (
            f"{st.visible_text}\n\n{visible_content}"
            if st.visible_text
            else visible_content
        )

      # E043 `--rescue-code`
      if (
          not openai_calls
          and FLAGS.rescue_code
          and finish_reason == "stop"
          and ds_eval.havu_arcprize_parse(visible_content) is None
      ):
        rescued_call = ds_eval.rescue_havu_tool_call(visible_content, st.turns)
        if rescued_call is not None:
          openai_calls = [rescued_call]
          visible_content = ""
          is_canonical = False
          rt.code_rescues += 1

      ev_dict = {
          "turn": st.turns,
          "output_tokens": n_tok,
          "thinking_tokens": think_toks,
          "tool_use_tokens": tool_toks,
          "answer_tokens": ans_toks,
          "finish_reason": finish_reason,
          "num_tool_calls": len(openai_calls),
          "is_canonical": is_canonical,
      }
      st.turn_logs.append(ev_dict)
      rt.events.append(ev_dict)

      clean_rc = sanitize_sft_text(reasoning_content)
      clean_vc = sanitize_sft_text(visible_content)

      if not openai_calls:
        # 1. Check if visible content contains a valid grid (`parse` or `lenient_parse`)
        cand_grid = ds_eval.havu_arcprize_parse(visible_content)
        if cand_grid is not None:
          asst_msg = {
              "role": "assistant",
              "content": clean_vc,
              "reasoning_content": clean_rc,
          }
          rt.messages.append(copy.deepcopy(asst_msg))
          rt.uncompacted_messages.append(asst_msg)
          rt.accepted_grid = cand_grid
          rt.answer_source = "parse"
          rt.stop_reason = "final"
          st.done = True
          rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)
          continue

        # 2. E043 `--rescue-code` nudge when the model stopped without a tool call, code, or grid (at most 3x)
        if (
            FLAGS.rescue_code
            and finish_reason == "stop"
            and rt.nudge_rescues < 3
            and turn_idx < FLAGS.max_turns - 1
        ):
          rt.code_rescues += 1
          rt.nudge_rescues += 1
          rt.exact_resync = True
          asst_msg = {
              "role": "assistant",
              "content": clean_vc,
              "reasoning_content": clean_rc,
          }
          user_nudge = {"role": "user", "content": ds_eval.HAVU_RESCUE_NUDGE}
          rt.messages.append(copy.deepcopy(asst_msg))
          rt.messages.append(copy.deepcopy(user_nudge))
          rt.uncompacted_messages.append(asst_msg)
          rt.uncompacted_messages.append(user_nudge)
          continue

        # 3. Per-turn thinking cap (`--cap-nudge`)
        rem_out_after = st.hard_budget - st.output_tokens
        rem_think_after = (
            FLAGS.max_reasoning_tokens - st.thinking_tokens
            if FLAGS.max_reasoning_tokens > 0
            else rem_out_after
        )
        rem_after = min(rem_out_after, rem_think_after)
        if (
            finish_reason == "length"
            and rem_after > 1024
            and turn_idx < FLAGS.max_turns - 1
        ):
          rt.cap_nudges += 1
          rt.exact_resync = True
          cid = f"cap_{turn_idx}"
          asst_msg = {
              "role": "assistant",
              "content": "",
              "reasoning_content": clean_rc,
              "tool_calls": [{
                  "id": cid,
                  "type": "function",
                  "function": {
                      "name": "run_python",
                      "arguments": json.dumps(
                          {"code": "pass  # per-turn thinking limit"}
                      ),
                  },
              }],
          }
          cap_k = max(1, round(sp_used.max_tokens / 1000))
          used_for_note = (
              st.thinking_tokens
              if FLAGS.max_reasoning_tokens > 0
              else st.output_tokens
          )
          limit_for_note = (
              FLAGS.max_reasoning_tokens
              if FLAGS.max_reasoning_tokens > 0
              else st.hard_budget
          )
          note = ds_eval.format_havu_budget_note(
              used_tokens=used_for_note,
              soft_target=st.soft_target,
              hard_limit=limit_for_note,
          ) + ds_eval.format_havu_turns_note(
              turn_index=turn_idx,
              max_turns=FLAGS.max_turns,
              threshold=FLAGS.turns_note_threshold,
          )
          tool_msg = {
              "role": "tool",
              "tool_call_id": cid,
              "content": ds_eval.HAVU_CAP_NOTE.format(k=cap_k) + note,
          }
          rt.messages.append(copy.deepcopy(asst_msg))
          rt.messages.append(copy.deepcopy(tool_msg))
          rt.uncompacted_messages.append(asst_msg)
          rt.uncompacted_messages.append(tool_msg)
          continue

        asst_msg = {
            "role": "assistant",
            "content": clean_vc,
            "reasoning_content": clean_rc,
        }
        rt.messages.append(copy.deepcopy(asst_msg))
        rt.uncompacted_messages.append(asst_msg)
        rt.stop_reason = "final" if finish_reason == "stop" else finish_reason
        st.done = True
        rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)
        continue

      asst_msg = {
          "role": "assistant",
          "content": clean_vc,
          "reasoning_content": clean_rc,
          "tool_calls": copy.deepcopy(openai_calls),
      }
      rt.messages.append(copy.deepcopy(asst_msg))
      rt.uncompacted_messages.append(asst_msg)
      if FLAGS.token_exact and is_canonical and finish_reason == "stop":
        rt.exact_token_ids = (
            list(out.prompt_token_ids) + list(gen_out.token_ids) + [eos_id]
        )
        rt.exact_upto_msg_count = len(rt.messages)
        rt.exact_resync = False
      else:
        rt.exact_resync = True
      to_exec.append((rt, openai_calls, finish_reason))

    # Execute tool calls in parallel across isolated child-process sandboxes
    if to_exec:
      cur_turn_idx = turn_idx

      def _exec_rt_tools(
          item: tuple[Top10kRolloutRuntime, list[dict[str, Any]], str],
          t_idx: int = cur_turn_idx,
      ) -> None:
        rt, calls, f_reason = item
        st = rt.state
        accepted_grid: list[list[int]] | None = None
        accepted_code: str | None = None

        for call in calls:
          st.tool_calls += 1
          call_id = call["id"]
          try:
            if f_reason == "length":
              raise ValueError("Tool skipped: truncated response")
            args_dict = json.loads(call["function"]["arguments"])
            if not isinstance(args_dict, dict) or not isinstance(
                args_dict.get("code"), str
            ):
              raise ValueError("Expected an object with a string code field")
            fn_name = call["function"]["name"]
            code_str = args_dict["code"]

            if accepted_grid is not None:
              tool_out = "[skipped: solution already accepted]"
            elif fn_name == "run_python":
              status, tool_out = rt.session.run_python(code_str)
              if status != "completed":
                rt.tool_errors += 1
            elif fn_name == "submit_answer":
              tool_out, grid = rt.session.submit_answer(code_str)
              if grid is not None:
                accepted_grid = grid
                accepted_code = code_str
              else:
                rt.tool_errors += 1
            else:
              raise ValueError(f"Unknown tool: {fn_name}")
          except Exception as exc:  # pylint: disable=broad-exception-caught
            tool_out = f"[tool error: {exc}]"
            rt.tool_errors += 1

          tool_msg = {
              "role": "tool",
              "tool_call_id": call_id,
              "content": tool_out,
          }
          rt.messages.append(copy.deepcopy(tool_msg))
          rt.uncompacted_messages.append(tool_msg)

        if accepted_grid is not None:
          rt.accepted_grid = accepted_grid
          rt.submit_code = accepted_code
          rt.answer_source = "submit"
          rt.stop_reason = "submitted"
          st.done = True
          rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)
          return

        used_for_note = (
            st.thinking_tokens
            if FLAGS.max_reasoning_tokens > 0
            else st.output_tokens
        )
        limit_for_note = (
            FLAGS.max_reasoning_tokens
            if FLAGS.max_reasoning_tokens > 0
            else st.hard_budget
        )
        note = ds_eval.format_havu_budget_note(
            used_tokens=used_for_note,
            soft_target=st.soft_target,
            hard_limit=limit_for_note,
        ) + ds_eval.format_havu_turns_note(
            turn_index=t_idx,
            max_turns=FLAGS.max_turns,
            threshold=FLAGS.turns_note_threshold,
        )
        if note and rt.messages and rt.messages[-1].get("role") == "tool":
          rt.messages[-1]["content"] += note
        if (
            note
            and rt.uncompacted_messages
            and rt.uncompacted_messages[-1].get("role") == "tool"
        ):
          rt.uncompacted_messages[-1]["content"] += note

      with futures.ThreadPoolExecutor(
          max_workers=min(32, len(to_exec))
      ) as pool:
        list(pool.map(_exec_rt_tools, to_exec))

    rem_active = sum(1 for rt in runtimes if not rt.state.done)
    submitted_so_far = sum(
        1 for rt in runtimes if rt.accepted_grid is not None
    )
    tok_per_s = total_wave_gen_toks / max(total_wave_gen_s, 1e-6)
    turn_pbar.set_postfix(
        active=f"{rem_active}/{len(runtimes)}",
        submitted=submitted_so_far,
        tok_s=f"{tok_per_s:.0f}",
    )
    turn_pbar.update(1)
    if rem_active == 0:
      break

  turn_pbar.close()

  # Post-loop: 3-tier session fallback (`answer` -> `solve(test_input)` -> `salvage_answer`)
  # + in-session 3-level check (`train` + `extra_verification_pairs`)
  def _finalize_one(rt: Top10kRolloutRuntime) -> None:
    if rt.stop_reason is None:
      rt.stop_reason = "max_turns"
    if rt.elapsed_seconds <= 0.0:
      rt.elapsed_seconds = round(time.perf_counter() - rt.started_at, 3)

    try:
      if rt.accepted_grid is None:
        # 1. Try session `answer` variable
        _, grid = rt.session.submit_answer("answer", record_execution=False)
        if grid is not None:
          rt.accepted_grid = grid
          rt.answer_source = "session"
          rt.submit_code = "answer"
        else:
          # 2. Try session `solve(test_input)` (`--fallback-solve`)
          _, grid = rt.session.submit_answer(
              "solve(test_input)", record_execution=False
          )
          if grid is not None:
            rt.accepted_grid = grid
            rt.answer_source = "session_solve"
            rt.submit_code = "solve(test_input)"
          else:
            # 3. Try Vu's `salvage_answer(session)`
            salv_grid, salv_name = rt.session.salvage_answer()
            if salv_grid is not None:
              rt.accepted_grid = salv_grid
              rt.answer_source = f"salvage:{salv_name}"
              fn_base = (salv_name or "solve").replace("(np)", "")
              rt.submit_code = (
                  f"{fn_base}(np.array(test_input))"
                  if (salv_name or "").endswith("(np)")
                  else f"{fn_base}(test_input)"
              )

      # 4. Run in-session 3-level verification (`train` + `extra_verification_pairs`)
      rt.train_labels = run_3level_session_check(
          session=rt.session,
          pred=rt.accepted_grid,
          code=rt.submit_code,
          extra_pairs=rt.extra_verification_pairs,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      rt.error_text = f"{type(exc).__name__}: {exc}"
    finally:
      rt.session.close()

  with futures.ThreadPoolExecutor(
      max_workers=min(32, len(runtimes))
  ) as pool:
    list(pool.map(_finalize_one, runtimes))

  return total_wave_gen_toks, total_wave_gen_s


def build_e031_rollout_row(
    rt: Top10kRolloutRuntime,
    target_output: list[list[int]] | None,
    active_model_name: str,
    active_backend: str,
    eval_task_ids: set[str],
    batch_label: str = "top10k_v8_pass1",
) -> tuple[dict[str, Any], bool, list[str]]:
  """Builds a single rollout record matching the 24-column E031 schema + 3-level verification metadata.

  Returns `(row_dict, is_verified_clean_sft, bad_audit_reasons)`.
  """
  st = rt.state
  meta = rt.candidate_meta
  task_id = str(
      meta.get("deepseek_task_id")
      or meta.get("original_task_id")
      or st.task_id
  )
  dataset_name = str(meta.get("dataset") or "unknown")
  source_str = str(meta.get("source_group") or dataset_name.lower())
  raw_parent = str(
      meta.get("seed_task_id")
      or meta.get("family_id")
      or meta.get("original_task_id")
      or task_id
  )
  parent_id = get_parent_puzzle_id(task_id, raw_parent)

  pred_grid = rt.accepted_grid
  test_exact = bool(
      pred_grid is not None
      and target_output is not None
      and pred_grid == target_output
  )
  t_pass = rt.train_labels.get("train_pass")
  t_frac = rt.train_labels.get("train_frac")
  l3_pass = rt.train_labels.get("level3_pass")
  l3_frac = rt.train_labels.get("level3_frac")
  n_l3 = len(rt.extra_verification_pairs)

  l1_ok = bool(t_pass is True)
  l2_ok = bool(test_exact)
  l3_ok = bool(l3_pass is True) if n_l3 > 0 else True
  all_3_ok = bool(l1_ok and l2_ok and l3_ok)

  # In E031 schema, `correct` is set to `bool(l2_ok and l3_ok)` so that
  # `correct is True and train_pass is True` in `sft/audit_and_prepare_data.py`
  # selects strictly 3-level verified trajectories (`all_3_levels_verified`)!
  e031_correct = bool(l2_ok and l3_ok)

  is_contam, contam_reason = is_task_contaminated(
      task_id, parent_id, eval_task_ids
  )
  bad_reasons = audit_rollout_messages(rt.uncompacted_messages, rt.events)
  is_verified_clean_sft = bool(
      all_3_ok
      and rt.answer_source == "submit"
      and not is_contam
      and not bad_reasons
  )

  row: dict[str, Any] = {
      # --- Exact 24 E031 Columns ---
      "task_id": task_id,
      "batch": batch_label,
      "source": source_str,
      "parent": parent_id,
      "pair": int(st.pair_idx),
      "attempt": int(st.attempt_idx),
      "correct": e031_correct,
      "train_pass": bool(t_pass) if t_pass is not None else None,
      "train_frac": float(t_frac) if t_frac is not None else None,
      "answer_source": rt.answer_source,
      "stop_reason": rt.stop_reason or "max_turns",
      "error": rt.error_text,
      "pred": (
          json.dumps(pred_grid, separators=(",", ":"))
          if pred_grid is not None
          else None
      ),
      "turns": int(st.turns),
      "tool_calls": int(st.tool_calls),
      "prompt_tokens": int(st.max_prompt_tokens or st.prompt_tokens),
      "cache_hit_tokens": 0,
      "completion_tokens": int(st.output_tokens),
      "seconds": float(rt.elapsed_seconds),
      "est_cost_usd": 0.0,
      "model": active_model_name,
      "provider": active_backend,
      "messages": json.dumps(rt.uncompacted_messages, ensure_ascii=False),
      "events": json.dumps(rt.events, ensure_ascii=False),
      # --- Explicit 3-Level Verification & Candidate Metadata Columns ---
      "test_exact": l2_ok,
      "level3_pass": bool(l3_pass) if l3_pass is not None else None,
      "level3_frac": float(l3_frac) if l3_frac is not None else None,
      "num_level3_pairs": int(n_l3),
      "all_3_levels_verified": all_3_ok,
      "verified_clean_sft": is_verified_clean_sft,
      "is_contaminated": is_contam,
      "contamination_reason": contam_reason,
      "audit_bad_reasons": json.dumps(bad_reasons),
      "train_trace": rt.train_labels.get("train_trace"),
      "submitted_solve_code": (
          rt.train_labels.get("verified_code") or rt.submit_code
      ),
      "thinking_tokens": int(st.thinking_tokens),
      "tool_use_tokens": int(st.tool_use_tokens),
      "answer_tokens": int(st.answer_tokens),
      "cap_nudges": int(rt.cap_nudges),
      "code_rescues": int(rt.code_rescues),
      "token_exact_breaks_saved": int(rt.token_exact_breaks_saved),
      "sample_id": meta.get("sample_id"),
      "dataset": dataset_name,
      "original_task_id": meta.get("original_task_id"),
      "learning_signal_score": meta.get("learning_signal_score"),
      "learning_tier": meta.get("learning_tier"),
      "rank": meta.get("rank"),
  }
  return row, is_verified_clean_sft, bad_reasons


def write_parquet_snapshot(rows: list[dict[str, Any]], out_path: str) -> bool:
  """Writes `rows` to a Parquet file on CNS via `pyarrow` + `gfile` if `pyarrow` is available."""
  if not _HAS_PYARROW or not rows or pa is None or pq is None:
    return False
  try:
    table = pa.Table.from_pylist(rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    tmp_path = f"{out_path}.tmp"
    with gfile.Open(tmp_path, "wb") as f:
      f.write(sink.getvalue().to_pybytes())
    gfile.Rename(tmp_path, out_path, overwrite=True)
    return True
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"WARNING: Parquet write failed for {out_path}: {exc}", flush=True)
    return False


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise ValueError(f"Unexpected command-line arguments: {argv[1:]}")

  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  if num_tpus not in (8, 16):
    raise RuntimeError(
        f"Expected 8 or 16 TensorCores for ds41_full_precision, got num_tpus={num_tpus} ({device_kind})"
    )

  eval_task_ids = load_eval_task_ids(FLAGS.eval_contamination_dir)

  print(
      f"Loading 10K candidate tasks from {FLAGS.candidates_jsonl}...",
      flush=True,
  )
  all_candidates: list[dict[str, Any]] = []
  with gfile.Open(FLAGS.candidates_jsonl, "r") as f:
    for line in f:
      line = line.strip()
      if line:
        all_candidates.append(json.loads(line))

  shard_idx = int(FLAGS.shard_idx)
  num_shards = int(FLAGS.num_shards)
  my_candidates = all_candidates[shard_idx::num_shards]
  if FLAGS.limit_tasks > 0:
    my_candidates = my_candidates[: FLAGS.limit_tasks]

  # Expand into (candidate_dict, pair_idx, test_pair) work items
  work_items: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
  for cand in my_candidates:
    test_pairs = cand.get("test") or []
    for p_idx, test_pair in enumerate(test_pairs):
      work_items.append((cand, p_idx, test_pair))

  gfile.MakeDirs(FLAGS.output_dir)
  shard_tag = f"shard_{shard_idx:03d}of{num_shards:03d}"
  rollouts_jsonl_path = os.path.join(
      FLAGS.output_dir, f"rollouts_{shard_tag}.jsonl"
  )
  verified_jsonl_path = os.path.join(
      FLAGS.output_dir, f"verified_rollouts_{shard_tag}.jsonl"
  )
  rollouts_parquet_path = os.path.join(
      FLAGS.output_dir, f"rollouts_{shard_tag}.parquet"
  )
  verified_parquet_path = os.path.join(
      FLAGS.output_dir, f"verified_rollouts_{shard_tag}.parquet"
  )
  summary_shard_path = os.path.join(
      FLAGS.output_dir, f"summary_{shard_tag}.json"
  )

  # Automatic resume if shard already has partial progress on CNS
  completed_rows: list[dict[str, Any]] = []
  verified_rows: list[dict[str, Any]] = []
  done_keys: set[tuple[str, int, int]] = set()
  if gfile.Exists(rollouts_jsonl_path):
    print(
        f"Found existing shard file {rollouts_jsonl_path}; loading for resume...",
        flush=True,
    )
    with gfile.Open(rollouts_jsonl_path, "r") as rf:
      for line in rf:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
          key = (
              str(row["task_id"]),
              int(row["pair"]),
              int(row["attempt"]),
          )
          if key not in done_keys:
            done_keys.add(key)
            completed_rows.append(row)
            if row.get("verified_clean_sft"):
              verified_rows.append(row)
        except Exception:  # pylint: disable=broad-exception-caught
          continue
    print(
        f"Resumed {len(completed_rows)} completed rollouts "
        f"({len(verified_rows)} verified clean SFT).",
        flush=True,
    )

  # Filter work_items to those still needing at least one attempt
  pending_items: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
  for cand, p_idx, test_pair in work_items:
    tid = str(cand.get("deepseek_task_id") or cand.get("original_task_id"))
    if not all(
        (tid, p_idx, att_i) in done_keys for att_i in range(FLAGS.num_attempts)
    ):
      pending_items.append((cand, p_idx, test_pair))

  vllm_mod: Any = ds_traces.CompatVllmNamespace
  active_backend = "ds41_full_precision"
  active_model_name = "DeepSeek-V4.1-Flash-1017 (Full-Precision BF16->FP32)"
  wave_bsz = max(2, int(FLAGS.pair_batch_size) * max(1, int(FLAGS.num_attempts)))
  llm = ds_eval.NativeDeepSeekV41FullPrecisionLLM(
      raw_checkpoint_dir=FLAGS.ds41_raw_checkpoint_dir,
      ocdbt_tp8_dir=FLAGS.ds41_ocdbt_tp8_dir,
      ocdbt_tp16_dir=FLAGS.ds41_ocdbt_tp16_dir,
      max_seq_len=FLAGS.max_model_len,
      batch_size=wave_bsz,
      steps_per_call=12,
      temperature=FLAGS.temperature,
      top_k=FLAGS.top_k,
  )

  print("=" * 88, flush=True)
  print(
      "DeepSeek-V4.1-Flash-1017 Top-10K SFT Rollout Generator (`v8` Best Pass@1 Setting)",
      flush=True,
  )
  print(
      f"  Model & Backend   : {active_model_name} ({active_backend})",
      flush=True,
  )
  print(
      f"  TPU Hardware      : {num_tpus}x {device_kind} (wave_bsz={wave_bsz})",
      flush=True,
  )
  print(
      f"  Shard             : {shard_idx + 1}/{num_shards} "
      f"({len(work_items)} total pairs, {len(pending_items)} pending)",
      flush=True,
  )
  print(
      f"  Attempts / Pair   : {FLAGS.num_attempts} (Pass@1 rollout generation)",
      flush=True,
  )
  print(
      f"  Sampling          : temperature={FLAGS.temperature}, top_k={FLAGS.top_k}, top_p={FLAGS.top_p}",
      flush=True,
  )
  print(
      f"  Budget & Context  : max_turns={FLAGS.max_turns}, max_turn_tokens={FLAGS.max_turn_tokens}, "
      f"total={FLAGS.total_token_budget}, soft={FLAGS.soft_token_target}, ctx={FLAGS.max_model_len}",
      flush=True,
  )
  print(f"  Output Directory  : {FLAGS.output_dir}", flush=True)
  print("=" * 88, flush=True)

  if not pending_items:
    print("All items in this shard are already completed!", flush=True)
    write_parquet_snapshot(completed_rows, rollouts_parquet_path)
    if verified_rows:
      write_parquet_snapshot(verified_rows, verified_parquet_path)
    return

  start_wall = time.time()
  cum_gen_tokens = 0
  cum_gen_seconds = 0.0

  batch_size = max(1, int(FLAGS.pair_batch_size))
  num_batches = (len(pending_items) + batch_size - 1) // batch_size

  shard_pbar = tqdm.tqdm(
      total=len(work_items) * FLAGS.num_attempts,
      initial=len(completed_rows),
      desc=f"Shard {shard_idx:03d}/{num_shards:03d} rollouts",
      dynamic_ncols=True,
  )

  for b_idx in range(num_batches):
    batch_slice = pending_items[b_idx * batch_size : (b_idx + 1) * batch_size]
    wave_runtimes: list[Top10kRolloutRuntime] = []
    rt_targets: list[list[list[int]] | None] = []

    for cand, pair_idx, test_pair in batch_slice:
      tid = str(cand.get("deepseek_task_id") or cand.get("original_task_id"))
      test_input = test_pair["input"]
      target_output = test_pair.get("output")
      extra_pairs = cand.get("extra_verification_pairs") or []
      pair_seed = int(
          hashlib.sha1(f"{tid}_{pair_idx}".encode()).hexdigest()[:8], 16
      )

      for att_idx in range(FLAGS.num_attempts):
        if (tid, pair_idx, att_idx) in done_keys:
          continue
        init_msgs, effort_val = ds_eval.build_havu_initial_messages(
            task=cand,
            test_input=test_input,
            attempt_idx=att_idx,
            compact_grids=FLAGS.compact_grid_prompts,
        )
        st = RolloutAttemptState(
            task_id=tid,
            pair_idx=pair_idx,
            attempt_idx=att_idx,
            soft_target=FLAGS.soft_token_target,
            hard_budget=FLAGS.total_token_budget,
        )
        session = ds_eval.HavuSubprocessSession(
            train=cand["train"],
            test_input=test_input,
            timeout_s=float(FLAGS.sandbox_timeout_s),
            max_output_chars=16000,
            style="ipython",
            mem_mb=4096,
            grace_s=5.0,
        )
        base_seed = int((pair_seed + (att_idx + 1) * 1000003) & 0x7FFFFFFF)
        rt = Top10kRolloutRuntime(
            state=st,
            messages=init_msgs,
            reasoning_effort=effort_val,
            session=session,
            base_seed=base_seed,
            candidate_meta=cand,
            extra_verification_pairs=extra_pairs,
        )
        wave_runtimes.append(rt)
        rt_targets.append(target_output)

    if not wave_runtimes:
      continue

    w_toks, w_sec = run_top10k_rollout_wave_batched(
        llm=llm,
        vllm_mod=vllm_mod,
        runtimes=wave_runtimes,
        wave_desc=f"Batch {b_idx + 1}/{num_batches}",
    )
    cum_gen_tokens += w_toks
    cum_gen_seconds += w_sec

    new_all_rows: list[dict[str, Any]] = []
    new_verified_rows: list[dict[str, Any]] = []
    for rt, target_output in zip(wave_runtimes, rt_targets):
      row, is_verified_clean, _ = build_e031_rollout_row(
          rt=rt,
          target_output=target_output,
          active_model_name=active_model_name,
          active_backend=active_backend,
          eval_task_ids=eval_task_ids,
          batch_label=FLAGS.batch_label,
      )
      completed_rows.append(row)
      new_all_rows.append(row)
      if is_verified_clean:
        verified_rows.append(row)
        new_verified_rows.append(row)
      done_keys.add((str(row["task_id"]), int(row["pair"]), int(row["attempt"])))

    # Append new rows to CNS JSONL shards immediately
    with gfile.Open(rollouts_jsonl_path, "a") as rf:
      for r in new_all_rows:
        rf.write(json.dumps(r, ensure_ascii=False) + "\n")
    if new_verified_rows:
      with gfile.Open(verified_jsonl_path, "a") as vf:
        for r in new_verified_rows:
          vf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Periodically flush Parquet snapshots
    if (
        FLAGS.parquet_flush_every_batches > 0
        and (b_idx + 1) % FLAGS.parquet_flush_every_batches == 0
    ) or (b_idx + 1) == num_batches:
      write_parquet_snapshot(completed_rows, rollouts_parquet_path)
      if verified_rows:
        write_parquet_snapshot(verified_rows, verified_parquet_path)

    # Update live shard summary JSON
    n_done = len(completed_rows)
    n_l1 = sum(1 for r in completed_rows if r.get("train_pass") is True)
    n_l2 = sum(1 for r in completed_rows if r.get("test_exact") is True)
    n_l12 = sum(
        1
        for r in completed_rows
        if r.get("train_pass") is True and r.get("test_exact") is True
    )
    n_l123 = sum(
        1 for r in completed_rows if r.get("all_3_levels_verified") is True
    )
    n_sft = len(verified_rows)
    elapsed_wall = time.time() - start_wall
    avg_tok_s = round(cum_gen_tokens / max(cum_gen_seconds, 1e-6), 1)
    rem_count = max(0, len(work_items) * FLAGS.num_attempts - n_done)
    done_this_run = max(1, (b_idx + 1) * len(wave_runtimes))
    eta_sec = round(elapsed_wall / done_this_run * rem_count, 1)

    by_dataset: dict[str, dict[str, int]] = {}
    for r in completed_rows:
      ds = str(r.get("dataset") or "unknown")
      d_stat = by_dataset.setdefault(
          ds,
          {
              "total": 0,
              "train_pass": 0,
              "test_exact": 0,
              "all_3_levels_verified": 0,
              "verified_clean_sft": 0,
          },
      )
      d_stat["total"] += 1
      if r.get("train_pass") is True:
        d_stat["train_pass"] += 1
      if r.get("test_exact") is True:
        d_stat["test_exact"] += 1
      if r.get("all_3_levels_verified") is True:
        d_stat["all_3_levels_verified"] += 1
      if r.get("verified_clean_sft") is True:
        d_stat["verified_clean_sft"] += 1

    summary_payload = {
        "exp_name": FLAGS.exp_name,
        "batch_label": FLAGS.batch_label,
        "shard_index": shard_idx,
        "num_shards": num_shards,
        "total_target_rollouts": len(work_items) * FLAGS.num_attempts,
        "completed_rollouts": n_done,
        "l1_train_pass_count": n_l1,
        "l1_train_pass_rate": round(n_l1 / max(1, n_done), 4),
        "l2_test_exact_count": n_l2,
        "l2_test_exact_rate": round(n_l2 / max(1, n_done), 4),
        "l1_and_l2_count": n_l12,
        "l1_and_l2_rate": round(n_l12 / max(1, n_done), 4),
        "all_3_levels_verified_count": n_l123,
        "all_3_levels_verified_rate": round(n_l123 / max(1, n_done), 4),
        "verified_clean_sft_count": n_sft,
        "verified_clean_sft_rate": round(n_sft / max(1, n_done), 4),
        "by_dataset": by_dataset,
        "decode_tokens_generated": cum_gen_tokens,
        "avg_decode_tok_per_s": avg_tok_s,
        "elapsed_wall_seconds": round(elapsed_wall, 1),
        "estimated_remaining_seconds": eta_sec,
        "rollouts_jsonl_path": rollouts_jsonl_path,
        "verified_jsonl_path": verified_jsonl_path,
        "rollouts_parquet_path": rollouts_parquet_path,
        "verified_parquet_path": verified_parquet_path,
    }
    with gfile.Open(summary_shard_path, "w") as sf:
      json.dump(summary_payload, sf, indent=2)
    if shard_idx == 0:
      with gfile.Open(os.path.join(FLAGS.output_dir, "summary.json"), "w") as mf:
        json.dump(summary_payload, mf, indent=2)

    shard_pbar.update(len(new_all_rows))
    shard_pbar.set_postfix(
        L1=f"{n_l1}/{n_done} ({100*n_l1/max(1,n_done):.1f}%)",
        L123=f"{n_l123}/{n_done} ({100*n_l123/max(1,n_done):.1f}%)",
        SFT=f"{n_sft}",
        tok_s=f"{avg_tok_s:.0f}",
    )

  shard_pbar.close()
  write_parquet_snapshot(completed_rows, rollouts_parquet_path)
  if verified_rows:
    write_parquet_snapshot(verified_rows, verified_parquet_path)
  print(
      f"Shard {shard_idx:03d}/{num_shards:03d} COMPLETE: "
      f"{len(completed_rows)} rollouts, {len(verified_rows)} verified clean SFT -> {FLAGS.output_dir}",
      flush=True,
  )


if __name__ == "__main__":
  app_runner.run(main)
