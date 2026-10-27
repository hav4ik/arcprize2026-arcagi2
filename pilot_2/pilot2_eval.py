"""High-efficiency continuous-batching vLLM TPU runner for pilot_2 ARC-AGI-2 experiments.

Written from scratch for pilot_2 with zero dependencies on pilot_1.
Guarantees:
  - Resume-safe: on startup, loads any already-completed pairs from `pairs_shard_*.jsonl`,
    restores their counts (`pass1_hits`, `pass2_hits`, `sel_oracle_hits`, `diag_oracle_hits`,
    `verified_pair_hits`) and `submission_dict`, and appends (`"a"`) new pairs without
    truncating existing CNS files.
  - Pair-ordered continuous scheduling: schedules all attempts of pair $k$ together so
    pairs complete continuously every ~60-90s and flush all 5 CNS files (`summary`,
    `pairs`, `submission`, `transcripts`, `code_traces`) after every single pair.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
import json
import os
import time
from typing import Any

from absl import flags

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import sandbox_verifier
from google3.pyglib import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_id",
    "h1_python_tools_base",
    f"Hypothesis ID from {sorted(prompts_and_selector.PILOT2_HYPOTHESES.keys())}.",
    allow_override=True,
)
flags.DEFINE_string(
    "output_dir",
    "",
    "CNS directory to write all traces and metrics.",
    allow_override=True,
)
flags.DEFINE_string(
    "dataset_dir",
    prompts_and_selector.FROZEN_BASELINE_CONFIG["dataset_dir"],
    "CNS directory containing ARC-AGI-2 evaluation task JSON files.",
    allow_override=True,
)
flags.DEFINE_string(
    "model_path",
    prompts_and_selector.FROZEN_BASELINE_CONFIG["model_path"],
    "CNS path to the Qwen3.8-27B-FP8 checkpoint.",
    allow_override=True,
)
flags.DEFINE_string(
    "lora_adapter_path",
    "",
    "Optional CNS path to a LoRA adapter directory. When set, shards collaboratively export merged block-128 FP8 to --model_path on Borg before starting vLLM.",
    allow_override=True,
)
flags.DEFINE_string(
    "full_sft_ckpt_dir",
    "",
    "Optional CNS path to a Full-SFT distributed checkpoint directory. When set, shards collaboratively export block-128 FP8 to --model_path on Borg before starting vLLM.",
    allow_override=True,
)
flags.DEFINE_integer(
    "shard_index",
    0,
    "0-based shard index for parallel TPU workers.",
    allow_override=True,
)
flags.DEFINE_integer(
    "num_shards",
    1,
    "Total number of shards for this experiment.",
    allow_override=True,
)
flags.DEFINE_integer(
    "max_tasks",
    0,
    "Optional cap on number of tasks for smoke testing (0 = full 120 tasks / 167 pairs).",
    allow_override=True,
)


@dataclasses.dataclass
class EvalPairSpec:
  """One evaluation test pair (`(task_id, pair_idx)` out of 167)."""

  global_pair_idx: int
  task_id: str
  test_idx: int
  train_pairs: list[dict[str, Any]]
  test_input: list[list[int]]
  ground_truth_output: list[list[int]] | None


@dataclasses.dataclass
class AttemptStreamState:
  """Live state of one `(pair, attempt_idx)` trajectory running on the continuous engine."""

  pair_spec: EvalPairSpec
  attempt_idx: int
  cfg: prompts_and_selector.Pilot2HypothesisConfig
  sandbox: sandbox_verifier.Pilot2AttemptSandbox
  system_prompt: str
  user_prompt: str
  transcript: str
  seed: int

  turn_idx: int = 0
  cum_gen_tokens: int = 0
  in_nudge_continuation: bool = False
  nudge_stage: int = 0
  waiting_for_wave1: bool = False
  partial_turn_text: str = ""
  partial_turn_tokens: int = 0
  finished: bool = False
  finish_reason: str = ""
  turn_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def load_eval_pairs(
    dataset_dir: str,
    shard_index: int,
    num_shards: int,
    max_tasks: int = 0,
) -> list[EvalPairSpec]:
  """Load and deterministically shard the ARC-AGI-2 evaluation set."""
  filenames = sorted(f for f in gfile.ListDir(dataset_dir) if f.endswith(".json"))
  if max_tasks > 0:
    filenames = filenames[:max_tasks]

  all_pairs: list[EvalPairSpec] = []
  global_idx = 0
  for fname in filenames:
    task_id = fname[:-5]
    fpath = os.path.join(dataset_dir, fname)
    with gfile.Open(fpath, "r") as f:
      task_data = json.load(f)
    train_pairs = task_data.get("train", [])
    test_list = task_data.get("test", [])
    for t_idx, t_pair in enumerate(test_list):
      all_pairs.append(
          EvalPairSpec(
              global_pair_idx=global_idx,
              task_id=task_id,
              test_idx=t_idx,
              train_pairs=train_pairs,
              test_input=t_pair["input"],
              ground_truth_output=t_pair.get("output"),
          )
      )
      global_idx += 1

  return [
      p for p in all_pairs if (p.global_pair_idx % max(1, num_shards)) == shard_index
  ]


def grids_equal(a: list[list[int]] | None, b: list[list[int]] | None) -> bool:
  """Return True if two 2D integer grids are identical in shape and values."""
  if a is None or b is None:
    return False
  if len(a) != len(b):
    return False
  for ra, rb in zip(a, b):
    if len(ra) != len(rb) or any(int(x) != int(y) for x, y in zip(ra, rb)):
      return False
  return True


def execute_tool_for_attempt(
    st: AttemptStreamState,
    assistant_text: str,
    turn_tokens: int,
) -> tuple[str, bool, str]:
  """Parse and execute the tool call from `assistant_text` in a worker thread."""
  st.cum_gen_tokens += turn_tokens
  tool_name, args, parse_err = prompts_and_selector.parse_tool_call_from_output(
      assistant_text,
      allow_think_call_rescue=st.cfg.think_call_rescue,
  )

  if parse_err is not None or tool_name is None or args is None:
    if st.cfg.use_native_xml_tools:
      fmt_hint = (
          "Reply with `<tool_call>\\n<function=run_python>\\n<parameter=code>\\n"
          "YOUR_PYTHON_CODE\\n</parameter>\\n</function>\\n</tool_call>`."
      )
    else:
      fmt_hint = (
          'Wrap a single JSON tool call inside `<tool_call>{"name": "run_python", '
          '"arguments": {"code": "..."}}</tool_call>`.'
      )
    resp = f"Tool call parse error: {parse_err}\n{fmt_hint}"
    st.turn_records.append({
        "turn_idx": st.turn_idx,
        "gen_tokens_this_turn": turn_tokens,
        "cum_gen_tokens": st.cum_gen_tokens,
        "assistant_output": assistant_text,
        "parsed_tool_name": None,
        "parse_error": parse_err,
        "tool_response": resp,
    })
    return resp, False, ""

  if tool_name in ("run_python", "python_exec"):
    code = str(args.get("code", ""))
    resp, achieved_tp = st.sandbox.execute_run_python(
        code=code,
        turn_idx=st.turn_idx,
        cum_gen_tokens=st.cum_gen_tokens,
    )
    st.turn_records.append({
        "turn_idx": st.turn_idx,
        "gen_tokens_this_turn": turn_tokens,
        "cum_gen_tokens": st.cum_gen_tokens,
        "assistant_output": assistant_text,
        "parsed_tool_name": "run_python",
        "tool_code": code,
        "tool_response": resp,
        "achieved_train_pass": achieved_tp,
    })
    return resp, False, ""

  if tool_name == "submit_program":
    code = str(args.get("code", ""))
    resp, done = st.sandbox.execute_submit_program(
        code=code,
        turn_idx=st.turn_idx,
        cum_gen_tokens=st.cum_gen_tokens,
    )
    st.turn_records.append({
        "turn_idx": st.turn_idx,
        "gen_tokens_this_turn": turn_tokens,
        "cum_gen_tokens": st.cum_gen_tokens,
        "assistant_output": assistant_text,
        "parsed_tool_name": "submit_program",
        "tool_code": code,
        "tool_response": resp,
        "verified_done": done,
    })
    return resp, done, ("verified_submit_program" if done else "")

  if tool_name == "submit_answer":
    if st.sandbox.use_e031_ipython_mode or (
        "code" in args and "grid" not in args
    ):
      code_expr = str(
          args.get("code")
          if args.get("code") is not None
          else args.get("grid", "")
      )
      resp, done = st.sandbox.execute_submit_answer_code(
          code=code_expr,
          turn_idx=st.turn_idx,
          cum_gen_tokens=st.cum_gen_tokens,
      )
      st.turn_records.append({
          "turn_idx": st.turn_idx,
          "gen_tokens_this_turn": turn_tokens,
          "cum_gen_tokens": st.cum_gen_tokens,
          "assistant_output": assistant_text,
          "parsed_tool_name": "submit_answer",
          "tool_code": code_expr,
          "tool_response": resp,
          "answer_done": done,
      })
      return resp, done, ("e031_submit_answer" if done else "")

    grid_obj = args.get("grid")
    resp, done = st.sandbox.execute_submit_answer(
        grid_obj=grid_obj,
        turn_idx=st.turn_idx,
        cum_gen_tokens=st.cum_gen_tokens,
    )
    st.turn_records.append({
        "turn_idx": st.turn_idx,
        "gen_tokens_this_turn": turn_tokens,
        "cum_gen_tokens": st.cum_gen_tokens,
        "assistant_output": assistant_text,
        "parsed_tool_name": "submit_answer",
        "tool_response": resp,
        "answer_done": done,
    })
    return resp, done, ("direct_submit_answer" if done else "")

  resp = (
      f"Unknown tool `{tool_name}`. Available tools: "
      "`run_python`, `submit_program`, `submit_answer`."
  )
  st.turn_records.append({
      "turn_idx": st.turn_idx,
      "gen_tokens_this_turn": turn_tokens,
      "cum_gen_tokens": st.cum_gen_tokens,
      "assistant_output": assistant_text,
      "parsed_tool_name": tool_name,
      "tool_response": resp,
  })
  return resp, False, ""


def build_attempt_candidate_summary(
    st: AttemptStreamState,
) -> prompts_and_selector.AttemptCandidateSummary:
  """Convert a finished `AttemptStreamState` into an `AttemptCandidateSummary` for voting."""
  sb = st.sandbox
  if sb.use_e031_ipython_mode:
    sb.finalize_e031_session_fallback(st.turn_idx, st.cum_gen_tokens)
  has_tp = sb.first_tp_pred is not None
  secondary: list[tuple[list[list[int]], int, int, float]] = []
  if has_tp and len(sb.verified_variants) > 1:
    first_key = sandbox_verifier.grid_to_canonical_key(sb.first_tp_pred)
    for var in sb.verified_variants[1:]:
      if sandbox_verifier.grid_to_canonical_key(var.grid) != first_key:
        secondary.append(
            (var.grid, var.turn_idx, var.cum_gen_tokens, var.equivariance_bonus)
        )

  return prompts_and_selector.AttemptCandidateSummary(
      attempt_idx=st.attempt_idx,
      prompt_style=st.cfg.attempt_prompt_styles[st.attempt_idx],
      has_task_summary=st.cfg.attempt_include_task_summary[st.attempt_idx],
      total_turns=st.turn_idx,
      total_gen_tokens=st.cum_gen_tokens,
      train_pass=has_tp,
      verified_grid=sb.first_tp_pred,
      verified_turn=sb.first_tp_turn,
      verified_gen_tokens=sb.first_tp_gen_tokens,
      equivariance_bonus=sb.first_tp_equivariance_bonus,
      secondary_verified_grids=secondary,
      partial_grid=sb.best_partial_pred,
      partial_train_frac=sb.best_partial_train_frac,
      partial_cell_acc=sb.best_partial_cell_acc,
      partial_gen_tokens=sb.best_partial_gen_tokens,
      fallback_grid=sb.submitted_answer_grid,
      fallback_gen_tokens=sb.submitted_answer_gen_tokens or st.cum_gen_tokens,
  )


def _safe_cns_append(path: str, text_payload: str) -> None:
  """Append text to a CNS file with 3-retry resilience."""
  for attempt in range(3):
    try:
      with gfile.Open(path, "a") as f:
        f.write(text_payload)
      return
    except Exception as exc:
      print(f"[WARN] CNS append to {path} failed (try {attempt+1}/3): {exc}", flush=True)
      time.sleep(2.0)


def _safe_cns_write_json(path: str, obj: Any, indent: int | None = None) -> None:
  """Write JSON to a CNS file with 3-retry resilience."""
  payload = json.dumps(obj, indent=indent)
  for attempt in range(3):
    try:
      with gfile.Open(path, "w") as f:
        f.write(payload)
      return
    except Exception as exc:
      print(f"[WARN] CNS write to {path} failed (try {attempt+1}/3): {exc}", flush=True)
      time.sleep(2.0)


def load_existing_completed_pairs(
    pairs_path: str,
) -> tuple[set[int], int, int, int, int, int, dict[str, list[dict[str, Any]]]]:
  """Load already-finished pairs from `pairs_path` if resuming after a preemption."""
  done_global_indices: set[int] = set()
  p1 = 0
  p2 = 0
  so = 0
  do = 0
  ver = 0
  sub_dict: dict[str, list[dict[str, Any]]] = {}
  if not gfile.Exists(pairs_path):
    return done_global_indices, p1, p2, so, do, ver, sub_dict

  try:
    with gfile.Open(pairs_path, "r") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except Exception:
          continue
        g_idx = int(row["global_pair_idx"])
        if g_idx in done_global_indices:
          continue
        done_global_indices.add(g_idx)
        if row.get("pass1"):
          p1 += 1
        if row.get("pass2"):
          p2 += 1
        if row.get("sel_oracle"):
          so += 1
        if row.get("diag_oracle"):
          do += 1
        if row.get("any_verified"):
          ver += 1
        tid = str(row["task_id"])
        t_idx = int(row["test_idx"])
        if "attempt_1" in row and "attempt_2" in row:
          sub_dict.setdefault(tid, []).append({
              "test_idx": t_idx,
              "attempt_1": row["attempt_1"],
              "attempt_2": row["attempt_2"],
          })
  except Exception as exc:
    print(f"[WARN] Could not read existing {pairs_path}: {exc}", flush=True)

  return done_global_indices, p1, p2, so, do, ver, sub_dict


def tasks_dict_to_eval_pairs(
    tasks_dict: dict[str, dict[str, Any]],
) -> list[EvalPairSpec]:
  """Convert a dictionary of `{task_id: task_dict}` (e.g. from debug_samples JSONs) into `EvalPairSpec`s."""
  pairs: list[EvalPairSpec] = []
  g_idx = 0
  for tid, tdata in sorted(tasks_dict.items()):
    for pidx, t_pair in enumerate(tdata.get("test", [])):
      pairs.append(
          EvalPairSpec(
              global_pair_idx=g_idx,
              task_id=tid,
              test_idx=pidx,
              train_pairs=tdata.get("train", []),
              test_input=t_pair["input"],
              ground_truth_output=t_pair.get("output"),
          )
      )
      g_idx += 1
  return pairs


def evaluate_pair_records(
    records: list[dict[str, Any]],
    label: str = "",
    print_report: bool = True,
) -> dict[str, Any]:
  """Compute Kaggle Pair Pass@1/Pass@2, Official ARC Prize Task Pass@1/Pass@2, Oracle, and TPS."""
  if not records:
    if print_report:
      print("No records to evaluate.")
    return {}
  n_pairs = len(records)
  pair_p1 = sum(1 for r in records if r["pass1"])
  pair_p2 = sum(1 for r in records if r["pass2"])
  pair_oracle = sum(1 for r in records if r.get("sel_oracle"))
  pair_ver = sum(1 for r in records if r.get("any_verified"))

  by_task: dict[str, list[dict[str, Any]]] = {}
  for r in records:
    by_task.setdefault(r["task_id"], []).append(r)

  n_tasks = len(by_task)
  task_p1_score = sum(
      sum(1.0 for r in plist if r["pass1"]) / len(plist)
      for plist in by_task.values()
  )
  task_p2_score = sum(
      sum(1.0 for r in plist if r["pass2"]) / len(plist)
      for plist in by_task.values()
  )
  task_oracle_score = sum(
      sum(1.0 for r in plist if r.get("sel_oracle")) / len(plist)
      for plist in by_task.values()
  )

  elapsed = float(records[0].get("elapsed_s", 0.0))
  total_toks = int(records[0].get("total_gen_tokens", 0))
  tps = float(records[0].get("tok_per_sec", 0.0))

  solved_task_ids: list[str] = []
  partial_task_ids: list[str] = []
  unsolved_task_ids: list[str] = []
  for tid, plist in sorted(by_task.items()):
    hits = sum(1 for r in plist if r["pass2"])
    if hits == len(plist):
      solved_task_ids.append(tid)
    elif hits > 0:
      partial_task_ids.append(tid)
    else:
      unsolved_task_ids.append(tid)

  if print_report:
    header = label or str(records[0].get("exp_id", "run"))
    print("=" * 88)
    print(f"CORRECTNESS REPORT: {header} ({n_tasks} tasks, {n_pairs} test pairs)")
    print("-" * 88)
    print(
        f"  Official Task Pass@1 : {task_p1_score:6.2f} / {n_tasks} "
        f"({100.0 * task_p1_score / n_tasks:5.2f}%)"
    )
    print(
        f"  Official Task Pass@2 : {task_p2_score:6.2f} / {n_tasks} "
        f"({100.0 * task_p2_score / n_tasks:5.2f}%)  <-- Kaggle Official Metric"
    )
    print(
        f"  Task Selection Oracle: {task_oracle_score:6.2f} / {n_tasks} "
        f"({100.0 * task_oracle_score / n_tasks:5.2f}%)"
    )
    print(
        f"  Pair Pass@1          : {pair_p1:6d} / {n_pairs} "
        f"({100.0 * pair_p1 / n_pairs:5.2f}%)"
    )
    print(
        f"  Pair Pass@2          : {pair_p2:6d} / {n_pairs} "
        f"({100.0 * pair_p2 / n_pairs:5.2f}%)"
    )
    print(
        f"  Verified Demo Pass   : {pair_ver:6d} / {n_pairs} "
        f"({100.0 * pair_ver / n_pairs:5.2f}%)"
    )
    print(
        f"  Throughput & Tokens  : {total_toks:,} tokens in {elapsed:.1f}s "
        f"({tps:.1f} tok/s)"
    )
    print("-" * 88)
    print(f"  SOLVED TASKS ({len(solved_task_ids)}/{n_tasks}): {solved_task_ids}")
    if partial_task_ids:
      print(f"  PARTIAL TASKS ({len(partial_task_ids)}/{n_tasks}): {partial_task_ids}")
    print(f"  UNSOLVED TASKS ({len(unsolved_task_ids)}/{n_tasks}): {unsolved_task_ids}")
    print("=" * 88)
    for r in records:
      status = (
          "SOLVED (PASS@1)"
          if r["pass1"]
          else (
              "SOLVED (PASS@2)"
              if r["pass2"]
              else ("UNSOLVED (ORACLE_TIE_LOSS)" if r.get("sel_oracle") else "UNSOLVED")
          )
      )
      ver_tag = "VERIFIED" if r.get("any_verified") else "unverified"
      cand_cnt = r.get("selector_diagnostics", {}).get("num_unique_candidates", 1)
      print(
          f"  [{r['task_id']} pair {r['test_idx']}] {status:26s} | "
          f"{ver_tag:10s} | cand_count={cand_cnt}"
      )

  return {
      "n_tasks": n_tasks,
      "n_pairs": n_pairs,
      "task_pass1_score": round(task_p1_score, 4),
      "task_pass2_score": round(task_p2_score, 4),
      "task_pass1_pct": round(100.0 * task_p1_score / n_tasks, 2),
      "task_pass2_pct": round(100.0 * task_p2_score / n_tasks, 2),
      "task_oracle_pct": round(100.0 * task_oracle_score / n_tasks, 2),
      "pair_pass1_pct": round(100.0 * pair_p1 / n_pairs, 2),
      "pair_pass2_pct": round(100.0 * pair_p2 / n_pairs, 2),
      "verified_pair_pct": round(100.0 * pair_ver / n_pairs, 2),
      "solved_task_ids": solved_task_ids,
      "partial_task_ids": partial_task_ids,
      "unsolved_task_ids": unsolved_task_ids,
      "total_gen_tokens": total_toks,
      "elapsed_s": round(elapsed, 2),
      "tok_per_sec": tps,
  }


def _get_engine_max_model_len(llm: Any, default: int = 131072) -> int:
  """Safely extract `max_model_len` from a live `vllm.LLM` instance."""
  try:
    return int(llm.llm_engine.model_config.max_model_len)
  except Exception:
    return default


def _count_prompt_tokens_safe(llm: Any, text: str) -> int:
  """Count prompt tokens using `llm.get_tokenizer()` with a conservative char fallback."""
  try:
    tok = llm.get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))
  except Exception:
    return len(text) // 3 + 64


def run_pairs_on_llm(
    llm: Any,
    tasks_or_pairs: dict[str, dict[str, Any]] | list[EvalPairSpec],
    exp_id: str = "h8_evolutionary_two_wave_repair",
    custom_cfg: prompts_and_selector.Pilot2HypothesisConfig | None = None,
    gated_early_stop: bool = False,
) -> list[dict[str, Any]]:
  """Run any `pilot_2` harness (`h1`..`h8` or custom config) directly on a live `vllm.LLM` instance (e.g. in Colab)."""
  import vllm  # pylint: disable=g-import-not-at-top

  cfg = (
      custom_cfg
      if custom_cfg is not None
      else prompts_and_selector.PILOT2_HYPOTHESES[exp_id]
  )
  cfg.validate()
  fb = prompts_and_selector.FROZEN_BASELINE_CONFIG
  todo_pairs = (
      tasks_dict_to_eval_pairs(tasks_or_pairs)
      if isinstance(tasks_or_pairs, dict)
      else list(tasks_or_pairs)
  )

  engine = llm.llm_engine
  engine_max_len = _get_engine_max_model_len(llm, fb["max_model_len"])
  final_only_kind = vllm.sampling_params.RequestOutputKind.FINAL_ONLY

  pending_streams: list[AttemptStreamState] = []
  attempts_by_pair: dict[int, list[AttemptStreamState]] = {}

  for pair_spec in todo_pairs:
    pair_streams: list[AttemptStreamState] = []
    for att_idx in range(cfg.num_attempts):
      sys_p = prompts_and_selector.build_system_prompt(cfg, att_idx)
      usr_p = prompts_and_selector.build_user_prompt(
          pair_spec.train_pairs, pair_spec.test_input, cfg, att_idx
      )
      init_transcript = prompts_and_selector.format_initial_chatml_transcript(
          sys_p, usr_p, use_native_xml_tools=cfg.use_native_xml_tools
      )
      seed = (
          cfg.base_seed
          + att_idx * cfg.attempt_seed_stride
          + pair_spec.global_pair_idx
      )
      sb = sandbox_verifier.Pilot2AttemptSandbox(
          train_pairs=pair_spec.train_pairs,
          test_input=pair_spec.test_input,
          use_arc_helpers=cfg.use_arc_helpers,
          auto_check_on_solve=cfg.auto_check_on_solve,
          run_equivariance_audit=cfg.use_equivariance_audit,
          max_output_chars=fb["max_tool_output_chars"],
          use_e031_ipython_mode=(
              cfg.attempt_prompt_styles[att_idx] == "lean2sub"
          ),
      )
      is_wave2 = bool(
          (cfg.use_evolutionary_two_wave or gated_early_stop) and att_idx >= 2
      )
      st = AttemptStreamState(
          pair_spec=pair_spec,
          attempt_idx=att_idx,
          cfg=cfg,
          sandbox=sb,
          system_prompt=sys_p,
          user_prompt=usr_p,
          transcript=init_transcript,
          seed=seed,
          waiting_for_wave1=is_wave2,
      )
      pair_streams.append(st)
      if not is_wave2:
        pending_streams.append(st)
    attempts_by_pair[pair_spec.global_pair_idx] = pair_streams

  active_requests: dict[str, AttemptStreamState] = {}
  rid_alias: dict[str, str] = {}
  pending_tool_futures: dict[
      futures.Future[tuple[str, bool, str]], AttemptStreamState
  ] = {}
  tool_pool = futures.ThreadPoolExecutor(max_workers=24)

  def _enqueue_generation(
      st: AttemptStreamState, is_nudge_grace: bool = False
  ) -> None:
    remaining_budget = max(256, st.cfg.total_token_budget - st.cum_gen_tokens)
    desired_toks = (
        min(st.cfg.turn_think_grace, remaining_budget)
        if is_nudge_grace
        else min(st.cfg.turn_think_cap, remaining_budget)
    )
    prompt_to_send = (
        st.transcript + st.partial_turn_text
        if is_nudge_grace
        else st.transcript
    )
    p_toks = _count_prompt_tokens_safe(llm, prompt_to_send)
    if (
        not is_nudge_grace
        and st.turn_idx >= 2
        and p_toks + desired_toks + 256 > engine_max_len
    ):
      st.transcript = prompts_and_selector.compact_chatml_transcript(
          st.transcript, keep_recent_turns=2, max_old_tool_chars=1200
      )
      prompt_to_send = st.transcript
      p_toks = _count_prompt_tokens_safe(llm, prompt_to_send)
    ctx_headroom = engine_max_len - p_toks - 128
    if ctx_headroom < 128:
      st.finished = True
      st.finish_reason = "context_length_exhausted"
      return
    max_toks = min(int(desired_toks), int(ctx_headroom))
    sp = vllm.SamplingParams(
        temperature=st.cfg.temperature,
        top_p=st.cfg.top_p,
        top_k=st.cfg.top_k,
        max_tokens=max(1, max_toks),
        stop=list(fb["stop_strings"]),
        include_stop_str_in_output=True,
    )
    sp.output_kind = final_only_kind
    rids = llm.enqueue([prompt_to_send], [sp], use_tqdm=False)
    internal = str(rids[0])
    external = internal.rsplit("-", 1)[0] if "-" in internal else internal
    active_requests[external] = st
    if internal != external:
      active_requests[internal] = st
      rid_alias[internal] = external
      rid_alias[external] = internal

  def _pop_active_request(rid: str) -> AttemptStreamState | None:
    st = active_requests.pop(rid, None)
    other = rid_alias.pop(rid, None)
    if other is not None:
      active_requests.pop(other, None)
      rid_alias.pop(other, None)
    return st

  def _maybe_schedule_wave2() -> None:
    if not (cfg.use_evolutionary_two_wave or gated_early_stop):
      return
    for pair_spec in todo_pairs:
      p_streams = attempts_by_pair[pair_spec.global_pair_idx]
      if len(p_streams) < 4 or not (
          p_streams[0].finished and p_streams[1].finished
      ):
        continue
      wave2_waiting = [s for s in p_streams[2:] if s.waiting_for_wave1]
      if not wave2_waiting:
        continue
      if gated_early_stop:
        p0 = p_streams[0].sandbox.first_tp_pred
        p1 = p_streams[1].sandbox.first_tp_pred
        if p0 is not None and p1 is not None and grids_equal(p0, p1):
          for s in wave2_waiting:
            s.waiting_for_wave1 = False
            s.finished = True
            s.finish_reason = "gated_early_stop_wave1_consensus"
          continue
      wave1_sbs = [p_streams[0].sandbox, p_streams[1].sandbox]
      for s in reversed(wave2_waiting):
        seed_note = (
            prompts_and_selector.build_wave2_evolutionary_seed(
                wave1_sbs, s.attempt_idx
            )
            if cfg.use_evolutionary_two_wave
            else ""
        )
        s.user_prompt = prompts_and_selector.build_user_prompt(
            pair_spec.train_pairs,
            pair_spec.test_input,
            cfg,
            s.attempt_idx,
            extra_seed_note=seed_note,
        )
        s.transcript = prompts_and_selector.format_initial_chatml_transcript(
            s.system_prompt,
            s.user_prompt,
            use_native_xml_tools=cfg.use_native_xml_tools,
        )
        s.waiting_for_wave1 = False
        pending_streams.insert(0, s)

  t_start = time.perf_counter()
  total_gen_tokens_all = 0
  try:
    while (
        pending_streams
        or engine.has_unfinished_requests()
        or pending_tool_futures
    ):
      done_futs = [f for f in list(pending_tool_futures.keys()) if f.done()]
      for fut in done_futs:
        st = pending_tool_futures.pop(fut)
        tool_resp, should_term, term_reason = fut.result()
        st.turn_idx += 1
        if should_term:
          st.finished = True
          st.finish_reason = term_reason
          continue
        hit_cap = (
            st.turn_idx >= st.cfg.max_turns
            or st.cum_gen_tokens >= st.cfg.total_token_budget - 512
        )
        if hit_cap:
          if st.sandbox.use_e031_ipython_mode:
            st.sandbox.finalize_e031_session_fallback(
                st.turn_idx, st.cum_gen_tokens
            )
          if (
              st.cfg.use_two_stage_think_nudge
              and st.sandbox.first_tp_pred is None
              and st.sandbox.submitted_answer_grid is None
              and st.nudge_stage != 3
          ):
            st.transcript = (
                prompts_and_selector.append_turn_to_chatml_transcript(
                    transcript=st.transcript,
                    assistant_output=st.partial_turn_text,
                    tool_response=tool_resp,
                    budget_note=(
                        "[Final turn: submit your best general `def solve(grid):` now.]"
                    ),
                    use_native_xml_tools=st.cfg.use_native_xml_tools,
                )
            )
            st.nudge_stage = 3
            st.in_nudge_continuation = True
            st.partial_turn_text = (
                prompts_and_selector.E031_FORCE_SUBMIT_FINAL
                if st.sandbox.use_e031_ipython_mode
                else prompts_and_selector.XML_FORCE_SUBMIT_FINAL
            )
            st.partial_turn_tokens = 0
            _enqueue_generation(st, is_nudge_grace=True)
            continue
          st.finished = True
          st.finish_reason = (
              "max_turns_reached"
              if st.turn_idx >= st.cfg.max_turns
              else "token_budget_exhausted"
          )
          continue

        if st.sandbox.use_e031_ipython_mode:
          b_note = prompts_and_selector.format_e031_budget_and_turns_note(
              cum_gen_tokens=st.cum_gen_tokens,
              turn_idx_after_tool=st.turn_idx,
              max_turns=st.cfg.max_turns,
              soft_target=st.cfg.soft_token_target,
              hard_budget=st.cfg.total_token_budget,
          )
        else:
          notes: list[str] = []
          turns_left = st.cfg.max_turns - st.turn_idx
          if turns_left <= 4:
            notes.append(
                f"[Notice: {turns_left} tool turns remaining. Finalize `def solve(grid):` and call `submit_program`.]"
            )
          elif st.cum_gen_tokens >= st.cfg.soft_token_target:
            notes.append(
                f"[Budget notice: {st.cum_gen_tokens}/{st.cfg.total_token_budget} tokens used. "
                "Focus on verifying `def solve(grid):` and calling `submit_program`.]"
            )
          if (
              st.cfg.use_two_stage_think_nudge
              and st.cum_gen_tokens >= 20000
              and not st.sandbox.has_defined_solve
          ):
            notes.append(
                "[Commit notice: Define `def solve(grid):` in your next `run_python` cell "
                "so the harness auto-checks it on every training pair and gives you exact cell diffs.]"
            )
          b_note = " ".join(notes)
        st.transcript = prompts_and_selector.append_turn_to_chatml_transcript(
            transcript=st.transcript,
            assistant_output=st.partial_turn_text,
            tool_response=tool_resp,
            budget_note=b_note,
            use_native_xml_tools=st.cfg.use_native_xml_tools,
        )
        if (
            st.cfg.use_transcript_compaction
            and st.turn_idx >= 3
            and len(st.transcript) > 56000
        ):
          st.transcript = prompts_and_selector.compact_chatml_transcript(
              st.transcript, keep_recent_turns=2, max_old_tool_chars=1500
          )
        st.partial_turn_text = ""
        st.partial_turn_tokens = 0
        st.in_nudge_continuation = False
        st.nudge_stage = 0
        _enqueue_generation(st, is_nudge_grace=False)

      _maybe_schedule_wave2()

      active_cnt = len({id(x) for x in active_requests.values()}) + len(
          pending_tool_futures
      )
      while pending_streams and active_cnt < fb["max_num_seqs"]:
        _enqueue_generation(pending_streams.pop(0), is_nudge_grace=False)
        active_cnt = len({id(x) for x in active_requests.values()}) + len(
            pending_tool_futures
        )

      if engine.has_unfinished_requests():
        for ro in engine.step():
          if not getattr(ro, "finished", False):
            continue
          st = _pop_active_request(str(ro.request_id))
          if st is None:
            continue
          comp = ro.outputs[0]
          gen_text = comp.text or ""
          gen_toks = len(comp.token_ids)
          total_gen_tokens_all += gen_toks
          combined_text = st.partial_turn_text + gen_text
          combined_toks = st.partial_turn_tokens + gen_toks

          has_complete_tool_call = (
              "</tool_call>" in combined_text or "</function>" in combined_text
          )
          if (
              not has_complete_tool_call
              and str(comp.finish_reason) == "length"
              and (
                  st.cum_gen_tokens + combined_toks + 512
                  < st.cfg.total_token_budget
              )
          ):
            if st.cfg.use_two_stage_think_nudge:
              if (
                  st.nudge_stage == 0
                  and "</think>" not in combined_text
                  and "<tool_call>" not in combined_text
              ):
                st.nudge_stage = 1
                st.in_nudge_continuation = True
                st.partial_turn_text = (
                    combined_text
                    + prompts_and_selector.XML_THINK_NUDGE_STAGE1
                )
                st.partial_turn_tokens = combined_toks
                _enqueue_generation(st, is_nudge_grace=True)
                continue
              if st.nudge_stage < 2:
                st.nudge_stage = 2
                st.in_nudge_continuation = True
                if "<tool_call>" in combined_text:
                  st.partial_turn_text = combined_text
                elif "</think>" in combined_text:
                  st.partial_turn_text = (
                      combined_text
                      + "\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n"
                  )
                else:
                  force_act = (
                      prompts_and_selector.XML_COMMIT_FORCE_ACT_STAGE2
                      if (
                          not st.sandbox.use_e031_ipython_mode
                          and st.cum_gen_tokens + combined_toks >= 20000
                          and not st.sandbox.has_defined_solve
                      )
                      else prompts_and_selector.XML_THINK_FORCE_ACT_STAGE2
                  )
                  st.partial_turn_text = combined_text + force_act
                st.partial_turn_tokens = combined_toks
                _enqueue_generation(st, is_nudge_grace=True)
                continue
            elif not st.in_nudge_continuation:
              st.in_nudge_continuation = True
              st.partial_turn_text = (
                  combined_text
                  if "<tool_call>" in combined_text
                  else combined_text + fb["think_nudge"]
              )
              st.partial_turn_tokens = combined_toks
              _enqueue_generation(st, is_nudge_grace=True)
              continue

          st.partial_turn_text = combined_text
          st.partial_turn_tokens = combined_toks
          fut = tool_pool.submit(
              execute_tool_for_attempt, st, combined_text, combined_toks
          )
          pending_tool_futures[fut] = st
      else:
        time.sleep(0.01)

      _maybe_schedule_wave2()
  finally:
    tool_pool.shutdown(wait=True)

  elapsed = time.perf_counter() - t_start
  results: list[dict[str, Any]] = []
  for pair_spec in todo_pairs:
    p_streams = [
        s
        for s in attempts_by_pair[pair_spec.global_pair_idx]
        if s.finish_reason != "gated_early_stop_wave1_consensus"
    ]
    att_summaries = [build_attempt_candidate_summary(s) for s in p_streams]
    pred_1, pred_2, sel_diag = prompts_and_selector.select_top_two_predictions(
        attempts=att_summaries,
        test_input=pair_spec.test_input,
        use_equivariance_tiebreak=cfg.use_equivariance_audit,
    )
    gt = pair_spec.ground_truth_output
    hit_1 = grids_equal(pred_1, gt)
    hit_2 = hit_1 or grids_equal(pred_2, gt)
    sel_oracle = any(
        grids_equal(g, gt)
        for a in att_summaries
        for g in (
            [a.verified_grid, a.partial_grid, a.fallback_grid]
            + [x[0] for x in a.secondary_verified_grids]
        )
    )
    results.append({
        "exp_id": cfg.exp_id,
        "global_pair_idx": pair_spec.global_pair_idx,
        "task_id": pair_spec.task_id,
        "test_idx": pair_spec.test_idx,
        "pass1": hit_1,
        "pass2": hit_2,
        "sel_oracle": sel_oracle,
        "any_verified": any(a.train_pass for a in att_summaries),
        "attempt_1": pred_1,
        "attempt_2": pred_2,
        "ground_truth": gt,
        "selector_diagnostics": sel_diag,
        "streams": p_streams,
        "elapsed_s": elapsed,
        "total_gen_tokens": total_gen_tokens_all,
        "tok_per_sec": round(total_gen_tokens_all / max(1e-6, elapsed), 1),
    })
  return results


def run_no_tools_on_llm(
    llm: Any,
    tasks_or_pairs: dict[str, dict[str, Any]] | list[EvalPairSpec],
    num_attempts: int = 2,
    max_tokens: int = 48000,
) -> list[dict[str, Any]]:
  """Run exact `exp1_no_tools_48k` baseline directly on a live `vllm.LLM` instance."""
  import vllm  # pylint: disable=g-import-not-at-top
  from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import prompts_and_voting  # pylint: disable=g-import-not-at-top

  pairs = (
      tasks_dict_to_eval_pairs(tasks_or_pairs)
      if isinstance(tasks_or_pairs, dict)
      else list(tasks_or_pairs)
  )
  engine_max_len = _get_engine_max_model_len(llm, 131072)
  prompts = []
  sp_list = []
  meta = []
  for p in pairs:
    chatml = prompts_and_voting.build_initial_transcript(
        task={"train": p.train_pairs},
        test_input=p.test_input,
        use_tools=False,
        soft_target=max_tokens // 2,
        hard_budget=max_tokens,
    )
    p_toks = _count_prompt_tokens_safe(llm, chatml)
    eff_max_tokens = min(int(max_tokens), max(256, engine_max_len - p_toks - 128))
    for a_idx in range(num_attempts):
      prompts.append(chatml)
      sp_list.append(
          vllm.SamplingParams(
              temperature=1.0,
              top_p=0.95,
              top_k=20,
              max_tokens=eff_max_tokens,
              stop=["<|im_end|>"],
          )
      )
      meta.append((p, a_idx))

  t0 = time.perf_counter()
  outputs = llm.generate(prompts, sp_list, use_tqdm=True)
  elapsed = time.perf_counter() - t0

  by_pair: dict[int, list[tuple[list[list[int]], int, str]]] = {}
  total_toks = 0
  for (p, _), ro in zip(meta, outputs):
    txt = ro.outputs[0].text or ""
    toks = len(ro.outputs[0].token_ids)
    total_toks += toks
    pred = prompts_and_voting.parse_last_grid(txt) or sandbox_verifier.to_grid(
        p.test_input
    )
    by_pair.setdefault(p.global_pair_idx, []).append((pred, toks, txt))

  records = []
  for p in pairs:
    atts = by_pair[p.global_pair_idx]
    p1 = atts[0][0]
    p2 = atts[1][0] if len(atts) > 1 else p1
    gt = p.ground_truth_output
    hit1 = grids_equal(p1, gt)
    hit2 = hit1 or grids_equal(p2, gt)
    records.append({
        "exp_id": "exp1_no_tools_48k",
        "global_pair_idx": p.global_pair_idx,
        "task_id": p.task_id,
        "test_idx": p.test_idx,
        "pass1": hit1,
        "pass2": hit2,
        "sel_oracle": any(grids_equal(a[0], gt) for a in atts),
        "any_verified": False,
        "attempt_1": p1,
        "attempt_2": p2,
        "ground_truth": gt,
        "elapsed_s": elapsed,
        "total_gen_tokens": total_toks,
        "tok_per_sec": round(total_toks / max(1e-6, elapsed), 1),
    })
  return records


def main(argv: list[str]) -> None:
  del argv
  if FLAGS.exp_id not in prompts_and_selector.PILOT2_HYPOTHESES:
    raise ValueError(
        f"Unknown --exp_id={FLAGS.exp_id}. Valid choices: "
        f"{sorted(prompts_and_selector.PILOT2_HYPOTHESES.keys())}"
    )

  cfg = prompts_and_selector.PILOT2_HYPOTHESES[FLAGS.exp_id]
  cfg.validate()

  out_dir = FLAGS.output_dir or os.path.join(
      prompts_and_selector.FROZEN_BASELINE_CONFIG["cns_output_root"],
      cfg.exp_id,
  )
  gfile.MakeDirs(out_dir)

  shard_tag = f"shard_{FLAGS.shard_index:02d}_of_{FLAGS.num_shards:02d}"
  config_path = os.path.join(out_dir, f"config_{shard_tag}.json")
  transcripts_path = os.path.join(out_dir, f"transcripts_{shard_tag}.jsonl")
  code_traces_path = os.path.join(out_dir, f"code_traces_{shard_tag}.jsonl")
  pairs_path = os.path.join(out_dir, f"pairs_{shard_tag}.jsonl")
  submission_path = os.path.join(out_dir, f"submission_{shard_tag}.json")
  summary_path = os.path.join(out_dir, f"summary_{shard_tag}.json")

  _safe_cns_write_json(
      config_path,
      {
          "hypothesis_config": dataclasses.asdict(cfg),
          "frozen_baseline_invariants": prompts_and_selector.FROZEN_BASELINE_CONFIG,
          "shard_index": FLAGS.shard_index,
          "num_shards": FLAGS.num_shards,
          "model_path": FLAGS.model_path,
          "dataset_dir": FLAGS.dataset_dir,
      },
      indent=2,
  )

  eval_pairs = load_eval_pairs(
      dataset_dir=FLAGS.dataset_dir,
      shard_index=FLAGS.shard_index,
      num_shards=FLAGS.num_shards,
      max_tasks=FLAGS.max_tasks,
  )

  (
      pair_finalized,
      pass1_hits,
      pass2_hits,
      sel_oracle_hits,
      diag_oracle_hits,
      verified_pair_hits,
      submission_dict,
  ) = load_existing_completed_pairs(pairs_path)
  completed_pairs_count = len(pair_finalized)

  todo_pairs = [p for p in eval_pairs if p.global_pair_idx not in pair_finalized]
  print(
      f"[pilot_2] Starting {cfg.exp_id} ({cfg.title}) | {shard_tag} | "
      f"total_shard_pairs={len(eval_pairs)}, already_done={completed_pairs_count}, "
      f"todo={len(todo_pairs)} | num_attempts={cfg.num_attempts} | "
      f"max_turns={cfg.max_turns} | budget={cfg.total_token_budget}",
      flush=True,
  )
  if FLAGS.lora_adapter_path or FLAGS.full_sft_ckpt_dir:
    from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize  # pylint: disable=g-import-not-at-top
    export_and_quantize.ensure_sharded_fp8_export(
        lora_adapter_path=FLAGS.lora_adapter_path,
        output_fp8_path=FLAGS.model_path,
        shard_index=FLAGS.shard_index,
        num_shards=FLAGS.num_shards,
        full_sft_ckpt_dir=FLAGS.full_sft_ckpt_dir,
    )

  if not todo_pairs:
    print(f"[pilot_2] All {len(eval_pairs)} pairs already completed for {shard_tag}.", flush=True)
    return

  import jax  # pylint: disable=g-import-not-at-top
  import vllm  # pylint: disable=g-import-not-at-top

  fb = prompts_and_selector.FROZEN_BASELINE_CONFIG
  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  t_init0 = time.time()
  llm = vllm.LLM(
      model=FLAGS.model_path,
      tensor_parallel_size=fb["tensor_parallel_size"],
      max_model_len=fb["max_model_len"],
      max_num_batched_tokens=fb["max_num_batched_tokens"],
      max_num_seqs=fb["max_num_seqs"],
      gpu_memory_utilization=fb["gpu_memory_utilization"],
      enable_prefix_caching=fb["enable_prefix_caching"],
      load_format="tpu_streaming_loader",
  )
  engine = llm.llm_engine
  final_only_kind = vllm.sampling_params.RequestOutputKind.FINAL_ONLY
  print(
      f"[pilot_2] vLLM initialized in {time.time() - t_init0:.1f}s on {num_tpus}x {device_kind}",
      flush=True,
  )

  pending_streams: list[AttemptStreamState] = []
  attempts_by_pair: dict[int, list[AttemptStreamState]] = {}
  for pair_spec in todo_pairs:
    pair_streams: list[AttemptStreamState] = []
    for att_idx in range(cfg.num_attempts):
      sys_p = prompts_and_selector.build_system_prompt(cfg, att_idx)
      usr_p = prompts_and_selector.build_user_prompt(
          pair_spec.train_pairs, pair_spec.test_input, cfg, att_idx
      )
      init_transcript = prompts_and_selector.format_initial_chatml_transcript(
          sys_p,
          usr_p,
          use_native_xml_tools=cfg.use_native_xml_tools,
      )
      seed = (
          cfg.base_seed
          + att_idx * cfg.attempt_seed_stride
          + pair_spec.global_pair_idx
      )
      sb = sandbox_verifier.Pilot2AttemptSandbox(
          train_pairs=pair_spec.train_pairs,
          test_input=pair_spec.test_input,
          use_arc_helpers=cfg.use_arc_helpers,
          auto_check_on_solve=cfg.auto_check_on_solve,
          run_equivariance_audit=cfg.use_equivariance_audit,
          max_output_chars=fb["max_tool_output_chars"],
          use_e031_ipython_mode=(
              cfg.attempt_prompt_styles[att_idx] == "lean2sub"
          ),
      )
      is_wave2 = bool(cfg.use_evolutionary_two_wave and att_idx >= 2)
      st = AttemptStreamState(
          pair_spec=pair_spec,
          attempt_idx=att_idx,
          cfg=cfg,
          sandbox=sb,
          system_prompt=sys_p,
          user_prompt=usr_p,
          transcript=init_transcript,
          seed=seed,
          waiting_for_wave1=is_wave2,
      )
      pair_streams.append(st)
      if not is_wave2:
        pending_streams.append(st)
    attempts_by_pair[pair_spec.global_pair_idx] = pair_streams

  max_concurrent_requests = fb["max_num_seqs"]
  active_requests: dict[str, AttemptStreamState] = {}
  rid_alias: dict[str, str] = {}
  pending_tool_futures: dict[futures.Future[tuple[str, bool, str]], AttemptStreamState] = {}

  def _enqueue_generation(st: AttemptStreamState, is_nudge_grace: bool = False) -> None:
    remaining_budget = max(256, st.cfg.total_token_budget - st.cum_gen_tokens)
    max_toks = (
        min(st.cfg.turn_think_grace, remaining_budget)
        if is_nudge_grace
        else min(st.cfg.turn_think_cap, remaining_budget)
    )
    prompt_to_send = (
        st.transcript + st.partial_turn_text if is_nudge_grace else st.transcript
    )
    engine_max_len = 131072
    prompt_toks = _count_prompt_tokens_safe(llm, prompt_to_send)
    if prompt_toks + max_toks > engine_max_len - 256:
      st.transcript = prompts_and_selector.compact_chatml_transcript(
          st.transcript, keep_recent_turns=1, max_old_tool_chars=500
      )
      prompt_to_send = (
          st.transcript + st.partial_turn_text
          if is_nudge_grace
          else st.transcript
      )
      prompt_toks = _count_prompt_tokens_safe(llm, prompt_to_send)
      max_toks = max(1, min(int(max_toks), engine_max_len - 256 - prompt_toks))

    sp = vllm.SamplingParams(
        temperature=st.cfg.temperature,
        top_p=st.cfg.top_p,
        top_k=st.cfg.top_k,
        max_tokens=max(1, int(max_toks)),
        stop=list(fb["stop_strings"]),
        include_stop_str_in_output=True,
    )
    sp.output_kind = final_only_kind
    rids = llm.enqueue([prompt_to_send], [sp], use_tqdm=False)
    internal = str(rids[0])
    external = internal.rsplit("-", 1)[0] if "-" in internal else internal
    active_requests[external] = st
    if internal != external:
      active_requests[internal] = st
      rid_alias[internal] = external
      rid_alias[external] = internal

  def _pop_active_request(rid: str) -> AttemptStreamState | None:
    st = active_requests.pop(rid, None)
    other = rid_alias.pop(rid, None)
    if other is not None:
      active_requests.pop(other, None)
      rid_alias.pop(other, None)
    return st

  start_time = time.time()
  total_gen_tokens_all = 0
  tool_pool = futures.ThreadPoolExecutor(max_workers=24)

  def _maybe_finalize_pairs() -> None:
    nonlocal completed_pairs_count, pass1_hits, pass2_hits
    nonlocal sel_oracle_hits, diag_oracle_hits, verified_pair_hits
    for pair_spec in todo_pairs:
      g_idx = pair_spec.global_pair_idx
      if g_idx in pair_finalized:
        continue
      p_streams = attempts_by_pair[g_idx]
      if not all(s.finished for s in p_streams):
        continue

      pair_finalized.add(g_idx)
      completed_pairs_count += 1

      att_summaries: list[prompts_and_selector.AttemptCandidateSummary] = []
      transcript_lines: list[str] = []
      code_trace_lines: list[str] = []

      for s in p_streams:
        att_sum = build_attempt_candidate_summary(s)
        att_summaries.append(att_sum)

        transcript_lines.append(
            json.dumps({
                "exp_id": cfg.exp_id,
                "global_pair_idx": g_idx,
                "task_id": pair_spec.task_id,
                "test_idx": pair_spec.test_idx,
                "attempt_idx": s.attempt_idx,
                "prompt_style": cfg.attempt_prompt_styles[s.attempt_idx],
                "has_task_summary": cfg.attempt_include_task_summary[s.attempt_idx],
                "seed": s.seed,
                "total_turns": s.turn_idx,
                "total_gen_tokens": s.cum_gen_tokens,
                "finish_reason": s.finish_reason,
                "train_pass": att_sum.train_pass,
                "verified_turn": att_sum.verified_turn,
                "verified_gen_tokens": att_sum.verified_gen_tokens,
                "system_prompt": s.system_prompt,
                "user_prompt": s.user_prompt,
                "turn_records": s.turn_records,
            })
            + "\n"
        )

        code_trace_lines.append(
            json.dumps({
                "exp_id": cfg.exp_id,
                "global_pair_idx": g_idx,
                "task_id": pair_spec.task_id,
                "test_idx": pair_spec.test_idx,
                "attempt_idx": s.attempt_idx,
                "train_pass": att_sum.train_pass,
                "first_tp_turn": s.sandbox.first_tp_turn,
                "first_tp_gen_tokens": s.sandbox.first_tp_gen_tokens,
                "first_tp_equivariance_bonus": s.sandbox.first_tp_equivariance_bonus,
                "first_tp_code": s.sandbox.first_tp_code,
                "latest_tp_code": s.sandbox.latest_tp_code,
                "best_partial_train_frac": s.sandbox.best_partial_train_frac,
                "best_partial_cell_acc": s.sandbox.best_partial_cell_acc,
                "code_turns": [dataclasses.asdict(ct) for ct in s.sandbox.code_traces],
            })
            + "\n"
        )

      _safe_cns_append(transcripts_path, "".join(transcript_lines))
      _safe_cns_append(code_traces_path, "".join(code_trace_lines))

      pred_1, pred_2, sel_diag = prompts_and_selector.select_top_two_predictions(
          attempts=att_summaries,
          test_input=pair_spec.test_input,
          use_equivariance_tiebreak=cfg.use_equivariance_audit,
      )

      gt = pair_spec.ground_truth_output
      hit_1 = grids_equal(pred_1, gt)
      hit_2 = hit_1 or grids_equal(pred_2, gt)

      sel_oracle = False
      diag_oracle = False
      any_verified = any(a.train_pass for a in att_summaries)
      for a_sum, s in zip(att_summaries, p_streams):
        for g_cand in (a_sum.verified_grid, a_sum.partial_grid, a_sum.fallback_grid):
          if grids_equal(g_cand, gt):
            sel_oracle = True
            diag_oracle = True
        for sec_g, _, _, _ in a_sum.secondary_verified_grids:
          if grids_equal(sec_g, gt):
            sel_oracle = True
            diag_oracle = True
        if grids_equal(s.sandbox.latest_tp_pred, gt):
          diag_oracle = True

      if hit_1:
        pass1_hits += 1
      if hit_2:
        pass2_hits += 1
      if sel_oracle:
        sel_oracle_hits += 1
      if diag_oracle:
        diag_oracle_hits += 1
      if any_verified:
        verified_pair_hits += 1

      submission_dict.setdefault(pair_spec.task_id, []).append({
          "test_idx": pair_spec.test_idx,
          "attempt_1": pred_1,
          "attempt_2": pred_2,
      })

      pair_record = {
          "exp_id": cfg.exp_id,
          "global_pair_idx": g_idx,
          "task_id": pair_spec.task_id,
          "test_idx": pair_spec.test_idx,
          "pass1": hit_1,
          "pass2": hit_2,
          "sel_oracle": sel_oracle,
          "diag_oracle": diag_oracle,
          "any_verified": any_verified,
          "attempt_1": pred_1,
          "attempt_2": pred_2,
          "selector_diagnostics": sel_diag,
          "attempts_overview": [
              {
                  "attempt_idx": a.attempt_idx,
                  "prompt_style": a.prompt_style,
                  "has_task_summary": a.has_task_summary,
                  "total_turns": a.total_turns,
                  "total_gen_tokens": a.total_gen_tokens,
                  "train_pass": a.train_pass,
                  "verified_turn": a.verified_turn,
                  "verified_gen_tokens": a.verified_gen_tokens,
                  "equivariance_bonus": a.equivariance_bonus,
                  "partial_train_frac": round(a.partial_train_frac, 4),
                  "matches_gt": grids_equal(
                      a.verified_grid or a.partial_grid or a.fallback_grid, gt
                  ),
              }
              for a in att_summaries
          ],
      }
      _safe_cns_append(pairs_path, json.dumps(pair_record) + "\n")

      clean_sub_inc: dict[str, list[dict[str, Any]]] = {}
      for tid, entries in sorted(submission_dict.items()):
        entries_sorted = sorted(entries, key=lambda x: int(x["test_idx"]))
        clean_sub_inc[tid] = [
            {"attempt_1": e["attempt_1"], "attempt_2": e["attempt_2"]}
            for e in entries_sorted
        ]
      _safe_cns_write_json(submission_path, clean_sub_inc)

      elapsed = time.time() - start_time
      p1_pct = 100.0 * pass1_hits / completed_pairs_count
      p2_pct = 100.0 * pass2_hits / completed_pairs_count
      so_pct = 100.0 * sel_oracle_hits / completed_pairs_count
      do_pct = 100.0 * diag_oracle_hits / completed_pairs_count
      ver_pct = 100.0 * verified_pair_hits / completed_pairs_count
      pairs_left = len(eval_pairs) - completed_pairs_count
      sec_per_pair = elapsed / max(1, completed_pairs_count)
      eta_min = (pairs_left * sec_per_pair) / 60.0

      summary_obj = {
          "exp_id": cfg.exp_id,
          "title": cfg.title,
          "shard_index": FLAGS.shard_index,
          "num_shards": FLAGS.num_shards,
          "status": "completed" if completed_pairs_count == len(eval_pairs) else "running",
          "completed_pairs": completed_pairs_count,
          "total_pairs_in_shard": len(eval_pairs),
          "pass1_count": pass1_hits,
          "pass1_pct": round(p1_pct, 2),
          "pass2_count": pass2_hits,
          "pass2_pct": round(p2_pct, 2),
          "sel_oracle_count": sel_oracle_hits,
          "sel_oracle_pct": round(so_pct, 2),
          "diag_oracle_count": diag_oracle_hits,
          "diag_oracle_pct": round(do_pct, 2),
          "verified_pairs_count": verified_pair_hits,
          "verified_pairs_pct": round(ver_pct, 2),
          "total_gen_tokens": total_gen_tokens_all,
          "elapsed_sec": round(elapsed, 1),
          "eta_minutes": round(eta_min, 1),
          "gen_tok_per_sec": round(total_gen_tokens_all / max(1.0, elapsed), 1),
      }
      _safe_cns_write_json(summary_path, summary_obj, indent=2)

      # Free heavy turn logs from memory once flushed to CNS.
      for s in p_streams:
        s.turn_records.clear()
        s.sandbox.code_traces.clear()

      print(
          f"[{cfg.exp_id} | {shard_tag}] Pair {completed_pairs_count}/{len(eval_pairs)} "
          f"({pair_spec.task_id}[{pair_spec.test_idx}]) -> "
          f"pass@1={pass1_hits}/{completed_pairs_count} ({p1_pct:.2f}%), "
          f"pass@2={pass2_hits}/{completed_pairs_count} ({p2_pct:.2f}%), "
          f"sel_oracle={so_pct:.2f}%, diag_oracle={do_pct:.2f}% | "
          f"elapsed={elapsed/60:.1f}m, ETA={eta_min:.1f}m ({summary_obj['gen_tok_per_sec']} tok/s)",
          flush=True,
      )

  last_heartbeat_time = time.time()
  _safe_cns_write_json(
      summary_path,
      {
          "exp_id": cfg.exp_id,
          "title": cfg.title,
          "shard_index": FLAGS.shard_index,
          "num_shards": FLAGS.num_shards,
          "status": "running",
          "completed_pairs": completed_pairs_count,
          "total_pairs_in_shard": len(eval_pairs),
          "pass1_count": pass1_hits,
          "pass1_pct": round(100.0 * pass1_hits / max(1, completed_pairs_count), 2) if completed_pairs_count else 0.0,
          "pass2_count": pass2_hits,
          "pass2_pct": round(100.0 * pass2_hits / max(1, completed_pairs_count), 2) if completed_pairs_count else 0.0,
          "sel_oracle_count": sel_oracle_hits,
          "sel_oracle_pct": round(100.0 * sel_oracle_hits / max(1, completed_pairs_count), 2) if completed_pairs_count else 0.0,
          "diag_oracle_count": diag_oracle_hits,
          "diag_oracle_pct": round(100.0 * diag_oracle_hits / max(1, completed_pairs_count), 2) if completed_pairs_count else 0.0,
          "verified_pairs_count": verified_pair_hits,
          "verified_pairs_pct": round(100.0 * verified_pair_hits / max(1, completed_pairs_count), 2) if completed_pairs_count else 0.0,
          "total_gen_tokens": 0,
          "elapsed_sec": 0.0,
          "eta_minutes": None,
          "gen_tok_per_sec": 0.0,
      },
      indent=2,
  )

  def _maybe_schedule_wave2() -> None:
    if not cfg.use_evolutionary_two_wave:
      return
    for pair_spec in todo_pairs:
      g_idx = pair_spec.global_pair_idx
      if g_idx in pair_finalized:
        continue
      p_streams = attempts_by_pair[g_idx]
      if len(p_streams) < 4:
        continue
      if not (p_streams[0].finished and p_streams[1].finished):
        continue
      wave2_waiting = [s for s in p_streams[2:] if s.waiting_for_wave1]
      if not wave2_waiting:
        continue
      wave1_sbs = [p_streams[0].sandbox, p_streams[1].sandbox]
      for s in reversed(wave2_waiting):
        seed_note = prompts_and_selector.build_wave2_evolutionary_seed(
            wave1_sbs, s.attempt_idx
        )
        s.user_prompt = prompts_and_selector.build_user_prompt(
            pair_spec.train_pairs,
            pair_spec.test_input,
            cfg,
            s.attempt_idx,
            extra_seed_note=seed_note,
        )
        s.transcript = prompts_and_selector.format_initial_chatml_transcript(
            s.system_prompt,
            s.user_prompt,
            use_native_xml_tools=cfg.use_native_xml_tools,
        )
        s.waiting_for_wave1 = False
        pending_streams.insert(0, s)

  try:
    while (
        pending_streams
        or engine.has_unfinished_requests()
        or pending_tool_futures
    ):
      done_futs = [f for f in list(pending_tool_futures.keys()) if f.done()]
      for fut in done_futs:
        st = pending_tool_futures.pop(fut)
        tool_resp, should_term, term_reason = fut.result()
        st.turn_idx += 1

        if should_term:
          st.finished = True
          st.finish_reason = term_reason
          continue

        hit_turn_or_budget_cap = (
            st.turn_idx >= st.cfg.max_turns
            or st.cum_gen_tokens >= st.cfg.total_token_budget - 512
        )
        if hit_turn_or_budget_cap:
          if st.sandbox.use_e031_ipython_mode:
            st.sandbox.finalize_e031_session_fallback(
                st.turn_idx, st.cum_gen_tokens
            )
          if (
              st.cfg.use_two_stage_think_nudge
              and st.sandbox.first_tp_pred is None
              and st.sandbox.submitted_answer_grid is None
              and st.nudge_stage != 3
          ):
            st.transcript = prompts_and_selector.append_turn_to_chatml_transcript(
                transcript=st.transcript,
                assistant_output=st.partial_turn_text,
                tool_response=tool_resp,
                budget_note="[Final turn: submit your best general `def solve(grid):` now.]",
                use_native_xml_tools=st.cfg.use_native_xml_tools,
            )
            st.nudge_stage = 3
            st.in_nudge_continuation = True
            st.partial_turn_text = (
                prompts_and_selector.E031_FORCE_SUBMIT_FINAL
                if st.sandbox.use_e031_ipython_mode
                else prompts_and_selector.XML_FORCE_SUBMIT_FINAL
            )
            st.partial_turn_tokens = 0
            _enqueue_generation(st, is_nudge_grace=True)
            continue
          st.finished = True
          st.finish_reason = (
              "max_turns_reached"
              if st.turn_idx >= st.cfg.max_turns
              else "token_budget_exhausted"
          )
          continue

        if st.sandbox.use_e031_ipython_mode:
          b_note = prompts_and_selector.format_e031_budget_and_turns_note(
              cum_gen_tokens=st.cum_gen_tokens,
              turn_idx_after_tool=st.turn_idx,
              max_turns=st.cfg.max_turns,
              soft_target=st.cfg.soft_token_target,
              hard_budget=st.cfg.total_token_budget,
          )
        else:
          notes: list[str] = []
          turns_left = st.cfg.max_turns - st.turn_idx
          if turns_left <= 4:
            notes.append(
                f"[Notice: {turns_left} tool turns remaining. Finalize `def solve(grid):` and call `submit_program`.]"
            )
          elif st.cum_gen_tokens >= st.cfg.soft_token_target:
            notes.append(
                f"[Budget notice: {st.cum_gen_tokens}/{st.cfg.total_token_budget} tokens used. "
                "Focus on verifying `def solve(grid):` and calling `submit_program`.]"
            )
          if (
              st.cfg.use_two_stage_think_nudge
              and st.cum_gen_tokens >= 20000
              and not st.sandbox.has_defined_solve
          ):
            notes.append(
                "[Commit notice: Define `def solve(grid):` in your next `run_python` cell "
                "so the harness auto-checks it on every training pair and gives you exact cell diffs.]"
            )
          b_note = " ".join(notes)
        st.transcript = prompts_and_selector.append_turn_to_chatml_transcript(
            transcript=st.transcript,
            assistant_output=st.partial_turn_text,
            tool_response=tool_resp,
            budget_note=b_note,
            use_native_xml_tools=st.cfg.use_native_xml_tools,
        )
        if (
            st.cfg.use_transcript_compaction
            and st.turn_idx >= 3
            and len(st.transcript) > 56000
        ):
          st.transcript = prompts_and_selector.compact_chatml_transcript(
              st.transcript, keep_recent_turns=2, max_old_tool_chars=1500
          )
        st.partial_turn_text = ""
        st.partial_turn_tokens = 0
        st.in_nudge_continuation = False
        st.nudge_stage = 0
        _enqueue_generation(st, is_nudge_grace=False)

      _maybe_schedule_wave2()

      active_stream_count = len({id(x) for x in active_requests.values()}) + len(
          pending_tool_futures
      )
      while pending_streams and active_stream_count < max_concurrent_requests:
        next_st = pending_streams.pop(0)
        _enqueue_generation(next_st, is_nudge_grace=False)
        active_stream_count = len({id(x) for x in active_requests.values()}) + len(
            pending_tool_futures
        )

      if engine.has_unfinished_requests():
        step_outputs = engine.step()
        for ro in step_outputs:
          if not getattr(ro, "finished", False):
            continue
          st = _pop_active_request(str(ro.request_id))
          if st is None:
            continue
          comp = ro.outputs[0]
          gen_text = comp.text or ""
          gen_toks = len(comp.token_ids)
          total_gen_tokens_all += gen_toks

          combined_text = st.partial_turn_text + gen_text
          combined_toks = st.partial_turn_tokens + gen_toks

          has_complete_tool_call = (
              "</tool_call>" in combined_text or "</function>" in combined_text
          )
          if (
              not has_complete_tool_call
              and str(comp.finish_reason) == "length"
              and (st.cum_gen_tokens + combined_toks + 512 < st.cfg.total_token_budget)
          ):
            if st.cfg.use_two_stage_think_nudge:
              if (
                  st.nudge_stage == 0
                  and "</think>" not in combined_text
                  and "<tool_call>" not in combined_text
              ):
                st.nudge_stage = 1
                st.in_nudge_continuation = True
                st.partial_turn_text = (
                    combined_text + prompts_and_selector.XML_THINK_NUDGE_STAGE1
                )
                st.partial_turn_tokens = combined_toks
                _enqueue_generation(st, is_nudge_grace=True)
                continue
              if st.nudge_stage < 2:
                st.nudge_stage = 2
                st.in_nudge_continuation = True
                if "<tool_call>" in combined_text:
                  st.partial_turn_text = combined_text
                elif "</think>" in combined_text:
                  st.partial_turn_text = (
                      combined_text
                      + "\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n"
                  )
                else:
                  force_act = (
                      prompts_and_selector.XML_COMMIT_FORCE_ACT_STAGE2
                      if (
                          not st.sandbox.use_e031_ipython_mode
                          and st.cum_gen_tokens + combined_toks >= 20000
                          and not st.sandbox.has_defined_solve
                      )
                      else prompts_and_selector.XML_THINK_FORCE_ACT_STAGE2
                  )
                  st.partial_turn_text = combined_text + force_act
                st.partial_turn_tokens = combined_toks
                _enqueue_generation(st, is_nudge_grace=True)
                continue
            elif not st.in_nudge_continuation:
              st.in_nudge_continuation = True
              st.partial_turn_text = (
                  combined_text
                  if "<tool_call>" in combined_text
                  else combined_text + fb["think_nudge"]
              )
              st.partial_turn_tokens = combined_toks
              _enqueue_generation(st, is_nudge_grace=True)
              continue

          st.partial_turn_text = combined_text
          st.partial_turn_tokens = combined_toks
          fut = tool_pool.submit(
              execute_tool_for_attempt,
              st,
              combined_text,
              combined_toks,
          )
          pending_tool_futures[fut] = st
      else:
        time.sleep(0.02)

      _maybe_schedule_wave2()
      _maybe_finalize_pairs()

      now = time.time()
      if now - last_heartbeat_time >= 60.0:
        last_heartbeat_time = now
        elapsed = now - start_time
        tok_s = total_gen_tokens_all / max(1.0, elapsed)
        active_cnt = len({id(x) for x in active_requests.values()}) + len(
            pending_tool_futures
        )
        print(
            f"[STATUS {cfg.exp_id} | {shard_tag}] done_pairs={completed_pairs_count}/{len(eval_pairs)} | "
            f"active_streams={active_cnt} | pending_streams={len(pending_streams)} | "
            f"gen_tokens={total_gen_tokens_all} ({tok_s:.1f} tok/s) | elapsed={elapsed/60:.1f}m",
            flush=True,
        )
        denom = max(1, completed_pairs_count)
        rem_pairs = max(0, len(eval_pairs) - completed_pairs_count)
        eta_hb = (
            round((elapsed / completed_pairs_count) * rem_pairs / 60.0, 1)
            if completed_pairs_count > 0
            else None
        )
        _safe_cns_write_json(
            summary_path,
            {
                "exp_id": cfg.exp_id,
                "title": cfg.title,
                "shard_index": FLAGS.shard_index,
                "num_shards": FLAGS.num_shards,
                "status": "running",
                "completed_pairs": completed_pairs_count,
                "total_pairs_in_shard": len(eval_pairs),
                "active_streams": active_cnt,
                "pending_streams": len(pending_streams),
                "pass1_count": pass1_hits,
                "pass1_pct": round(100.0 * pass1_hits / denom, 2) if completed_pairs_count else 0.0,
                "pass2_count": pass2_hits,
                "pass2_pct": round(100.0 * pass2_hits / denom, 2) if completed_pairs_count else 0.0,
                "sel_oracle_count": sel_oracle_hits,
                "sel_oracle_pct": round(100.0 * sel_oracle_hits / denom, 2) if completed_pairs_count else 0.0,
                "diag_oracle_count": diag_oracle_hits,
                "diag_oracle_pct": round(100.0 * diag_oracle_hits / denom, 2) if completed_pairs_count else 0.0,
                "verified_pairs_count": verified_pair_hits,
                "verified_pairs_pct": round(100.0 * verified_pair_hits / denom, 2) if completed_pairs_count else 0.0,
                "total_gen_tokens": total_gen_tokens_all,
                "elapsed_sec": round(elapsed, 1),
                "eta_minutes": eta_hb,
                "gen_tok_per_sec": round(tok_s, 1),
            },
            indent=2,
        )

    _maybe_finalize_pairs()
    print(
        f"[pilot_2] COMPLETED {cfg.exp_id} ({shard_tag}). "
        f"Outputs written to {out_dir}",
        flush=True,
    )
  finally:
    tool_pool.shutdown(wait=True)


if __name__ == "__main__":
  from vllm.g3_integration import app_runner  # pylint: disable=g-import-not-at-top
  app_runner.run(main)
