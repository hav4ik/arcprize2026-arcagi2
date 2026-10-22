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
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import sandbox_verifier
from google3.pyglib import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_id",
    "h1_python_tools_base",
    f"Hypothesis ID from {sorted(prompts_and_selector.PILOT2_HYPOTHESES.keys())}.",
)
flags.DEFINE_string(
    "output_dir",
    "",
    "CNS directory to write all traces and metrics.",
)
flags.DEFINE_string(
    "dataset_dir",
    prompts_and_selector.FROZEN_BASELINE_CONFIG["dataset_dir"],
    "CNS directory containing ARC-AGI-2 evaluation task JSON files.",
)
flags.DEFINE_string(
    "model_path",
    prompts_and_selector.FROZEN_BASELINE_CONFIG["model_path"],
    "CNS path to the Qwen3.8-27B-FP8 checkpoint.",
)
flags.DEFINE_integer("shard_index", 0, "0-based shard index for parallel TPU workers.")
flags.DEFINE_integer("num_shards", 1, "Total number of shards for this experiment.")
flags.DEFINE_integer(
    "max_tasks",
    0,
    "Optional cap on number of tasks for smoke testing (0 = full 120 tasks / 167 pairs).",
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

  if tool_name == "run_python":
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
    sp = vllm.SamplingParams(
        temperature=st.cfg.temperature,
        top_p=st.cfg.top_p,
        top_k=st.cfg.top_k,
        max_tokens=max(1, int(max_toks)),
        stop=list(fb["stop_strings"]),
        include_stop_str_in_output=True,
    )
    sp.output_kind = final_only_kind
    prompt_to_send = (
        st.transcript + st.partial_turn_text if is_nudge_grace else st.transcript
    )
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
          if (
              st.cfg.use_two_stage_think_nudge
              and st.sandbox.first_tp_pred is None
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
            st.partial_turn_text = prompts_and_selector.XML_FORCE_SUBMIT_FINAL
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
        st.transcript = prompts_and_selector.append_turn_to_chatml_transcript(
            transcript=st.transcript,
            assistant_output=st.partial_turn_text,
            tool_response=tool_resp,
            budget_note=" ".join(notes),
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

          if (
              "<tool_call>" not in combined_text
              and str(comp.finish_reason) == "length"
              and (st.cum_gen_tokens + combined_toks + 512 < st.cfg.total_token_budget)
          ):
            if st.cfg.use_two_stage_think_nudge:
              if st.nudge_stage == 0 and "</think>" not in combined_text:
                st.nudge_stage = 1
                st.in_nudge_continuation = True
                st.partial_turn_text = (
                    combined_text + prompts_and_selector.XML_THINK_NUDGE_STAGE1
                )
                st.partial_turn_tokens = combined_toks
                _enqueue_generation(st, is_nudge_grace=True)
                continue
              if st.nudge_stage == 1:
                st.nudge_stage = 2
                st.in_nudge_continuation = True
                force_act = (
                    prompts_and_selector.XML_COMMIT_FORCE_ACT_STAGE2
                    if (
                        st.cum_gen_tokens + combined_toks >= 20000
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
              st.partial_turn_text = combined_text + fb["think_nudge"]
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
        if completed_pairs_count == 0:
          _safe_cns_write_json(
              summary_path,
              {
                  "exp_id": cfg.exp_id,
                  "title": cfg.title,
                  "shard_index": FLAGS.shard_index,
                  "num_shards": FLAGS.num_shards,
                  "status": "running",
                  "completed_pairs": 0,
                  "total_pairs_in_shard": len(eval_pairs),
                  "pass1_count": 0,
                  "pass1_pct": 0.0,
                  "pass2_count": 0,
                  "pass2_pct": 0.0,
                  "sel_oracle_count": 0,
                  "sel_oracle_pct": 0.0,
                  "diag_oracle_count": 0,
                  "diag_oracle_pct": 0.0,
                  "verified_pairs_count": 0,
                  "verified_pairs_pct": 0.0,
                  "total_gen_tokens": total_gen_tokens_all,
                  "elapsed_sec": round(elapsed, 1),
                  "eta_minutes": None,
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
  app_runner.run(main)
