r"""ARC-AGI-2 Round-2 runner: six controlled inference-only configurations.

Design (see round2_plan.md):
  * Continuous batching: attempts are independent state machines driven by
    `LLMEngine.add_request()` / `step()` (the primitives behind `LLM.generate`),
    so no attempt ever waits for another attempt's turn to finish. Tool calls
    run in a thread pool while the engine keeps decoding other attempts.
  * One model turn = one assistant message. Think-cap continuations belong to
    the same turn; forced final submissions are turns. `gen_calls` and
    `tool_executions` are logged separately.
  * 48k cumulative generated-token cap per attempt over every generate call.
  * Candidates are preserved (first verified submission frozen as primary),
    re-evaluated in isolation at attempt end, split into selection / diagnostic
    pools and ranked with the frozen selector. The evaluator sees test outputs
    only after a pair's attempts are complete.

Configurations: exp0_base16, exp1_turns40, exp2_autochk40, exp3_mixed,
exp4_audit40, exp5_views16 (see `attempt_specs`).
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from concurrent import futures
import dataclasses
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any

from absl import flags
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import prompts_and_voting as pv
from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import round2_candidates as r2
from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import sandbox
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "CNS path to model weights.",
)
flags.DEFINE_string(
    "cns_data_path",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data",
    "CNS path to ARC-AGI-2 dataset (expects an `evaluation/` subdirectory).",
)
flags.DEFINE_string("output_dir", "", "CNS output directory for this config.")
flags.DEFINE_enum(
    "r2_config",
    "exp0_base16",
    [
        "exp0_base16",
        "exp1_turns40",
        "exp2_autochk40",
        "exp3_mixed",
        "exp4_audit40",
        "exp5_views16",
    ],
    "Round-2 configuration.",
)
flags.DEFINE_integer("num_attempts", 4, "Attempts per test pair.")
flags.DEFINE_integer("total_token_budget", 48000, "Generated-token cap per attempt.")
flags.DEFINE_integer("soft_token_target", 24000, "Soft target shown in budget notes.")
flags.DEFINE_integer("turn_think_cap", 8000, "Max tokens per generate call.")
flags.DEFINE_integer("audit_extra_turns", 4, "EXP-4: extra turns after first verified submit.")
flags.DEFINE_integer("attempts_in_flight", 64, "Attempts concurrently in flight per worker.")
flags.DEFINE_integer("num_shards", 1, "Total shards.")
flags.DEFINE_integer("shard_idx", 0, "This shard.")
flags.DEFINE_integer("limit_tasks", 0, "Smoke test: only first N tasks (sorted).")
flags.DEFINE_string("task_ids", "", "Smoke test: comma-separated task ids.")
flags.DEFINE_float("temperature", 0.6, "Sampling temperature.")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling top_p.")
flags.DEFINE_integer("top_k", 20, "Top-k sampling.")
flags.DEFINE_integer("tensor_parallel_size", 4, "TP size.")
flags.DEFINE_integer("max_model_len", 131072, "vLLM max_model_len.")
flags.DEFINE_integer("max_num_batched_tokens", 8192, "vLLM max_num_batched_tokens.")
flags.DEFINE_integer("max_num_seqs", 64, "vLLM max_num_seqs.")
flags.DEFINE_float("gpu_memory_utilization", 0.90, "HBM fraction.")
flags.DEFINE_float("exec_timeout_s", 5.0, "Sandbox timeout per tool execution.")
flags.DEFINE_integer("exec_workers", 24, "Thread pool size for tool execution.")
flags.DEFINE_integer("finalize_workers", 6, "Thread pool size for isolated re-evaluation.")
flags.DEFINE_boolean("save_transcripts", True, "Store full transcripts in attempt rows.")
flags.DEFINE_integer("progress_every_s", 60, "Progress log interval.")
flags.DEFINE_boolean("use_request_seed", False, "Pass per-request seeds to SamplingParams (unsupported by the JAX/TPU sampler).")

AUDIT_INSTRUCTION = (
    "AUDIT (targeted assumption check): Your program fits every demonstration,"
    " but several rules may fit those examples. Inspect the visible test input"
    " for properties that differ from the demonstrations. Identify any"
    " unsupported assumption in the program concerning symmetry, orientation,"
    " marker location, absolute coordinates, periodic phase, object counts,"
    " object dimensions, colors or output size.\n\nIf a plausible competing rule"
    " also explains every demonstration, implement and test it. Prefer a"
    " coherent alternative to a task-specific patch. If no justified"
    " alternative exists, retain the original. Do not modify the program merely"
    " to satisfy this request.\n\nExamples of the kinds of assumptions to"
    " examine: sorting objects does not necessarily imply making them"
    " symmetric; a repeating pattern may be anchored to a marker rather than"
    " row zero; a rule observed with two objects may need to apply to three;"
    " reflection offsets may depend on odd/even dimensions; output dimensions"
    " may follow an object count rather than a fixed constant.\n\n[harness: your"
    " original program is saved. You have up to 4 more turns. If you find a"
    " justified alternative, call `submit_program` with it; otherwise reply"
    " briefly without a tool call to retain the original.]"
)

AUTO_CHECK_NOTE = (
    "Note: whenever a `run_python` cell defines or changes `def solve(grid):`,"
    " the harness automatically evaluates it on all demonstrations and appends"
    " concise feedback (passed count, output dimensions, differing cells)."
)

ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n"


@dataclasses.dataclass(frozen=True)
class AttemptSpec:
  max_turns: int
  auto_check: bool
  audit: bool
  view: str


def attempt_specs(config: str) -> list[AttemptSpec]:
  """Per-attempt specifications for each configuration (frozen)."""
  plain16 = AttemptSpec(16, False, False, "identity")
  plain40 = AttemptSpec(40, False, False, "identity")
  auto40 = AttemptSpec(40, True, False, "identity")
  audit40 = AttemptSpec(40, False, True, "identity")
  if config == "exp0_base16":
    return [plain16] * 4
  if config == "exp1_turns40":
    return [plain40] * 4
  if config == "exp2_autochk40":
    return [auto40] * 4
  if config == "exp3_mixed":
    return [plain16, plain16, auto40, auto40]
  if config == "exp4_audit40":
    return [audit40] * 4
  if config == "exp5_views16":
    return [
        plain16,
        plain16,
        AttemptSpec(16, False, False, "rot90"),
        AttemptSpec(16, False, False, "fliplr"),
    ]
  raise ValueError(config)


class AttemptState:
  """One independent attempt: transcript, sandbox, counters, lifecycle flags."""

  def __init__(
      self,
      task_id: str,
      pair_idx: int,
      attempt_idx: int,
      spec: AttemptSpec,
      solver_task: dict[str, Any],
      test_input: list[list[int]],
  ):
    self.task_id = task_id
    self.pair_idx = pair_idx
    self.attempt_idx = attempt_idx
    self.spec = spec
    self.view = spec.view
    self.seed = r2.stable_seed(task_id, pair_idx, attempt_idx)
    self.task_view, self.test_input_view = r2.transform_task_view(
        solver_task, test_input, spec.view
    )
    self.sandbox = sandbox.ArcPythonSandbox(
        task=self.task_view,
        test_input=self.test_input_view,
        freeze_first_verified=True,
        auto_check=spec.auto_check,
        structured_feedback=False,
        auto_check_style="concise",
        snapshot_defs=True,
        compute_invariance=False,
    )
    transcript = pv.build_initial_transcript(
        task=self.task_view,
        test_input=self.test_input_view,
        use_tools=True,
        soft_target=FLAGS.soft_token_target,
        hard_budget=FLAGS.total_token_budget,
        coder_next_format=False,
    )
    if spec.auto_check:
      tail = f"<|im_end|>\n{ASSISTANT_PREFIX}"
      assert transcript.endswith(tail)
      transcript = transcript[: -len(tail)] + f"\n\n{AUTO_CHECK_NOTE}" + tail
    self.transcript = transcript
    self.ctx_tokens_est = 0
    self.turns = 0
    self.gen_calls = 0
    self.gen_tokens = 0
    self.input_tokens_total = 0
    self.input_tokens_cached = 0
    self.tool_executions = 0
    self.think_nudges = 0
    self.eos_continues = 0
    self.nudge_pending = False
    self.force_submit_pending = False
    self.force_reason = ""
    self.audit_started = False
    self.audit_turns_used = 0
    self.audit_verified_at_start = 0
    self.done = False
    self.termination = ""
    self.infra_error: str | None = None
    self.t0 = time.time()
    self.wall_s = 0.0
    self.turn_log: list[dict[str, Any]] = []
    self.current_rid: str | None = None

  def elapsed(self) -> float:
    return time.time() - self.t0

  def set_sandbox_ctx(self) -> None:
    self.sandbox.ctx = {
        "turn": self.turns,
        "gen_tokens": self.gen_tokens,
        "t": self.elapsed(),
    }


class Runner:
  """Continuous-batching driver over a vLLM LLMEngine."""

  def __init__(self, llm: Any, vllm_mod: Any):
    self.llm = llm
    self.vllm = vllm_mod
    self.engine = llm.llm_engine
    self.tokenizer = llm.get_tokenizer()
    from vllm.sampling_params import RequestOutputKind  # pylint: disable=g-import-not-at-top

    self._final_only = RequestOutputKind.FINAL_ONLY
    self.inflight: dict[str, AttemptState] = {}
    self.rid_alias: dict[str, str] = {}
    self.exec_pool = futures.ThreadPoolExecutor(max_workers=FLAGS.exec_workers)
    self.exec_futures: dict[futures.Future, tuple[AttemptState, bool]] = {}
    self.rid_counter = 0
    self.lock = threading.Lock()
    self.finished: list[AttemptState] = []
    self.total_gen_tokens = 0
    self.n_steps = 0
    self.step_time_total = 0.0
    self.step_time_max = 0.0
    self.t_start = time.time()

  # ---------------------------------------------------------------- helpers
  def _count(self, text: str) -> int:
    if not text:
      return 0
    try:
      return len(self.tokenizer.encode(text))
    except Exception:  # pylint: disable=broad-exception-caught
      return max(1, len(text) // 3)

  def _params(self, att: AttemptState, max_tokens: int, temperature: float, stop: list[str] | None = None) -> Any:
    kwargs: dict[str, Any] = dict(
        temperature=temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
        max_tokens=max(1, int(max_tokens)),
        stop=stop,
    )
    if FLAGS.use_request_seed:
      # NOTE: the JAX/TPU sampler raises "JAX does not support per-request
      # seed" (observed in the smoke test), so this is off by default.
      kwargs["seed"] = att.seed
    sp = self.vllm.SamplingParams(**kwargs)
    sp.output_kind = self._final_only
    return sp

  def _add_request(self, att: AttemptState, params: Any, kind: str) -> None:
    self.rid_counter += 1
    att.turn_log.append({"turn": att.turns, "kind": kind, "max_tokens": params.max_tokens, "t": round(att.elapsed(), 1)})
    try:
      # `enqueue` is the same path `LLM.generate` uses (prompt rendering +
      # FINAL_ONLY outputs) minus the blocking `_run_engine` loop. It returns
      # the *internal* id "<external>-<8 random chars>" (InputProcessor.
      # assign_request_id) whereas RequestOutput.request_id is the external
      # id, so register both keys.
      rids = self.llm.enqueue([att.transcript], [params], use_tqdm=False)
      internal = str(rids[0])
      external = internal.rsplit("-", 1)[0] if "-" in internal else internal
      att.current_rid = external
      self.inflight[external] = att
      if internal != external:
        self.inflight[internal] = att
        self.rid_alias[internal] = external
        self.rid_alias[external] = internal
    except Exception as exc:  # pylint: disable=broad-exception-caught
      att.infra_error = f"enqueue: {type(exc).__name__}: {exc}"[:500]
      self.finish(att, "infra_failure")

  def _pop_inflight(self, rid: str) -> AttemptState | None:
    att = self.inflight.pop(rid, None)
    other = self.rid_alias.pop(rid, None)
    if other is not None:
      self.inflight.pop(other, None)
      self.rid_alias.pop(other, None)
    return att



  def finish(self, att: AttemptState, reason: str) -> None:
    if att.done:
      return
    att.done = True
    att.termination = reason
    att.wall_s = att.elapsed()
    self.finished.append(att)

  # ------------------------------------------------------------- scheduling
  def schedule_next(self, att: AttemptState) -> None:
    """Decides and submits the next generate call for `att` (or finishes it)."""
    if att.done:
      return
    sb = att.sandbox
    budget = FLAGS.total_token_budget
    headroom = FLAGS.max_model_len - att.ctx_tokens_est - 512
    if headroom <= 256:
      self.finish(att, "ctx_cap")
      return
    rem = min(budget - att.gen_tokens, headroom)
    if rem <= 256:
      self.finish(att, "token_cap")
      return

    if att.nudge_pending:
      # Continuation of the current turn after a think-cap nudge.
      self._add_request(att, self._params(att, min(2048, rem), FLAGS.temperature, stop=["</parameter>"]), "nudge_continue")
      return

    if att.audit_started:
      if sb.num_verified_submissions > att.audit_verified_at_start:
        self.finish(att, "audit_complete")
        return
      if att.audit_turns_used >= FLAGS.audit_extra_turns:
        self.finish(att, "audit_complete")
        return
      if att.turns >= att.spec.max_turns:
        self.finish(att, "turn_cap")
        return
      att.audit_turns_used += 1
      att.turns += 1
      self._add_request(att, self._params(att, min(FLAGS.turn_think_cap, rem), FLAGS.temperature), "audit_turn")
      return

    if sb.train_passed:
      self.finish(att, "solved")
      return

    turn_cap_hit = att.turns >= att.spec.max_turns - 1
    token_low = rem <= 1536
    if turn_cap_hit or token_low:
      if headroom >= 640 and rem >= 320:
        att.force_submit_pending = True
        att.force_reason = "turn_cap" if turn_cap_hit else "token_cap"
        att.transcript += pv.FORCE_SUBMIT_PROMPT
        att.ctx_tokens_est += 24
        att.turns += 1
        self._add_request(
            att,
            self._params(att, min(1536, rem - 64), 0.2, stop=["</parameter>", "</tool_call>"]),
            "forced_submit",
        )
      else:
        self.finish(att, "turn_cap" if turn_cap_hit else "token_cap")
      return

    att.turns += 1
    self._add_request(att, self._params(att, min(FLAGS.turn_think_cap, rem), FLAGS.temperature), "turn")

  # ------------------------------------------------------------- callbacks
  def on_generation(self, att: AttemptState, out: Any) -> None:
    gen = out.outputs[0]
    text = gen.text or ""
    n_tok = len(gen.token_ids)
    finish_reason = str(gen.finish_reason)
    n_prompt = len(out.prompt_token_ids or [])
    n_cached = int(getattr(out, "num_cached_tokens", None) or 0)
    att.gen_calls += 1
    att.gen_tokens += n_tok
    att.input_tokens_total += n_prompt
    att.input_tokens_cached += n_cached
    att.ctx_tokens_est = n_prompt + n_tok
    self.total_gen_tokens += n_tok
    if att.turn_log:
      att.turn_log[-1].update({"gen_tokens": n_tok, "prompt_tokens": n_prompt, "cached_tokens": n_cached, "finish": finish_reason})
    att.set_sandbox_ctx()
    sb = att.sandbox

    if att.force_submit_pending:
      att.force_submit_pending = False
      code, _ = pv.strip_excessive_comments(text.split("</parameter>")[0].strip())
      closing = f"{code}\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
      att.transcript += closing
      att.ctx_tokens_est += self._count(closing) - n_tok
      if code:
        self._submit_exec(att, [("submit_program", code)], final=True)
      else:
        self.finish(att, att.force_reason)
      return

    if att.nudge_pending:
      att.nudge_pending = False
      code, _ = pv.strip_excessive_comments(text.split("</parameter>")[0].strip())
      closing = f"{code}\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
      att.transcript += closing
      att.ctx_tokens_est += self._count(closing) - n_tok
      if code:
        self._submit_exec(att, [("run_python", code)], final=False)
      else:
        self.finish(att, "model_stop")
      return

    att.transcript += text
    calls = pv.parse_tool_calls(text)
    if calls:
      if not att.transcript.endswith("<|im_end|>\n"):
        att.transcript += "<|im_end|>\n"
        att.ctx_tokens_est += 2
      self._submit_exec(att, calls, final=False)
      return

    if finish_reason == "length" and "</think>" not in text:
      if att.gen_tokens + 1024 < FLAGS.total_token_budget:
        att.think_nudges += 1
        att.transcript += pv.THINK_NUDGE
        att.ctx_tokens_est += 60
        att.nudge_pending = True
        self.schedule_next(att)
      else:
        self.finish(att, "token_cap")
      return

    if att.audit_started and sb.train_passed:
      self.finish(att, "audit_complete")
      return

    parsed = pv.parse_last_grid(text)
    if parsed is not None:
      sb._record_candidate(  # pylint: disable=protected-access
          grid=parsed,
          source="visible_json",
          train_pass=None,
          train_frac=None,
          cell_frac=0.0,
          suspect=False,
          is_primary=(sb.primary_grid is None),
          code="",
          detail=[],
      )
      self.finish(att, "model_stop")
      return

    if att.eos_continues < 3 and att.gen_tokens + 1024 < FLAGS.total_token_budget:
      att.eos_continues += 1
      if not att.transcript.endswith("<|im_end|>\n"):
        att.transcript += "<|im_end|>\n"
      bnote = pv.format_budget_note(att.gen_tokens, FLAGS.soft_token_target, FLAGS.total_token_budget)
      appended = (
          "<|im_start|>user\nPlease call `run_python` to test your rule or"
          " `submit_program` with `def solve(grid):`."
          f"{bnote}<|im_end|>\n{ASSISTANT_PREFIX}"
      )
      att.transcript += appended
      att.ctx_tokens_est += self._count(appended)
      self.schedule_next(att)
      return

    self.finish(att, "model_stop")

  def _submit_exec(self, att: AttemptState, calls: list[tuple[str, str]], final: bool) -> None:
    att.set_sandbox_ctx()

    def _run() -> list[str]:
      sb = att.sandbox
      outs: list[str] = []
      was_passed = sb.train_passed
      for fn_name, code in calls:
        att.tool_executions += 1
        try:
          outs.append(sb.execute_turn(fn_name, code, timeout_s=FLAGS.exec_timeout_s))
        except Exception as exc:  # pylint: disable=broad-exception-caught
          outs.append(f"ERROR: harness execution failure: {type(exc).__name__}: {exc}"[:800])
        if not was_passed and sb.train_passed:
          break
      return outs

    fut = self.exec_pool.submit(_run)
    self.exec_futures[fut] = (att, final)

  def on_exec_done(self, att: AttemptState, outs: list[str], final: bool) -> None:
    sb = att.sandbox
    if final:
      self.finish(att, att.force_reason or "turn_cap")
      return
    if sb.train_passed:
      if att.spec.audit and not att.audit_started:
        att.audit_started = True
        att.audit_verified_at_start = sb.num_verified_submissions
        outs = list(outs) + [AUDIT_INSTRUCTION]
      elif att.audit_started and sb.num_verified_submissions > att.audit_verified_at_start:
        self.finish(att, "audit_complete")
        return
      elif not att.audit_started:
        self.finish(att, "solved")
        return
    combined = "\n---\n".join(outs)
    bnote = pv.format_budget_note(att.gen_tokens, FLAGS.soft_token_target, FLAGS.total_token_budget)
    appended = (
        f"<|im_start|>user\n<tool_response>\n{combined}{bnote}\n"
        f"</tool_response><|im_end|>\n{ASSISTANT_PREFIX}"
    )
    att.transcript += appended
    att.ctx_tokens_est += self._count(appended)
    self.schedule_next(att)

  # ------------------------------------------------------------------ loop
  def poll_exec(self) -> None:
    if not self.exec_futures:
      return
    done = [f for f in list(self.exec_futures) if f.done()]
    for f in done:
      att, final = self.exec_futures.pop(f)
      try:
        outs = f.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        att.infra_error = f"exec: {type(exc).__name__}: {exc}"[:500]
        self.finish(att, "exec_failure")
        continue
      self.on_exec_done(att, outs, final)

  def step_engine(self) -> bool:
    """One engine step; returns True if any request finished."""
    if not self.engine.has_unfinished_requests():
      return False
    t0 = time.perf_counter()
    try:
      outs = self.engine.step()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Infrastructure failure: fail every in-flight attempt rather than hang.
      print(f"[ERROR] engine.step raised: {exc!r}", flush=True)
      for rid, att in list(self.inflight.items()):
        att.infra_error = f"engine.step: {type(exc).__name__}: {exc}"[:500]
        self.finish(att, "infra_failure")
      self.inflight.clear()
      self.rid_alias.clear()
      raise
    dt = time.perf_counter() - t0
    self.n_steps += 1
    self.step_time_total += dt
    self.step_time_max = max(self.step_time_max, dt)
    any_done = False
    for o in outs:
      if not getattr(o, "finished", False):
        continue
      att = self._pop_inflight(str(o.request_id))
      if att is None:
        print(f"[WARN] finished output for unknown request id {o.request_id!r}", flush=True)
        continue
      any_done = True
      try:
        self.on_generation(att, o)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        att.infra_error = f"on_generation: {type(exc).__name__}: {exc}"[:500]
        self.finish(att, "infra_failure")
    return any_done

  def engine_snapshot(self) -> str:
    """Best-effort view of the in-process scheduler for progress logs."""
    try:
      sched = self.engine.engine_core.engine_core.scheduler
      running = list(getattr(sched, "running", []))
      waiting = getattr(sched, "waiting", [])
      max_out = max((int(getattr(r, "num_output_tokens", 0)) for r in running), default=0)
      max_comp = max((int(getattr(r, "num_computed_tokens", 0)) for r in running), default=0)
      usage = getattr(getattr(sched, "kv_cache_manager", None), "usage", None)
      usage_s = f"{usage:.3f}" if isinstance(usage, float) else str(usage)
      return (
          f"sched running={len(running)} waiting={len(waiting)} kv_usage={usage_s}"
          f" max_out_tok={max_out} max_computed={max_comp}"
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      return f"sched n/a ({type(exc).__name__})"



def attempt_row(att: AttemptState, config: str) -> dict[str, Any]:
  """Finalizes candidates (isolated re-evaluation) and builds the attempt row."""
  sb = att.sandbox
  cands = r2.finalize_attempt_candidates(
      raw_cands=sb.candidates,
      cell_history=sb.cell_history,
      task_view=att.task_view,
      test_input_view=att.test_input_view,
      view=att.view,
      attempt_idx=att.attempt_idx,
      timeout_s=FLAGS.exec_timeout_s,
  )
  primary = next((c for c in cands if c.get("is_primary")), None)
  row = {
      "task_id": att.task_id,
      "pair_idx": att.pair_idx,
      "config": config,
      "config_version": r2.CONFIG_VERSION,
      "attempt_idx": att.attempt_idx,
      "seed": att.seed,
      "seed_applied": bool(FLAGS.use_request_seed),
      "view": att.view,
      "max_turns": att.spec.max_turns,
      "auto_check": att.spec.auto_check,
      "audit": att.spec.audit,
      "gen_tokens": att.gen_tokens,
      "input_tokens_total": att.input_tokens_total,
      "input_tokens_cached": att.input_tokens_cached,
      "input_tokens_computed": att.input_tokens_total - att.input_tokens_cached,
      "final_context_tokens": att.ctx_tokens_est,
      "turns": att.turns,
      "gen_calls": att.gen_calls,
      "tool_executions": att.tool_executions,
      "think_nudges": att.think_nudges,
      "eos_continues": att.eos_continues,
      "wall_s": round(att.wall_s, 1),
      "termination": att.termination,
      "infra_error": att.infra_error,
      "audit_started": att.audit_started,
      "audit_turns_used": att.audit_turns_used,
      "train_passed_live": bool(sb.train_passed),
      "num_verified_submissions": sb.num_verified_submissions,
      "frozen_overrides": sb.frozen_overrides,
      "num_auto_checks": sb.num_auto_checks,
      "num_verifications": sb.num_verifications,
      "ever_defined_solve": bool(sb.ever_defined_solve),
      "n_candidates": len(cands),
      "n_selection_candidates": sum(1 for c in cands if c.get("pool") == "selection" and c.get("grid") is not None),
      "primary_verified": bool(primary and primary.get("verified")),
      "cells": [
          {k: v for k, v in c.items() if k != "code"} | {"code_chars": len(c.get("code", ""))}
          for c in sb.cell_history
      ],
      "turn_log": att.turn_log,
      "candidates": cands,
      "transcript": att.transcript if FLAGS.save_transcripts else None,
  }
  sb.ns.clear()
  return row


def load_completed_pairs(output_dir: str, shard_suffix: str) -> set[tuple[str, int]]:
  done: set[tuple[str, int]] = set()
  path = os.path.join(output_dir, f"pairs{shard_suffix}.jsonl")
  if not gfile.Exists(path):
    return done
  try:
    with gfile.Open(path, "r") as f:
      for line in f:
        line = line.strip()
        if line:
          r = json.loads(line)
          done.add((r["task_id"], int(r["pair_idx"])))
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"[WARN] could not read {path}: {exc}", flush=True)
  return done


def shard_summary(pair_rows: list[dict[str, Any]], attempt_rows_meta: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
  n = len(pair_rows)
  if n == 0:
    return {"pairs_completed": 0, "wall_s": round(wall_s, 1)}
  by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  for r in pair_rows:
    by_task[r["task_id"]].append(r)
  term = collections.Counter(a["termination"] for a in attempt_rows_meta)
  return {
      "config": FLAGS.r2_config,
      "config_version": r2.CONFIG_VERSION,
      "shard_idx": FLAGS.shard_idx,
      "num_shards": FLAGS.num_shards,
      "pairs_completed": n,
      "tasks_completed": len(by_task),
      "pair_top1_pct": round(100.0 * sum(r["top1_exact"] for r in pair_rows) / n, 2),
      "pair_top2_pct": round(100.0 * sum(r["top2_exact"] for r in pair_rows) / n, 2),
      "selection_oracle_pct": round(100.0 * sum(r["selection_oracle"] for r in pair_rows) / n, 2),
      "diagnostic_oracle_pct": round(100.0 * sum(r["diagnostic_oracle"] for r in pair_rows) / n, 2),
      "task_avg_top2_pct": round(
          100.0 * sum(sum(r["top2_exact"] for r in v) / len(v) for v in by_task.values()) / len(by_task), 2
      ),
      "attempts": len(attempt_rows_meta),
      "mean_gen_tokens": round(sum(a["gen_tokens"] for a in attempt_rows_meta) / max(1, len(attempt_rows_meta)), 1),
      "mean_turns": round(sum(a["turns"] for a in attempt_rows_meta) / max(1, len(attempt_rows_meta)), 2),
      "total_gen_tokens": sum(a["gen_tokens"] for a in attempt_rows_meta),
      "total_input_tokens": sum(a["input_tokens_total"] for a in attempt_rows_meta),
      "termination_counts": dict(term),
      "wall_s": round(wall_s, 1),
      "gen_tokens_per_s": round(sum(a["gen_tokens"] for a in attempt_rows_meta) / max(1.0, wall_s), 1),
      "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
  }


def main(argv: Sequence[str]) -> None:
  del argv
  assert FLAGS.output_dir, "--output_dir is required"
  t_main0 = time.time()
  # 1. Data -> local disk (before JAX init).
  local_data = f"/tmp/arc_agi_2_r2_shard_{FLAGS.shard_idx}"
  if os.path.exists(local_data):
    shutil.rmtree(local_data)
  os.makedirs(local_data, exist_ok=True)
  r = subprocess.run(
      ["fileutil", "cp", "-R", "-parallelism=32", f"{FLAGS.cns_data_path}/evaluation", f"{local_data}/evaluation"],
      capture_output=True, text=True, check=False,
  )
  print(f"Data copy rc={r.returncode} ({time.time() - t_main0:.1f}s)", flush=True)
  evalset = {
      f[:-5]: json.load(open(f"{local_data}/evaluation/{f}"))
      for f in sorted(os.listdir(f"{local_data}/evaluation"))
      if f.endswith(".json")
  }
  # Manifest derived from the dataset (never assumed).
  manifest = {
      "n_tasks": len(evalset),
      "n_pairs": sum(len(t["test"]) for t in evalset.values()),
      "pairs": [[tid, i] for tid, t in sorted(evalset.items()) for i in range(len(t["test"]))],
      "config_version": r2.CONFIG_VERSION,
  }
  if FLAGS.task_ids:
    keep = {t.strip() for t in FLAGS.task_ids.split(",") if t.strip()}
    unknown = sorted(keep - set(evalset))
    if unknown:
      raise ValueError(f"Unknown task ids: {unknown}")
    evalset = {k: v for k, v in evalset.items() if k in keep}
  if FLAGS.limit_tasks > 0:
    evalset = dict(sorted(evalset.items())[: FLAGS.limit_tasks])

  all_pairs = [(tid, pidx, task, tpair) for tid, task in sorted(evalset.items()) for pidx, tpair in enumerate(task["test"])]
  shard_pairs = [it for i, it in enumerate(all_pairs) if i % FLAGS.num_shards == FLAGS.shard_idx]
  gfile.MakeDirs(FLAGS.output_dir)
  shard_suffix = f"_shard_{FLAGS.shard_idx:03d}of{FLAGS.num_shards:03d}"
  attempts_path = os.path.join(FLAGS.output_dir, f"attempts{shard_suffix}.jsonl")
  pairs_path = os.path.join(FLAGS.output_dir, f"pairs{shard_suffix}.jsonl")
  summary_path = os.path.join(FLAGS.output_dir, f"summary{shard_suffix}.json")
  if FLAGS.shard_idx == 0:
    try:
      with gfile.Open(os.path.join(FLAGS.output_dir, "manifest.json"), "w") as mf:
        json.dump(manifest, mf)
      with gfile.Open(os.path.join(FLAGS.output_dir, "config.json"), "w") as cf:
        json.dump(
            {
                "config": FLAGS.r2_config,
                "config_version": r2.CONFIG_VERSION,
                "attempt_specs": [dataclasses.asdict(s) for s in attempt_specs(FLAGS.r2_config)],
                "total_token_budget": FLAGS.total_token_budget,
                "turn_think_cap": FLAGS.turn_think_cap,
                "temperature": FLAGS.temperature,
                "top_p": FLAGS.top_p,
                "top_k": FLAGS.top_k,
                "max_model_len": FLAGS.max_model_len,
                "model_path": FLAGS.model_path,
                "ranking_tuple": list(r2.RANKING_TUPLE),
                "num_shards": FLAGS.num_shards,
                "attempts_in_flight": FLAGS.attempts_in_flight,
            },
            cf,
            indent=2,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[WARN] manifest write failed: {exc}", flush=True)

  completed = load_completed_pairs(FLAGS.output_dir, shard_suffix)
  todo = [it for it in shard_pairs if (it[0], it[1]) not in completed]
  specs = attempt_specs(FLAGS.r2_config)
  print(
      f"[{FLAGS.r2_config}] manifest tasks={manifest['n_tasks']} pairs={manifest['n_pairs']} |"
      f" shard {FLAGS.shard_idx}/{FLAGS.num_shards}: {len(shard_pairs)} pairs, {len(todo)} todo |"
      f" specs={[dataclasses.asdict(s) for s in specs]}",
      flush=True,
  )
  if not todo:
    print("Nothing to do for this shard.", flush=True)
    return

  # 2. Engine.
  import jax  # pylint: disable=g-import-not-at-top
  import vllm  # pylint: disable=g-import-not-at-top

  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  t_init0 = time.time()
  llm = vllm.LLM(
      model=FLAGS.model_path,
      tensor_parallel_size=FLAGS.tensor_parallel_size,
      max_model_len=FLAGS.max_model_len,
      max_num_batched_tokens=FLAGS.max_num_batched_tokens,
      max_num_seqs=FLAGS.max_num_seqs,
      gpu_memory_utilization=FLAGS.gpu_memory_utilization,
      enable_prefix_caching=True,
      load_format="tpu_streaming_loader",
  )
  print(f"vLLM initialized in {time.time() - t_init0:.1f}s on {num_tpus}x {device_kind}", flush=True)
  runner = Runner(llm, vllm)

  # 3. Work queue.
  pending: collections.deque[tuple[str, int, int]] = collections.deque()
  pair_info: dict[tuple[str, int], dict[str, Any]] = {}
  for tid, pidx, task, tpair in todo:
    solver_task = r2.strip_test_outputs(task)
    pair_info[(tid, pidx)] = {
        "solver_task": solver_task,
        "test_input": tpair["input"],
        "target": tpair["output"],  # evaluator-only
        "attempts": {},
        "rows": {},
    }
    for a in range(FLAGS.num_attempts):
      pending.append((tid, pidx, a))

  fin_pool = futures.ThreadPoolExecutor(max_workers=FLAGS.finalize_workers)
  fin_futures: dict[futures.Future, AttemptState] = {}
  active: set[AttemptState] = set()
  pair_rows_written: list[dict[str, Any]] = []
  attempt_meta_written: list[dict[str, Any]] = []
  t_eval0 = time.time()
  last_progress = time.time()
  n_pairs_total = len(todo)

  def _write_pair(key: tuple[str, int]) -> None:
    info = pair_info[key]
    rows = [info["rows"][a] for a in sorted(info["rows"])]
    score = r2.score_pair(rows, fallback_grid=info["test_input"], target=info["target"])
    pair_row = {
        "task_id": key[0],
        "pair_idx": key[1],
        "config": FLAGS.r2_config,
        "config_version": r2.CONFIG_VERSION,
        "n_attempts": len(rows),
        **score,
        "attempts": [
            {
                k: a[k]
                for k in (
                    "attempt_idx", "view", "max_turns", "auto_check", "audit", "gen_tokens",
                    "input_tokens_total", "input_tokens_computed", "turns", "gen_calls", "tool_executions", "wall_s",
                    "termination", "n_candidates", "n_selection_candidates", "primary_verified",
                    "audit_started", "num_verified_submissions",
                )
            }
            | {"primary_exact": next((c.get("exact") for c in a["candidates"] if c.get("is_primary")), None)}
            for a in rows
        ],
        "total_gen_tokens": sum(a["gen_tokens"] for a in rows),
        "total_wall_s": round(sum(a["wall_s"] for a in rows), 1),
    }
    for _ in range(3):
      try:
        with gfile.Open(attempts_path, "a") as af:
          for a in rows:
            af.write(json.dumps(a) + "\n")
        with gfile.Open(pairs_path, "a") as pf:
          pf.write(json.dumps(pair_row) + "\n")
        break
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[WARN] CNS write failed, retrying: {exc}", flush=True)
        time.sleep(5)
    pair_rows_written.append(pair_row)
    attempt_meta_written.extend(pair_row["attempts"])
    for a in rows:
      a["candidates"] = None
      a["transcript"] = None
    info["rows"] = {}
    info["attempts"] = {}
    try:
      with gfile.Open(summary_path, "w") as sf:
        json.dump(shard_summary(pair_rows_written, attempt_meta_written, time.time() - t_eval0), sf, indent=2)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[WARN] summary write failed: {exc}", flush=True)

  while pending or active or fin_futures:
    # a) start new attempts up to the in-flight cap
    while pending and len(active) < FLAGS.attempts_in_flight:
      tid, pidx, a_idx = pending.popleft()
      info = pair_info[(tid, pidx)]
      att = AttemptState(tid, pidx, a_idx, specs[a_idx], info["solver_task"], info["test_input"])
      att.ctx_tokens_est = runner._count(att.transcript)  # pylint: disable=protected-access
      info["attempts"][a_idx] = att
      active.add(att)
      runner.schedule_next(att)
    # b) tool executions that completed
    runner.poll_exec()
    # c) engine step
    progressed = runner.step_engine()
    if not progressed and not runner.engine.has_unfinished_requests():
      time.sleep(0.02)
    # d) finished attempts -> isolated finalization in background threads
    if runner.finished:
      batch, runner.finished = runner.finished, []
      for att in batch:
        active.discard(att)
        fin_futures[fin_pool.submit(attempt_row, att, FLAGS.r2_config)] = att
    # e) finalized attempts -> pair completion -> write
    for f in [f for f in list(fin_futures) if f.done()]:
      att = fin_futures.pop(f)
      key = (att.task_id, att.pair_idx)
      try:
        row = f.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[WARN] finalize failed for {key} att {att.attempt_idx}: {exc!r}", flush=True)
        att.infra_error = f"finalize: {exc!r}"[:500]
        att.sandbox.candidates = []
        row = attempt_row(att, FLAGS.r2_config)
      pair_info[key]["rows"][att.attempt_idx] = row
      if len(pair_info[key]["rows"]) >= FLAGS.num_attempts:
        _write_pair(key)
    # f) progress
    if time.time() - last_progress >= FLAGS.progress_every_s:
      last_progress = time.time()
      el = time.time() - t_eval0
      print(
          f"[progress {el/60:.1f}m] pairs {len(pair_rows_written)}/{n_pairs_total} | active {len(active)}"
          f" (inflight {len(set(runner.inflight.values()))}, exec {len(runner.exec_futures)}, finalizing {len(fin_futures)})"
          f" | pending {len(pending)} | gen tok/s {runner.total_gen_tokens / max(1.0, el):.0f}"
          f" | total gen {runner.total_gen_tokens/1e6:.2f}M"
          f" | steps {runner.n_steps} (mean {1000*runner.step_time_total/max(1,runner.n_steps):.0f}ms, max {runner.step_time_max:.1f}s)"
          f" | {runner.engine_snapshot()}",
          flush=True,
      )

  fin_pool.shutdown(wait=True)
  runner.exec_pool.shutdown(wait=False)
  final = shard_summary(pair_rows_written, attempt_meta_written, time.time() - t_eval0)
  final["tpu_hardware"] = f"{num_tpus}x {device_kind} (tp={FLAGS.tensor_parallel_size})"
  final["engine_init_s"] = round(time.time() - t_init0, 1)
  try:
    with gfile.Open(summary_path, "w") as sf:
      json.dump(final, sf, indent=2)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"[WARN] final summary write failed: {exc}", flush=True)
  print("=" * 80, flush=True)
  print(json.dumps(final, indent=2), flush=True)
  print("=" * 80, flush=True)


if __name__ == "__main__":
  app_runner.run(main)
