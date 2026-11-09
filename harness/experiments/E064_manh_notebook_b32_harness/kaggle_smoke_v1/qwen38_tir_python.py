"""Opt-in stateful Python exploration, with host-owned ARC verification."""

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


WORKER = Path(__file__).with_name("qwen38_tir_python_worker.py")
TEMPLATE = Path(__file__).with_name("qwen38_tir_python_prompt.txt")
MAX_JOURNAL_SOURCE = 1_000_000


def function_tool(name, description, properties, required):
    return dict(type="function", function=dict(name=name, description=description,
        parameters=dict(type="object", properties=properties, required=required, additionalProperties=False)))


ENTRYPOINT = dict(type="string", description="Name of the function already defined in python, e.g. solve_puzzle.")
TOOLS = [
    function_tool("python", "Execute a small Python cell in this puzzle's persistent, isolated CPU session. "
        "Variables, arrays and helper functions persist. Public train pairs and test_inputs are preloaded. "
        "Use print or a final expression to inspect features and test hypotheses; no complete solver is required.",
        dict(code=dict(type="string", description="Python source for this cell, without markdown fences.")), ["code"]),
    function_tool("verifier", "Test the currently defined function on the original training examples. "
        "Return correctness, shape/cell errors and debug output. Does not submit or finish the puzzle. "
        "Execution uses a fresh replay of saved cells; validation never trusts mutated notebook data.",
        dict(entrypoint=ENTRYPOINT), ["entrypoint"]),
    function_tool("submit", "Submit the currently defined function. The host rechecks all original training "
        "examples and validates test prediction grids. Only a passing function can finish the puzzle. "
        "Test answers are never available. You do not need to repeat the Python source.",
        dict(entrypoint=ENTRYPOINT), ["entrypoint"]),
]


class StateError(RuntimeError):
    """A lost/unreproducible session is not an incorrect model prediction."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def public_task(task):
    # Deliberately project fields, even if the caller supplied evaluation labels.
    return deepcopy(dict(train=[dict(input=p["input"], output=p["output"]) for p in task["train"]],
                         test=[dict(input=p["input"]) for p in task["test"]]))


def parse_call(calls):
    if len(calls) != 1:
        raise ValueError("Make exactly one python, verifier or submit call per turn")
    call = calls[0]
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    key = "code" if name == "python" else "entrypoint"
    if name not in ("python", "verifier", "submit"):
        raise ValueError("Available tools are python, verifier and submit")
    if not isinstance(args, dict) or set(args) != {key} or not isinstance(args[key], str):
        raise ValueError(f"{name} requires exactly one string parameter: {key}")
    if name == "python":
        if not 0 < len(args[key]) <= 100_000:
            raise ValueError("Python cell must contain 1..100000 characters")
    elif not args[key].isidentifier() or args[key].startswith("_"):
        raise ValueError("Entrypoint must be a public Python identifier")
    return name, args[key]


class Child:
    def __init__(self, task, *, memory_mib, cpu_seconds):
        self.init = dict(task=public_task(task), memory_mib=memory_mib, cpu_seconds=cpu_seconds)
        self.proc = None
        self.stderr_task = None

    async def start(self, timeout):
        self.proc = await asyncio.create_subprocess_exec(
            # -I also ignores PYTHONHASHSEED, which makes replay of sets and
            # set-derived ordering inconsistent. -s/-P provide its user-site
            # and unsafe-path protections; replace (never inherit) the entire
            # environment so no PYTHONPATH/startup/credential settings leak in.
            sys.executable, "-s", "-P", "-B", str(WORKER), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=2_000_001, close_fds=True, start_new_session=True, cwd=str(WORKER.parent),
            env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})

        async def drain_stderr():
            # Drain without accumulating unbounded output or deadlocking a pipe.
            tail = b""
            while chunk := await self.proc.stderr.read(4096):
                tail = (tail + chunk)[-4096:]
            return tail

        self.stderr_task = asyncio.create_task(drain_stderr())
        response = await self.rpc(dict(op="init", payload=self.init), timeout)
        if response.get("error") or not response.get("ready"):
            raise StateError(f"Python isolation setup failed: {response.get('error')}")

    async def rpc(self, request, timeout):
        async with asyncio.timeout(timeout):
            if self.proc is None or self.proc.returncode is not None:
                raise StateError("Python worker is not alive")
            raw = json.dumps(request, allow_nan=False).encode() + b"\n"
            if len(raw) > 2_000_000:
                raise StateError("Host message exceeds worker limit")
            self.proc.stdin.write(raw)
            await self.proc.stdin.drain()
            line = await self.proc.stdout.readline()
            if not line or len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise StateError("Python worker died or returned an invalid response")
            response = json.loads(line)
            if not isinstance(response, dict) or not response.get("sandbox_enabled") or response.get("fatal"):
                raise StateError(f"Invalid/unisolated worker response: {str(response)[:1000]}")
            return response

    async def close(self):
        if self.proc is not None:
            if self.proc.returncode is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await self.proc.wait()
            if self.proc.stdin:
                self.proc.stdin.close()
        if self.stderr_task is not None:
            await asyncio.gather(self.stderr_task, return_exceptions=True)


class PythonSession:
    """Persistent scratch state; verification replays cells independently per grid.

    Ordinary exceptions keep their partial state and are journaled. Timeout or
    process death rolls back the uncommitted cell by killing the worker and
    replaying only committed cells on the next call. No arbitrary pickle load.
    """
    def __init__(self, task, *, timeout=5, memory_mib=512, max_calls=32,
                 verification_timeout=30, deadline=None):
        self.task = public_task(task)
        self.timeout = timeout
        self.memory_mib = memory_mib
        self.cpu_seconds = max(1, math.ceil((max_calls + 2) * timeout + verification_timeout))
        self.verification_timeout = verification_timeout
        self.deadline = deadline if deadline is not None else time.monotonic() + 86400
        self.journal = []
        self.child = None

    def remaining(self, seconds):
        left = min(seconds, self.deadline - time.monotonic())
        if left <= 0:
            raise TimeoutError("Python task deadline exhausted")
        return left

    def new_child(self):
        return Child(self.task, memory_mib=self.memory_mib, cpu_seconds=self.cpu_seconds)

    async def replay(self, child):
        for cell in self.journal:
            actual = await child.rpc(dict(op="cell", code=cell["code"]), self.remaining(self.timeout))
            if actual != cell["result"]:
                raise StateError("Python replay differs from recorded output/error; refusing stale state")

    async def ensure_started(self):
        if self.child is None:
            self.child = self.new_child()
            try:
                async with asyncio.timeout(self.remaining(self.verification_timeout)):
                    await self.child.start(self.remaining(self.timeout))
                    await self.replay(self.child)
            except BaseException:
                await self.close()
                raise

    async def execute(self, source):
        if sum(len(c["code"]) for c in self.journal) + len(source) > MAX_JOURNAL_SOURCE:
            return dict(sandbox_enabled=True, error="Python session source limit reached; cell not executed",
                        stdout="", stdout_truncated=False, committed=False)
        await self.ensure_started()
        try:
            result = await self.child.rpc(dict(op="cell", code=source), self.remaining(self.timeout))
        except asyncio.CancelledError:
            await self.close()
            raise
        except (TimeoutError, StateError, BrokenPipeError, ConnectionError, ValueError) as exc:
            await self.close()
            return dict(sandbox_enabled=True, error=f"{type(exc).__name__}: cell interrupted; "
                "uncommitted changes were rolled back. Earlier committed cells will be restored.",
                stdout="", stdout_truncated=False, committed=False)
        self.journal.append(dict(code=source, result=deepcopy(result)))
        return dict(result, committed=True)

    async def verify(self, entrypoint):
        report = dict(sandbox_enabled=True, error=None, cases=[])
        grids = [p["input"] for p in self.task["train"] + self.task["test"]]
        try:
            async with asyncio.timeout(self.remaining(self.verification_timeout)):
                for grid in grids:
                    # A process, not just globals, per input: a solver may have
                    # mutated a permitted module/class during a previous call.
                    child = self.new_child()
                    try:
                        await child.start(self.remaining(self.timeout))
                        await self.replay(child)
                        response = await child.rpc(dict(op="evaluate", entrypoint=entrypoint, grid=grid),
                                                   self.remaining(self.timeout))
                        if response.get("error") or len(response.get("cases", [])) != 1:
                            raise StateError("Invalid verification response")
                        report["cases"].extend(response["cases"])
                    finally:
                        await child.close()
        except TimeoutError:
            report.update(error="Verification/replay timed out; shorten expensive scratch computations", cases=[])
        return report

    async def close(self):
        if self.child is not None:
            child, self.child = self.child, None
            await child.close()


def python_feedback(result):
    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"].rstrip())
    if result.get("stdout_truncated"):
        parts.append("[Python output truncated; print a smaller summary.]")
    if result.get("error"):
        parts.append("Python error: " + result["error"])
    if not parts:
        parts.append("Cell executed successfully (no output). State retained.")
    return "\n".join(parts)


def append_feedback(payload, answer, calls, text):
    message = dict(role="assistant", content=answer.get("content") or "", tool_calls=deepcopy(calls))
    if answer.get("reasoning"):
        message.update(reasoning=answer["reasoning"], reasoning_content=answer["reasoning"])
    if answer.get("reasoning_details"):
        message["reasoning_details"] = deepcopy(answer["reasoning_details"])
    payload["messages"].append(message)
    for call in calls:
        payload["messages"].append(dict(role="tool", tool_call_id=call["id"],
            name=call["function"]["name"], content=text))


async def preflight(args, deadline):
    data = dict(train=[dict(input=[[1, 2]], output=[[2, 3]])], test=[dict(input=[[3]])])
    session = PythonSession(data, timeout=args.tir_timeout_seconds, memory_mib=args.tir_memory_mib,
                            verification_timeout=args.tir_python_verification_seconds, deadline=deadline)
    try:
        first = await session.execute("offset = 1\ncounts = Counter(train[0]['input'].ravel()).most_common(1)")
        second = await session.execute("assert counts == [(1, 1)]\ndef solve_puzzle(g):\n    return g + offset")
        report = await session.verify("solve_puzzle")
        if (first.get("error") or second.get("error") or report.get("error")
                or [c.get("prediction") for c in report["cases"]] != [[[2, 3]], [[4]]]):
            raise StateError(f"Stateful Python preflight failed: {first}, {second}, {report}")
        return dict(mode="python", state_persistence=True, independent_replay=True, **report)
    finally:
        await session.close()


async def solve_task(args, task, payload, count, client, url, output, deadline, *, stream_answer,
                     write_json, check_tokenization=None, on_predictions=None, tokenize_messages=None,
                     replay_root=None):
    # Lazy import avoids a cycle when qwen38_tir dispatches here.
    import qwen38_tir as tir
    output.mkdir()
    start = time.monotonic()
    deadline = min(deadline, start + args.request_timeout_seconds)
    task, payload = public_task(task), deepcopy(payload)
    if payload.get("tools") != TOOLS:
        raise ValueError("Tokenize the stateful Python tool schemas before starting TIR")
    replay_root = replay_root or output / "journal"
    replay_root.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=1, task=digest(task), initial_request=digest(payload),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(__file__), WORKER, Path(tir.worker.__file__), Path(tir.__file__))},
        options={name: getattr(args, name) for name in (
            "context", "tir_max_tokens_per_round", "tir_token_budget", "tir_timeout_seconds",
            "tir_memory_mib", "tir_python_max_calls", "tir_python_verification_seconds")})
    manifest = replay_root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise StateError("Stateful replay task/protocol/source changed; use a fresh experiment")
    if not manifest.exists():
        write_json(manifest, identity)
    session = PythonSession(task, timeout=args.tir_timeout_seconds, memory_mib=args.tir_memory_mib,
        max_calls=args.tir_python_max_calls, verification_timeout=args.tir_python_verification_seconds,
        deadline=deadline)
    initial_budget = payload["max_tokens"]
    total_completion, total_prompt, decode_seconds = 0, 0, 0.0
    best, sources = [None] * len(task["test"]), [None] * len(task["test"])
    rounds, error, cancelled, status = [], None, False, "round-limit"
    infrastructure_error = False
    try:
        for number in range(1, args.tir_python_max_calls + 1):
            if time.monotonic() >= deadline:
                status, error = "deadline", "Stateful Python task deadline exhausted"
                break
            if number > 1:
                async with asyncio.timeout(max(0.001, deadline-time.monotonic())):
                    if tokenize_messages is None:
                        response = await client.post(url + "/tokenize", json=tir.tokenization_payload(payload), timeout=120)
                        response.raise_for_status()
                        tokenization = response.json()
                    else:
                        tokenization = await tokenize_messages(payload)
                    if check_tokenization is not None:
                        check_tokenization(tokenization)
                    count = tokenization["count"]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid stateful Python prompt token count")
            available = tir.round_output_budget(args, count, initial_budget, total_completion)
            if available <= 0:
                status, error = "token-budget", "No room for another Python/tool turn; history was not truncated"
                break
            payload["max_tokens"] = available
            folder = output / f"round_{number:03d}"
            print(f"TIR python {output.name} turn {number}/{args.tir_python_max_calls}: "
                  f"prompt={count}, max_output={available}", flush=True)
            answer = await stream_answer(client, url, payload, folder,
                max(0.001, deadline-time.monotonic()), parse_grid=False)
            usage = answer.get("usage") or {}
            total_completion += usage.get("completion_tokens", 0)
            total_prompt += usage.get("prompt_tokens", 0)
            if answer.get("first_content_s") is not None and answer.get("last_content_s") is not None:
                decode_seconds += max(0, answer["last_content_s"] - answer["first_content_s"])
            record = dict(round=number, directory=folder.name, prompt_tokens=count, max_output_tokens=available,
                          usage=usage, error=answer.get("error"), finish_reason=answer.get("finish_reason"))
            rounds.append(record)
            if answer.get("cancelled") or answer.get("error") or answer.get("finish_reason") != "tool_calls":
                cancelled = bool(answer.get("cancelled"))
                status, error = "generation-failed", answer.get("error") or "Incomplete/non-tool response; no code executed"
                break
            calls = answer.get("tool_calls") or []
            if not calls:
                status, error = "generation-failed", "tool_calls finish without a complete call"
                break
            tir.validate_tool_history(calls)
            event_path = replay_root / f"turn_{number:03d}.json"
            before = digest(dict(cells=session.journal, calls=calls))
            old = json.loads(event_path.read_text()) if event_path.exists() else None
            if old is not None and old["before_sha256"] != before:
                raise StateError("Tool call or predecessor state changed during resume")
            execution, accepted = None, False
            try:
                name, argument = parse_call(calls)
            except (ValueError, TypeError, KeyError) as exc:
                name = "invalid"
                feedback = dict(text=f"Invalid tool call: {exc}. No cells were executed.")
            else:
                record["tool"] = name
                if name == "python":
                    (folder / "cell.py").write_text(argument)
                    # A previously rolled-back cell must not be silently retried
                    # on resume; its old feedback remains part of the transcript.
                    if old is not None and not old["execution"].get("committed"):
                        execution = old["execution"]
                    else:
                        execution = await session.execute(argument)
                    feedback = dict(text=python_feedback(execution))
                else:
                    execution = await session.verify(argument)
                    feedback = tir.training_feedback(execution, task)
                    all_tests_valid = (not execution.get("error") and
                        all(c.get("prediction") is not None and c.get("error") is None
                            for c in execution["cases"][len(task["train"]):]))
                    accepted = name == "submit" and feedback["all_training_correct"] and all_tests_valid
                    if name == "verifier":
                        feedback["text"] += "\nVerification only; not submitted. Continue exploring or call submit with the function name."
                    elif not accepted:
                        feedback["text"] += "\nSubmission rejected. Fix the implementation using python, then verify or submit again."
                    else:
                        feedback["text"] += "\nSubmission accepted; test predictions saved by the host."
                    record.update(training_correct=feedback["correct"], training_total=feedback["total"],
                                  entrypoint=argument)
            event = dict(before_sha256=before, after_sha256=digest(session.journal),
                         execution=execution, feedback=feedback, accepted=accepted)
            if old is not None and old != event:
                raise StateError("Replayed tool result differs from its immutable journal; refusing stale history")
            if old is None:
                write_json(event_path, event)
            if execution is not None:
                write_json(folder / "execution.json", execution)
            write_json(folder / "feedback.json", feedback)
            write_json(folder / "state.json", dict(cells=session.journal, sha256=digest(session.journal)))
            write_json(output / "rounds.json", rounds)
            append_feedback(payload, answer, calls, feedback["text"])
            write_json(folder / "messages_after_feedback.json", payload["messages"])
            if accepted:
                # A durable replay recipe, not a misleading standalone solver.
                recipe = dict(schema=1, entrypoint=argument, task=task, cells=session.journal,
                              source_identity=identity, journal_sha256=digest(session.journal))
                write_json(output / "accepted_session.json", recipe)
                best = [c["prediction"] for c in execution["cases"][len(task["train"]):]]
                sources = [str(output / "accepted_session.json")] * len(best)
                write_json(output / "accepted.json", dict(final_grids=best, solver_paths=sources))
                if on_predictions is not None:
                    on_predictions(best)
                status, error = "accepted", None
                break
    except asyncio.CancelledError:
        status, error, cancelled = "cancelled", "Cancelled: stateful Python task stopped", True
    except StateError as exc:
        status, error, infrastructure_error = "state-error", str(exc), True
    except Exception as exc:
        status, error, infrastructure_error = "failed", f"{type(exc).__name__}: {exc}", True
    finally:
        await session.close()
    if status == "round-limit":
        error = "Python tool-call limit exhausted without an accepted submission"
    result = dict(solve_mode="tir", tir_interface="python", tir_status=status, tir_rounds=len(rounds),
        rounds=rounds, final_grids=best, solver_paths=sources, error=error, cancelled=cancelled,
        infrastructure_error=infrastructure_error, finish_reason="stop" if status == "accepted" else None,
        usage=dict(prompt_tokens=total_prompt, completion_tokens=total_completion,
                   total_tokens=total_prompt+total_completion), initial_output_budget=initial_budget,
        token_budget_policy=args.tir_token_budget, elapsed_s=time.monotonic()-start,
        decode_tokens_per_s_estimate=total_completion/decode_seconds if decode_seconds > 0 else None)
    write_json(output / "result.json", result)
    return result
