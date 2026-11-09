"""ARC program synthesis and opt-in stateful Python exploration for Qwen3.8.

The parent never executes generated source or sends test labels to the worker.
"""

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import qwen38_tir_worker as worker
import qwen38_tir_python as python_mode
import qwen38_repl


TEXT_TEMPLATE = Path(__file__).with_name("qwen38_tir_prompt.txt")
TOOL_TEMPLATE = Path(__file__).with_name("qwen38_tir_tool_prompt.txt")
VERIFIER_TOOL = dict(type="function", function=dict(
    name="verifier",
    description=("Run a complete Python ARC solver on all supplied input grids in an isolated sandbox. "
                 "Return training-example correctness, shape/cell errors and bounded debug output. "
                 "Test answers are unknown. A solver passing every training example is used for test predictions."),
    parameters=dict(type="object", properties=dict(code=dict(type="string", description=(
        "Complete Python source, including imports, any helper functions, and solve_puzzle(input_grid). "
        "No markdown fences, input literals, main function or calls to the solver."))),
        required=["code"], additionalProperties=False)))


def tool_mode(args):
    return getattr(args, "tir_interface", "text") in ("tools", "python")


def template_path(args):
    # Preserve explicit custom templates and the old text-mode default.
    if getattr(args, "tir_interface", "text") == "python" and args.tir_template == TEXT_TEMPLATE:
        return python_mode.TEMPLATE
    if getattr(args, "tir_interface", "text") == "repl" and args.tir_template == TEXT_TEMPLATE:
        return qwen38_repl.TEMPLATE
    return TOOL_TEMPLATE if tool_mode(args) and args.tir_template == TEXT_TEMPLATE else args.tir_template


def add_verifier_tool(payload, interface="tools"):
    if interface not in ("tools", "python"):
        raise ValueError("Unknown native TIR interface")
    payload.update(tools=deepcopy(python_mode.TOOLS if interface == "python" else [VERIFIER_TOOL]),
                   tool_choice="auto", parallel_tool_calls=False)
    return payload


def source_paths():
    return (Path(__file__), Path(worker.__file__), Path(python_mode.__file__), python_mode.WORKER,
            Path(qwen38_repl.__file__), qwen38_repl.RUNTIME,
            Path(__file__).with_name("qwen38_repl_cot.py"),
            Path(__file__).with_name("qwen38_repl_challenger.py"),
            Path(__file__).with_name("qwen38_repl_audit.py"),
            Path(__file__).with_name("qwen38_repl_execution.py"),
            Path(__file__).with_name("qwen38_repl_interactions.py"),
            Path(__file__).with_name("qwen38_repl_disclosure.py"))


def tokenization_payload(payload):
    # The schema is part of the prompt; omitting it undercounts every tool turn.
    return {k: payload[k] for k in
            ("model", "messages", "tools", "chat_template_kwargs", "mm_processor_kwargs") if k in payload}


def template_messages(messages):
    """Match vLLM's wire-JSON -> template-object normalization (no mutation)."""
    result = deepcopy(messages)
    for message in result:
        if isinstance(message.get("reasoning"), str):
            message["reasoning_content"] = message["reasoning"]
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return result


class ToolCallAccumulator:
    """Assemble OpenAI streaming tool deltas by index; never execute partial code."""
    def __init__(self):
        self.calls = {}
        self.characters = 0

    def add(self, deltas):
        if not isinstance(deltas, list):
            raise ValueError("Tool-call deltas must be a list")
        for delta in deltas:
            index = delta.get("index")
            if type(index) is not int or not 0 <= index < 16:
                raise ValueError("Invalid tool-call index")
            call = self.calls.setdefault(index, dict(id="", type="function", function=dict(name="", arguments="")))
            if delta.get("type") not in (None, "function"):
                raise ValueError("Only function tool calls are supported")
            if delta.get("id"):
                if not isinstance(delta["id"], str) or (call["id"] and call["id"] != delta["id"]):
                    raise ValueError("Conflicting tool-call ID")
                call["id"] = delta["id"]
            for key in ("name", "arguments"):
                value = (delta.get("function") or {}).get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("Tool-call function fields must be strings")
                    call["function"][key] += value
                    self.characters += len(value)
            if self.characters > 1_000_000:
                raise ValueError("Tool-call stream exceeds size limit")

    def snapshot(self):
        return [deepcopy(self.calls[i]) for i in sorted(self.calls)]

    def validate_complete(self):
        calls = self.snapshot()
        if sorted(self.calls) != list(range(len(calls))):
            raise ValueError("Non-contiguous tool-call indices")
        ids = [call["id"] for call in calls]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError("Missing or duplicate tool-call IDs")
        if any(not call["function"]["name"] for call in calls):
            raise ValueError("Tool call has no function name")
        return calls


def verifier_code(calls):
    if len(calls) != 1:
        raise ValueError("Call verifier exactly once per turn with one complete candidate program")
    call = calls[0]
    if call.get("type") != "function" or call.get("function", {}).get("name") != "verifier":
        raise ValueError("Only the verifier function is available")
    arguments = json.loads(call["function"]["arguments"])
    if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str):
        raise ValueError("verifier expects exactly one string parameter named code")
    code = arguments["code"]
    _, entry = worker.validate_code(code)
    return code.rstrip() + "\n", entry


def validate_tool_history(calls):
    """Malformed envelopes cannot be replayed by the checkpoint chat template."""
    ids = []
    for call in calls:
        if (not isinstance(call, dict) or call.get("type") != "function"
                or not isinstance(call.get("id"), str) or not call["id"]):
            raise ValueError("Missing function tool-call type or ID")
        ids.append(call["id"])
        function = call.get("function") or {}
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise ValueError("Missing tool function name")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("Wire tool arguments must be a JSON string")
        if not isinstance(json.loads(function["arguments"]), dict):
            raise ValueError("Tool arguments must decode to an object")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tool-call IDs")


def append_tool_feedback(payload, answer, calls, feedback):
    message = dict(role="assistant", content=answer.get("content") or "", tool_calls=deepcopy(calls))
    if answer.get("reasoning"):
        # Current vLLM's wire parser reads reasoning, then exposes both names to
        # Jinja. The checkpoint/older servers use reasoning_content directly.
        message["reasoning"] = answer["reasoning"]
        message["reasoning_content"] = answer["reasoning"]
    if answer.get("reasoning_details"):
        message["reasoning_details"] = deepcopy(answer["reasoning_details"])
    payload["messages"].append(message)
    for call in calls:
        payload["messages"].append(dict(role="tool", tool_call_id=call["id"], name=call["function"]["name"],
            content=feedback["text"] + "\n\nIf revision is needed, call verifier again with the complete replacement program."))


def add_arguments(parser):
    parser.add_argument("--solve-mode", choices=("direct", "tir", "hybrid"), default="direct",
                        help="direct: CoT per test input; tir: one program per task; hybrid: REPL for large-test-input tasks, otherwise CoT.")
    parser.add_argument("--hybrid-min-test-area", type=int, default=400,
                        help="Hybrid uses task-level REPL if any test input has at least this many cells. No test labels used.")
    parser.add_argument("--task-token-budget", type=int,
                        help="Aggregate completion-token cap per task (reasoning included). REPL shares it across all turns/trajectories; "
                             "direct divides it evenly across test inputs, without redistributing unused shares. "
                             "With --tir-repl-second-attempt cot this caps ONLY the first REPL pass, not CoT. "
                             "Default: 150000 for hybrid; no additional cap for existing direct/TIR modes. REPL or direct only.")
    parser.add_argument("--tir-interface", choices=("text", "tools", "python", "repl"), default="text",
                        help="text: code-block loop; tools: complete-program verifier; python: persistent Python cells, "
                             "verifier and explicit submit; repl: the arc2_harness REPL loop on flash-llama, flash-next, 27b, qwen35-9b or dsv4.")
    parser.add_argument("--tir-max-rounds", type=int, default=4)
    parser.add_argument("--tir-max-tokens-per-round", type=int,
                        help="Reasoning plus code per revision; defaults to 32768 for text/tools, full context for python.")
    parser.add_argument("--tir-token-budget", choices=("total", "context"),
                        help="total shares generation budget; context bounds each full history plus output. Defaults: total for text/tools, context for python.")
    parser.add_argument("--tir-python-max-calls", type=int, default=32,
                        help="Stateful Python tool turns per puzzle (independent of the complete-program --tir-max-rounds).")
    parser.add_argument("--tir-python-verification-seconds", type=float, default=30,
                        help="Stateful Python full verification/replay wall timeout; --tir-timeout-seconds still bounds individual cells.")
    parser.add_argument("--tir-timeout-seconds", type=float, default=5,
                        help="CPU wall limit per proposed program (text/tools), or individual cell/solver call (python).")
    parser.add_argument("--tir-memory-mib", type=int, default=512)
    parser.add_argument("--tir-template", type=Path, default=TEXT_TEMPLATE,
                        help="Custom prompt; tools/python automatically select their native-tool prompt unless overridden.")
    qwen38_repl.add_arguments(parser)


def validate_options(args):
    if args.solve_mode not in ("direct", "tir", "hybrid"):
        raise ValueError("Unknown solve mode")
    if getattr(args, "tir_repl_second_attempt", "repl") == "cot" and not (
            args.solve_mode == "tir" and args.tir_interface == "repl"
            and getattr(args, "tir_repl_trajectory_schedule", "sequential") == "queued"):
        raise ValueError("CoT second attempt requires queued REPL solving")
    if getattr(args, "tir_repl_trajectory_schedule", "sequential") == "queued" and not (
            args.solve_mode == "tir" and args.tir_interface == "repl"):
        raise ValueError("Queued trajectories require --solve-mode tir --tir-interface repl")
    if getattr(args, "tir_interface", "text") not in ("text", "tools", "python", "repl"):
        raise ValueError("Unknown TIR interface")
    if getattr(args, "tir_repl_holdout_first", False) and (
        args.solve_mode not in ("tir", "hybrid") or args.tir_interface != "repl"
    ):
        raise ValueError("--tir-repl-holdout-first requires TIR/REPL solving (or hybrid with REPL)")
    if type(args.hybrid_min_test_area) is not int or not 1 <= args.hybrid_min_test_area <= 900:
        raise ValueError("--hybrid-min-test-area must be an integer in 1..900")
    if args.solve_mode == "hybrid":
        if args.tir_interface != "repl":
            raise ValueError("--solve-mode hybrid requires --tir-interface repl")
        if args.task_token_budget is None:
            args.task_token_budget = 150_000
    if args.task_token_budget is not None:
        if type(args.task_token_budget) is not int or not 1000 <= args.task_token_budget <= 2_000_000:
            raise ValueError("--task-token-budget must be an integer in 1000..2000000")
        if args.solve_mode != "direct" and args.tir_interface != "repl":
            raise ValueError("--task-token-budget supports direct or REPL solving only")
    if getattr(args, "tir_repl_test_reveal", "all") != "all" and not (
            args.solve_mode in ("tir", "hybrid") and args.tir_interface == "repl"):
        raise ValueError("--tir-repl-test-reveal requires REPL solving")
    if args.solve_mode in ("tir", "hybrid") and args.tir_interface == "repl":
        if args.qwen38_variant not in ("flash-llama", "flash-next", "27b", "qwen35-9b", "dsv4", "dsv4-vllm"):
            raise ValueError("--tir-interface repl runs on --qwen38-variant flash-llama, flash-next, 27b, qwen35-9b, "
                             "dsv4 or dsv4-vllm")
        if getattr(args, "input_mode", "text") != "text" and args.qwen38_variant == "flash-llama":
            raise ValueError("Multimodal REPL requires the vLLM 27b, flash-next or qwen35-9b backend, or dsv4 with its mmproj")
        qwen38_repl.validate_options(args)
    if args.tir_max_tokens_per_round is None:
        args.tir_max_tokens_per_round = args.context if args.tir_interface == "python" else 32768
    if args.tir_token_budget is None:
        args.tir_token_budget = "context" if args.tir_interface == "python" else "total"
    if args.tir_max_rounds <= 0 or args.tir_max_tokens_per_round <= 0:
        raise ValueError("TIR round/token limits must be positive")
    if getattr(args, "tir_token_budget", "total") not in ("total", "context"):
        raise ValueError("Unknown TIR token budget policy")
    if not math.isfinite(args.tir_timeout_seconds) or not 0 < args.tir_timeout_seconds <= 300:
        raise ValueError("--tir-timeout-seconds must be positive and at most 300")
    if not 128 <= args.tir_memory_mib <= 4096:
        raise ValueError("--tir-memory-mib must be between 128 and 4096")
    if not 1 <= args.tir_python_max_calls <= 128:
        raise ValueError("--tir-python-max-calls must be between 1 and 128")
    if not math.isfinite(args.tir_python_verification_seconds) or not 0 < args.tir_python_verification_seconds <= 300:
        raise ValueError("--tir-python-verification-seconds must be positive and at most 300")


def task_solve_mode(args, task):
    """A whole task shares its REPL program, even when only one test input is large."""
    if args.solve_mode != "hybrid":
        return args.solve_mode
    max_area = max((len(p["input"]) * len(p["input"][0]) for p in task["test"]), default=0)
    return "tir" if max_area >= args.hybrid_min_test_area else "direct"


def direct_token_allowance(task_budget, test_count, index, context_budget):
    """Static shares make the total cap independent of concurrency and finish order."""
    if task_budget is None:
        return context_budget
    share, remainder = divmod(task_budget, test_count)
    allowance = min(context_budget, share + int(index < remainder))
    if allowance <= 0:
        raise ValueError("Task token budget leaves no tokens for this test input")
    return allowance


def make_prompt(template, task):
    if not task["train"]:
        raise ValueError("TIR needs at least one training pair for verification")
    examples = "\n\n".join(f"Example {i}:\nINPUT: {json.dumps(p['input'])}\nOUTPUT: {json.dumps(p['output'])}"
                             for i, p in enumerate(task["train"]))
    tests = "\n\n".join(f"Test input {i}:\n{json.dumps(p['input'])}" for i, p in enumerate(task["test"]))
    return template.format(training_examples=examples, test_inputs=tests)


def extract_code(content):
    blocks = re.findall(r"^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*\r?$", content,
                        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if len(blocks) > 1:
        raise ValueError("Return one complete Python code block, not multiple alternatives")
    code = blocks[0].strip() if blocks else content.strip()
    _, entrypoint = worker.validate_code(code)
    return code + "\n", entrypoint


async def execute_code(code, grids, *, timeout, memory_mib):
    """Minimal environment, isolated Python, bounded pipes and owned-child cleanup."""
    start = time.monotonic()
    proc, readers = None, []
    report = dict(sandbox_enabled=False, error=None, cases=[])

    async def read_limited(stream, limit):
        chunks, total = [], 0
        while chunk := await stream.read(65536):
            total += len(chunk)
            if total > limit:
                raise ValueError("Sandbox output limit exceeded")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        async with asyncio.timeout(timeout):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-B", str(Path(worker.__file__).resolve()),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(worker.__file__).resolve().parent), start_new_session=True, close_fds=True,
                env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
            readers = [asyncio.create_task(read_limited(proc.stdout, 2_000_000)),
                       asyncio.create_task(read_limited(proc.stderr, 65536))]
            proc.stdin.write(json.dumps(dict(code=code, grids=grids, cpu_seconds=max(1, math.ceil(timeout)),
                                             memory_mib=memory_mib)).encode())
            await proc.stdin.drain()
            proc.stdin.close()
            stdout, stderr, returncode = await asyncio.gather(*readers, proc.wait())
            if returncode:
                raise RuntimeError(f"Solver child exited {returncode}: {stderr.decode(errors='replace')[-2000:]}")
            report = json.loads(stdout)
            if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
                raise ValueError("Invalid sandbox response")
            if not report.get("error") and (not report.get("sandbox_enabled") or len(report["cases"]) != len(grids)):
                raise ValueError("Incomplete/unisolated sandbox response")
    except Exception as exc:
        report = dict(sandbox_enabled=False, error=f"{type(exc).__name__}: {exc}", cases=[])
    finally:
        if proc is not None:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await proc.wait()
            if proc.stdin is not None:
                proc.stdin.close()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    report["elapsed_s"] = time.monotonic()-start
    return report


async def preflight(args, deadline):
    if getattr(args, "tir_interface", "text") == "python":
        return await python_mode.preflight(args, deadline)
    if getattr(args, "tir_interface", "text") == "repl":
        return await qwen38_repl.preflight(args, deadline)
    probe = ("import numpy as np\nfrom collections import Counter\ndef solve_puzzle(input_grid):\n"
             "    assert int(input_grid.max()) == 2\n"
             "    assert Counter(input_grid.ravel()).most_common(1) == [(1, 1)]\n"
             "    print('NumPy ready', np.unique(input_grid))\n"
             "    return input_grid.copy()\n")
    report = await execute_code(probe, [[[1, 2]]],
        timeout=min(args.tir_timeout_seconds, max(0.001, deadline-time.monotonic())), memory_mib=args.tir_memory_mib)
    if (report.get("error") or not report.get("sandbox_enabled") or not report["cases"]
            or report["cases"][0].get("prediction") != [[1, 2]]):
        raise RuntimeError(f"TIR sandbox preflight failed; no unsafe fallback: {report}")
    return report


def training_feedback(report, task):
    """Expected values are training-only and stay outside generated-code execution."""
    train = task["train"]
    if report.get("error"):
        return dict(all_training_correct=False, correct=0, total=len(train), examples=[],
                    text="Program execution failed: " + report["error"][:2000])
    if not report.get("sandbox_enabled") or len(report.get("cases", [])) != len(train) + len(task["test"]):
        raise ValueError("Incomplete sandbox report")
    examples, summaries, details = [], [], []
    for i, (case, pair) in enumerate(zip(report["cases"], train)):
        prediction, truth = case["prediction"], pair["output"]
        exact = prediction == truth and case["error"] is None
        row = dict(index=i, correct=exact, error=case["error"], predicted=prediction, expected=truth)
        if case["error"]:
            summary = (f"Example {i}: runtime/invalid-output error: {case['error']} "
                       f"(expected shape {[len(truth), len(truth[0])]})")
        elif exact:
            summary = f"Example {i}: correct."
        elif (len(prediction), len(prediction[0])) != (len(truth), len(truth[0])):
            row.update(predicted_shape=[len(prediction), len(prediction[0])], expected_shape=[len(truth), len(truth[0])])
            summary = f"Example {i}: wrong shape {row['predicted_shape']}; expected {row['expected_shape']}."
        else:
            cells = [dict(row=r, col=c, predicted=v, expected=truth[r][c])
                     for r, line in enumerate(prediction) for c, v in enumerate(line) if v != truth[r][c]]
            row["wrong_cells"] = len(cells)
            summary = f"Example {i}: {len(cells)} wrong cells."
            if len(cells) <= 5:
                row["cell_details"] = cells
                summary += " " + "; ".join(f"({x['row']},{x['col']}) predicted {x['predicted']}, expected {x['expected']}" for x in cells)
        examples.append(row)
        summaries.append(summary)
        if not exact:
            details.append(f"Example {i} PREDICTED: {json.dumps(prediction, separators=(',', ':'))}\n"
                           f"Example {i} EXPECTED: {json.dumps(truth, separators=(',', ':'))}")
    exact = sum(row["correct"] for row in examples)
    head = (f"All {len(train)} training examples are correct." if exact == len(train)
            else f"{exact}/{len(train)} training examples are correct.")
    text = head + "\n" + "\n".join(summaries)
    for i, case in enumerate(report["cases"]):
        name = f"Example {i}" if i < len(train) else f"Test input {i-len(train)}"
        if i >= len(train) and case["error"]:
            text += f"\n{name}: runtime/invalid-output error: {case['error']} (no test answer available)."
        if case.get("stdout"):
            details.append(f"{name} debug output{' (truncated)' if case['stdout_truncated'] else ''}:\n{case['stdout']}")
    for detail in details:
        if len(text) + len(detail) > 32768:
            text += "\nAdditional grid/debug details omitted to bound feedback size."
            break
        text += "\n\n" + detail
    if len(text) > 32768:
        text = text[:32600] + "\nFeedback truncated; full per-example reports are saved by the host."
    return dict(all_training_correct=bool(train) and exact == len(train), correct=exact, total=len(train),
                examples=examples, text=text)


def round_output_budget(args, prompt_tokens, initial_budget, completion_so_far):
    """Never truncate conversation history; retain the normal context reserve."""
    available = min(args.context-prompt_tokens-16, args.tir_max_tokens_per_round)
    if getattr(args, "tir_token_budget", "total") == "total":
        available = min(available, initial_budget-completion_so_far)
    return available


async def solve_task(args, task, payload, count, client, url, output, deadline, *, stream_answer, write_json,
                     check_tokenization=None, on_predictions=None, tokenize_messages=None, replay_root=None):
    """One conversation for every test input, with explicit total/context budgets."""
    if getattr(args, "tir_interface", "text") == "python":
        return await python_mode.solve_task(args, task, payload, count, client, url, output, deadline,
            stream_answer=stream_answer, write_json=write_json, check_tokenization=check_tokenization,
            on_predictions=on_predictions, tokenize_messages=tokenize_messages, replay_root=replay_root)
    output.mkdir()
    start = time.monotonic()
    deadline = min(deadline, start + args.request_timeout_seconds)
    payload = deepcopy(payload)
    native_tools = tool_mode(args)
    if native_tools and payload.get("tools") != [VERIFIER_TOOL]:
        raise ValueError("Tokenize the verifier schema before starting native TIR")
    initial_budget = payload["max_tokens"]
    total_completion, total_prompt, decode_seconds = 0, 0, 0.0
    best = [None] * len(task["test"])
    sources = [None] * len(best)
    rounds, error, cancelled, status = [], None, False, "round-limit"
    grids = [p["input"] for p in task["train"]] + [p["input"] for p in task["test"]]
    try:
        for number in range(1, args.tir_max_rounds+1):
            if time.monotonic() >= deadline:
                status, error = "deadline", "TIR task deadline exhausted"
                break
            if number > 1:
                async with asyncio.timeout(max(0.001, deadline-time.monotonic())):
                    if tokenize_messages is None:
                        response = await client.post(url + "/tokenize", json=tokenization_payload(payload), timeout=120)
                        response.raise_for_status()
                        tokenization = response.json()
                    else:
                        # API benchmarks can use a pinned local tokenizer when
                        # their provider does not expose vLLM's /tokenize API.
                        tokenization = await tokenize_messages(payload)
                    if check_tokenization is not None:
                        check_tokenization(tokenization)
                    count = tokenization["count"]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid TIR prompt token count")
            available = round_output_budget(args, count, initial_budget, total_completion)
            if available <= 0:
                status, error = "token-budget", "No room for another TIR round; conversation was not truncated"
                break
            payload["max_tokens"] = available
            folder = output / f"round_{number:03d}"
            print(f"TIR {output.name} round {number}/{args.tir_max_rounds}: prompt={count}, max_output={available}", flush=True)
            answer = await stream_answer(client, url, payload, folder, max(0.001, deadline-time.monotonic()), parse_grid=False)
            usage = answer.get("usage") or {}
            total_completion += usage.get("completion_tokens", 0)
            total_prompt += usage.get("prompt_tokens", 0)
            if answer.get("first_content_s") is not None and answer.get("last_content_s") is not None:
                decode_seconds += max(0, answer["last_content_s"]-answer["first_content_s"])
            record = dict(round=number, directory=folder.name, prompt_tokens=count, max_output_tokens=available,
                          usage=usage, error=answer["error"], finish_reason=answer["finish_reason"])
            rounds.append(record)
            expected_finish = "tool_calls" if native_tools else "stop"
            if answer["cancelled"] or answer["error"] or answer["finish_reason"] != expected_finish:
                cancelled = answer["cancelled"]
                status, error = "generation-failed", answer["error"] or "Incomplete/length-limited program; not executed"
                if native_tools and answer["finish_reason"] == "stop" and not answer["error"]:
                    status, error = "no-tool-call", "Model ended without calling verifier; text was not executed"
                break
            calls = answer.get("tool_calls") or []
            if native_tools and not calls:
                status, error = "generation-failed", "tool_calls finish without any complete tool call"
                break
            if native_tools:
                try:
                    validate_tool_history(calls)
                except (ValueError, TypeError) as exc:
                    status, error = "invalid-tool-call", f"Invalid tool envelope; not executed: {exc}"
                    break
            try:
                code, entry = verifier_code(calls) if native_tools else extract_code(answer["content"])
            except (SyntaxError, ValueError, RecursionError) as exc:
                feedback = dict(all_training_correct=False, correct=0, total=len(task["train"]),
                                examples=[], text=f"Invalid program: {type(exc).__name__}: {str(exc)[:2000]}")
            else:
                (folder / "solver.py").write_text(code)
                record.update(entrypoint=entry, source_sha256=hashlib.sha256(code.encode()).hexdigest())
                execution = await execute_code(code, grids,
                    timeout=min(args.tir_timeout_seconds, max(0.001, deadline-time.monotonic())), memory_mib=args.tir_memory_mib)
                write_json(folder / "execution.json", execution)
                feedback = training_feedback(execution, task)
                if feedback["all_training_correct"]:
                    for i, case in enumerate(execution["cases"][len(task["train"]):]):
                        if case["prediction"] is not None and case["error"] is None:
                            best[i], sources[i] = case["prediction"], str(folder / "solver.py")
                    write_json(output / "accepted.json", dict(final_grids=best, solver_paths=sources))
                    if on_predictions is not None:
                        on_predictions(best)
            record.update(training_correct=feedback["correct"], training_total=feedback["total"])
            write_json(folder / "feedback.json", feedback)
            write_json(output / "rounds.json", rounds)
            if native_tools:
                append_tool_feedback(payload, answer, calls, feedback)
                write_json(folder / "messages_after_feedback.json", payload["messages"])
            print(f"TIR {output.name} round {number}: {feedback['correct']}/{feedback['total']} training examples correct", flush=True)
            if all(grid is not None for grid in best):
                status, error = "accepted", None
                break
            if not native_tools:
                payload["messages"].extend([
                    dict(role="assistant", content=answer["content"]),
                    dict(role="user", content=feedback["text"] + "\n\nRevise the general rule. Return the complete replacement Python program in one code block, including imports and all helper functions, not grids or a diff.")])
    except asyncio.CancelledError:
        status, error, cancelled = "cancelled", "Cancelled: pipeline stopped or server exited", True
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    if status == "round-limit":
        error = "TIR round limit exhausted without a passing solver for every test input"
    result = dict(solve_mode="tir", tir_interface=getattr(args, "tir_interface", "text"),
                  tir_status=status, tir_rounds=len(rounds), rounds=rounds,
                  final_grids=best, solver_paths=sources, error=error, cancelled=cancelled,
                  finish_reason="stop" if status == "accepted" else None,
                  usage=dict(prompt_tokens=total_prompt, completion_tokens=total_completion,
                             total_tokens=total_prompt+total_completion),
                  initial_output_budget=initial_budget, token_budget_policy=getattr(args, "tir_token_budget", "total"),
                  elapsed_s=time.monotonic()-start,
                  decode_tokens_per_s_estimate=total_completion/decode_seconds if decode_seconds > 0 else None)
    write_json(output / "result.json", result)
    return result
