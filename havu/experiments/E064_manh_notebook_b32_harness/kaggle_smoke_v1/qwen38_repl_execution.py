"""Bounded, label-blind observations of the saved solver's actual execution.

Expected values are model predictions, NOT hidden answers. Watches do not evaluate
expressions or run a second handwritten approximation of the solver. Execution is
inside the existing resource-limited sandbox, against the accepted journal snapshot.
"""
import inspect
import json
import math
import sys
import types


GUIDANCE = """

# Check execution, then challenge the mechanism
For this experiment, replace the generic post-submit input audit with one concrete
execution check. audit_inputs remains available as a supplementary measurement,
but matching input features cannot validate the rule. Do not simply reprint grids.

Choose a consequential test case and PREDICT what the saved solver will do before
calling trace_solver. Inspect a local variable at the stage actually used to make
the decision, and a resulting output cell/value. The tool executes the SAVED solver
and its helpers; later Python redefinitions do not change that snapshot. It records
whether those predicted effects actually occurred, without seeing test answers.
An empty watches list returns numbered source for the saved function; it is not a
completed check. Each watch names a local variable, optional function_name, optional
line (1-based within the returned source, BEFORE that line executes; omitted means
function return), optional path of integer/string indices, occurrence ('last' by
default, or a zero-based event index), and expected scalar or short scalar list.
Use variable='$output' for the actual returned grid and path=[row,col] for a cell.
Paths are indexing only, not Python expressions. Prefer at most 4 small watches.
At least one local watch and one output watch must be observed. If the solver has
several stages, watch the state AFTER preprocessing, not just the original input.
The same watch specification can be used on a training input for comparison,
but a training-only trace does not fulfill the required test-input check.
Tool matches mean execution matches your prediction, NOT that the ARC rule is true.
An unreached line, missing variable, timeout or truncation is UNKNOWN, not a pass.
If actual effects contradict your explanation, reconcile the executed stages before
claiming success; do not patch particular test cells. Submit a revised general
solver only if it still passes every training pair. The old answer stays safe.

Then challenge the CORE mechanism, not merely an arbitrary threshold or tie-break.
State a competing causal explanation consistent with concrete training observations,
what observation distinguishes the explanations, and what predictions would differ.
Use compare_rules with change_scope='mechanism', 'parameter', 'boundary', 'tie-break'
or 'none', and a discriminator describing that evidence. Small variants are useful
but do not establish that their shared mechanism is correct. If no defensible
alternative exists, omit function_name and state why; that is a model judgment,
not a verified generalization certificate. Never invent an alternative just to
fill attempt_2, assume universal color/rotation invariance, or use test answers.
All checks use the SAME existing token/time/review budgets; keep them concise.
"""

REQUEST = (
    "[execution check] The preferred candidate is saved; test correctness is UNKNOWN. "
    "Predict one consequential test effect, then call trace_solver(split='test', index=..., "
    "watches=[...]) with a local-state watch at the actual decision stage and a '$output' "
    "watch. Each needs expected, variable and optional path, function_name, line, occurrence. "
    "Omit line for a function-return snapshot; watches=[] shows numbered saved source. "
    "Check the executed intermediate state, not a separate calculation on raw input. "
    "Reconcile contradictions before asserting the change works. Existing review limits apply."
)
MECHANISM_REQUEST = (
    "[mechanism challenge] Execution consistency is not rule correctness. Name the strongest "
    "defensible competing mechanism and a concrete discriminator from visible evidence. "
    "Use compare_rules with change_scope and discriminator, plus uncertainty and evidence. "
    "A parameter/boundary/tie-break variant does not validate its shared mechanism. "
    "If evidence leaves no defensible alternative, omit function_name and record that reasoning; "
    "do not manufacture a different answer. Keep the preferred candidate and existing budget."
)
SCOPES = ("mechanism", "parameter", "boundary", "tie-break", "none")


def pending_request(record):
    preferred = record.get("accepted")
    if not record.get("execution_audit_enabled") or preferred is None:
        return None
    if record.get("execution_audit_completed_for") != preferred:
        return REQUEST
    if record.get("mechanism_review_completed_for") != preferred:
        return MECHANISM_REQUEST
    return None


def scalar(value):
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float) and abs(value) <= 1e12 and math.isfinite(value):
        return value
    if type(value) is str and len(value) <= 120:
        return value
    raise ValueError("Expected/observed values must be finite scalars or strings up to 120 characters")


def small_value(value):
    if type(value) in (list, tuple):
        if len(value) > 16:
            raise ValueError("Use path to select at most 16 scalar values, not a grid dump")
        return [scalar(x) for x in value]
    return scalar(value)


def validate_spec(spec, task):
    if not isinstance(spec, dict) or set(spec) - {"split", "index", "watches", "inspect_function"}:
        raise ValueError("Trace accepts split, index, watches and optional inspect_function only")
    split, index = spec.get("split"), spec.get("index")
    if split not in ("train", "test") or type(index) is not int or not 0 <= index < len(task[split]):
        raise ValueError("Choose an existing train/test input using its zero-based index")
    name = spec.get("inspect_function")
    if name is not None and (not isinstance(name, str) or not name.isidentifier()):
        raise ValueError("inspect_function must be a named saved function")
    watches = spec.get("watches")
    if not isinstance(watches, list) or len(watches) > 6:
        raise ValueError("watches must be a list of at most 6 watches; [] inspects saved source")
    for watch in watches:
        if not isinstance(watch, dict) or set(watch) - {
                "variable", "path", "expected", "function_name", "line", "occurrence"}:
            raise ValueError("Unknown watch fields")
        var = watch.get("variable")
        if not isinstance(var, str) or not (var.isidentifier() or var == "$output"):
            raise ValueError("variable must be a local name or '$output'")
        name = watch.get("function_name")
        if name is not None and (not isinstance(name, str) or not name.isidentifier()):
            raise ValueError("function_name must be a named saved function")
        line = watch.get("line")
        if line is not None and (type(line) is not int or not 1 <= line <= 10000):
            raise ValueError("line must be a positive source-relative line number")
        occurrence = watch.get("occurrence", "last")
        if occurrence != "last" and (type(occurrence) is not int or not 0 <= occurrence < 200000):
            raise ValueError("occurrence must be 'last' or a zero-based event index")
        path = watch.get("path", [])
        if not isinstance(path, list) or len(path) > 4 or any(
                not ((type(x) is int and abs(x) <= 10000) or (type(x) is str and len(x) <= 80)) for x in path):
            raise ValueError("path must have at most 4 integer/string indices, not expressions")
        if var == "$output" and any(k in watch for k in ("line", "function_name", "occurrence")):
            raise ValueError("$output has one final value and no function/line/occurrence")
        if "expected" not in watch:
            raise ValueError("Predict expected before tracing; this is not a hidden test answer")
        small_value(watch["expected"])
    return spec


def trace_execution(fn_name, ns, task, spec, *, np, to_grid, max_events=200000):
    """Runs under the runtime's alarm/seccomp. Never changes the model's code."""
    validate_spec(spec, task)
    fn = ns.get(fn_name)
    if type(fn) is not types.FunctionType:
        raise ValueError("Saved solver must be a Python function to trace")
    if not spec["watches"]:
        target = ns.get(spec.get("inspect_function", fn_name))
        if type(target) is not types.FunctionType or not target.__code__.co_filename.startswith("<cell "):
            raise ValueError("Only saved model-defined functions can be inspected")
        lines, _ = inspect.getsourcelines(target)
        return dict(inspection=True, source="".join(f"{i}: {s}" for i, s in enumerate(lines[:160], 1)),
                    source_truncated=len(lines) > 160, function=spec.get("inspect_function", fn_name),
                    available_functions=sorted(k for k, v in ns.items() if type(v) is types.FunctionType
                        and v.__code__.co_filename.startswith("<cell "))[:40])

    observations, targets = [], []
    for watch in spec["watches"]:
        observations.append(dict(watch=watch, observed=None, hits=0, error="not reached", matches=None))
        if watch["variable"] == "$output":
            targets.append(None)
            continue
        target = ns.get(watch.get("function_name", fn_name))
        if type(target) is not types.FunctionType or not target.__code__.co_filename.startswith("<cell "):
            raise ValueError("Watches must name a saved model-defined function")
        lines, start = inspect.getsourcelines(target)
        line = watch.get("line")
        if line is not None and line > len(lines):
            raise ValueError("Watch line is outside saved source; inspect with watches=[] first")
        targets.append((target.__code__, start + line - 1 if line is not None else None))

    def observe(i, value, available=True):
        row, watch = observations[i], spec["watches"][i]
        occurrence = watch.get("occurrence", "last")
        row["hits"] += 1
        if occurrence != "last" and row["hits"] - 1 != occurrence:
            return
        try:
            if not available:
                raise ValueError("variable unavailable at this stage")
            for key in watch.get("path", []):
                if type(value) not in (dict, list, tuple, np.ndarray):
                    raise ValueError("Path must index a plain container/NumPy array")
                value = value[key]
            if type(value) is np.ndarray:
                if value.size > 16 or value.ndim > 1:
                    raise ValueError("Use path to select a scalar/short vector from this array")
                value = value.tolist()
            if isinstance(value, np.generic):
                value = value.item()
            if type(value) in (list, tuple):
                value = [x.item() if isinstance(x, np.generic) else x for x in value]
            row.update(observed=small_value(value), error=None)
            row["matches"] = row["observed"] == watch["expected"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            row.update(observed=None, matches=None, error=str(exc)[:240])

    watched_codes = {target[0] for target in targets if target is not None}
    line_codes = {target[0] for target in targets if target is not None and target[1] is not None}
    events = 0

    def trace(frame, event, arg):
        nonlocal events
        if frame.f_code not in watched_codes:
            return None
        if event == "call":
            frame.f_trace_lines = frame.f_code in line_codes
        events += 1
        if events > max_events:
            raise ValueError("Execution trace event limit exceeded; use a smaller helper/checkpoint")
        for i, target in enumerate(targets):
            if target is None or target[0] is not frame.f_code:
                continue
            if (target[1] is None and event == "return") or (event == "line" and target[1] == frame.f_lineno):
                var = spec["watches"][i]["variable"]
                observe(i, frame.f_locals.get(var), var in frame.f_locals)
        return trace

    previous = sys.gettrace()
    error, output = None, None
    try:
        sys.settrace(trace)
        output = to_grid(fn(np.array(task[spec["split"]][spec["index"]]["input"])))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        sys.settrace(previous)
    if error is None:
        for i, watch in enumerate(spec["watches"]):
            if watch["variable"] == "$output":
                observe(i, output)
    complete = error is None and all(row["error"] is None for row in observations)
    return dict(inspection=False, error=error, output=output, observations=observations,
                events=events, complete=complete, expectations_match=(
                    all(row["matches"] for row in observations) if complete else None))


def tool_schema(progress_note, required_note):
    scalar_schema = {"type": ["boolean", "number", "string", "null"]}
    return {"type": "function", "function": {
        "name": "trace_solver", "description": "Check predicted local-state and output effects of the SAVED "
            "preferred solver in the sandbox. No test labels. Empty watches inspects numbered source only.",
        "parameters": {"type": "object", "properties": {
            "split": {"type": "string", "enum": ["train", "test"]}, "index": {"type": "integer"},
            "inspect_function": {"type": "string"}, "progress_note": progress_note,
            "watches": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {
                "variable": {"type": "string", "description": "Local name, or '$output' for the final grid."},
                "function_name": {"type": "string"}, "line": {"type": "integer", "minimum": 1},
                "path": {"type": "array", "maxItems": 4, "items": {"type": ["integer", "string"]}},
                "occurrence": {"type": ["integer", "string"], "description": "'last' or a zero-based hit index."},
                "expected": {"anyOf": [scalar_schema, {"type": "array", "maxItems": 16, "items": scalar_schema}]}},
                "required": ["variable", "expected"], "additionalProperties": False}}},
            "required": ["split", "index", "watches"] + (["progress_note"] if required_note else [])}}}


async def trace_solver(session, task, arguments, record):
    session.last_tool_failed = False
    preferred = record.get("accepted")
    if preferred is None:
        session.last_tool_failed = True
        return "Submit a preferred candidate before tracing its saved execution."
    spec = {k: v for k, v in arguments.items() if k != "progress_note"}
    try:
        # The checkpoint's XML tool protocol serializes top-level parameters as
        # strings; native JSON callers may already supply typed values.
        for key in ("index", "watches"):
            if isinstance(spec.get(key), str):
                spec[key] = json.loads(spec[key])
        for key in ("split", "inspect_function"):
            if isinstance(spec.get(key), str):
                spec[key] = spec[key].strip()
        validate_spec(spec, task)
    except ValueError as exc:
        session.last_tool_failed = True
        return str(exc)
    entry = record["submissions"][preferred]
    result = await session.trace_saved(entry, spec)
    if result.get("inspection"):
        return ("Saved snapshot source; line watches run BEFORE that line. This inspection is not an audit.\n"
                + json.dumps(result, ensure_ascii=False))
    expected_output = (entry["test"][spec["index"]] if spec["split"] == "test"
                       else task["train"][spec["index"]]["output"])
    snapshot_matches = result.get("output") == expected_output
    old_identity, new_identity = entry.get("identity") or {}, result.get("identity") or {}
    # A previously known identity becoming unknown is also not a verified match.
    identity_mismatch = bool(old_identity.get("digest") and old_identity != new_identity)
    kinds = {"output" if w["variable"] == "$output" else "local" for w in spec["watches"]}
    complete = bool(result.get("complete") and snapshot_matches and not identity_mismatch
                    and kinds == {"local", "output"})
    event = dict(turn=record["turns"], preferred_submission=preferred, spec=spec,
        completion_tokens=record.get("completion_tokens", 0), **result)
    event.update(snapshot_matches=snapshot_matches, identity_mismatch=identity_mismatch,
                 complete=complete, test_answers_used=False,
                 interpretation="Model-predicted effects, NOT test correctness or verified rule invariants.")
    record.setdefault("execution_audits", []).append(event)
    matched = complete and result.get("expectations_match")
    if matched and spec["split"] == "test":
        record["execution_audit_completed_for"] = preferred
    elif not matched:
        record["execution_audit_completed_for"] = None
    session.last_tool_failed = not complete
    summary = {k: v for k, v in event.items() if k not in ("output", "identity", "spec")}
    text = json.dumps(summary, ensure_ascii=False)
    if len(text) > 18000:
        text = text[:18000] + "\n[report truncated; use fewer/smaller watches]"
        event["report_truncated"] = True
        event["complete"] = complete = False
        record["execution_audit_completed_for"] = None
        session.last_tool_failed = True
    # A supplementary, matching training trace may preserve an existing test
    # check, but cannot replace it. Failed/unknown observations still invalidate
    # the current execution check, as they did before.
    success_request = (MECHANISM_REQUEST if record.get("execution_audit_completed_for") == preferred else
                       "The training-input execution matched, but the required test-input check is still pending.\n" + REQUEST)
    return ("Actual saved-solver execution (no test answers):\n" + text +
            "\n" + (success_request if complete and result.get("expectations_match") else
            "An expectation was contradicted or the check is incomplete. Inspect the actual stages; "
            "unreached/missing values are UNKNOWN. Include a local watch and an output watch. "
            "Do not confuse raw input with mutated state. Saved answer and budgets remain unchanged."))
