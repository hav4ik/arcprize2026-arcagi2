"""Opt-in, label-blind rule coverage/interaction review of EVERY test input.

The harness enforces coverage of the review, not correctness of model-authored
probes. Unresolved checks are legitimate results; no candidate is rejected or
reranked by this module and no search/review budget is extended.
"""
import json


GUIDANCE = """

# Per-input coverage and interaction review
After saving a training-passing function, perform ONE compact audit of its rule,
then explicitly review EVERY test input. Do not just compare sizes/color counts,
predict the same grid twice, or assert that matching the demonstrations proves it.

1. Define measure(grid) using the actual solver helpers/stages where possible.
   Return a dict of at most 8 compact features. Declare 2..4 checks in
   audit_interactions: at least one coverage check (does the rule account for all
   relevant objects/cases?) and one interaction check (can operations invalidate
   each other, compete for cells, change connectivity, or rely on order?). A
   single-stage rule with no such interaction may instead explain why in
   interaction_not_applicable. These are questions, NOT universal ARC invariances.
   Each declared feature must be True, False, or None (unknown). True means the
   stated INPUT/stage condition was measured; not that a test output is correct.
   You may add concise counts/coordinates as evidence. Reuse actual stage state
   rather than a disconnected reimplementation; never compare to a self-invented
   expected test grid. Explain how each condition is supported by training pairs.
2. audit_interactions runs the SAME measurement on ALL original train/test inputs.
   Read its per-input results, including failures, uncertainty and interactions.
   Ask whether the evidence covers the mechanism of the rule, not just its output
   format. For each test input explain the relevant cases and operations, including
   whether a training-supported interpretation handles its novel configurations.
3. In a LATER assistant turn, after reading the measurements, call
   review_interactions with one entry for EVERY test index and a supported
   or unresolved assessment of EVERY declared check. False, unknown, failed, or
   stale measurements cannot be called supported. An unresolved entry is allowed:
   it records uncertainty, not an incorrect test answer and not a veto of the
   saved candidate. State why the ambiguity remains. If a defensible competing
   rule follows from training evidence, use compare_rules to preserve it for
   attempt_2. Do not invent differences or patch individual test cells.

This replaces the mandatory raw-input audit, not adds a second mandatory audit.
Keep the preferred solver/helpers intact while measuring it. A genuine revision
needs a new audit; re-submitting unchanged code does not restart review budgets.
All work uses the EXISTING review/task time and token limits. Stop within them;
saved answers survive incomplete audits. Measurement support is not a proof of
generalization: the model still chooses which conditions to check.
"""

REQUEST = (
    "[per-input interaction review] Candidate saved; test correctness UNKNOWN. "
    "Define measure(grid) for 2..4 rule-dependent checks: coverage of relevant "
    "objects/cases plus interaction of operations (or explain why no interaction "
    "applies). Call audit_interactions, read ALL train/test input measurements, "
    "then review_interactions for EVERY test index. Use actual solver helpers/stage "
    "conditions, not just dimensions or your own predicted grid as ground truth. "
    "Unresolved cases are allowed; keep the saved answer and existing budget."
)


def pending_request(record):
    preferred = record.get("accepted")
    if preferred is None or record.get("interaction_review_completed_for") == preferred:
        return None
    events = record.get("interaction_audits", [])
    if events and events[-1]["preferred_submission"] == preferred and events[-1]["report_complete"]:
        return (f"[per-input interaction review] Read audit_id={events[-1]['audit_id']} and call "
                "review_interactions with EVERY test index and EVERY declared check. Explain "
                "the evidence for each input; mark contradicted/unknown cases unresolved. "
                "Do not manufacture certainty. The saved answer and budget are unchanged.")
    return REQUEST


def tool_schemas(progress_note, require_note):
    def tool(name, description, properties, required):
        return {"type": "function", "function": dict(name=name, description=description,
            parameters=dict(type="object", properties={**properties, "progress_note": progress_note},
                            required=required + (["progress_note"] if require_note else [])))}
    return [tool("audit_interactions",
        "Measure rule coverage and operation interactions on EVERY original train/test INPUT. "
        "Requires a saved preferred solver; never verifies unknown test answers.",
        {"measure_name": {"type": "string", "description": "Previously defined measure(grid) function."},
         "checks": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
             "type": "object", "properties": {
                 "feature": {"type": "string", "description": "Boolean/None feature returned by measure."},
                 "kind": {"type": "string", "enum": ["coverage", "interaction", "assumption"]},
                 "claim": {"type": "string", "description": "Actual condition that True represents."},
                 "training_evidence": {"type": "string", "description": "Training observations supporting the condition."}},
             "required": ["feature", "kind", "claim", "training_evidence"]}},
         "interaction_not_applicable": {"type": "string", "description":
             "Only if no interaction check: explain why this rule has no interacting operations/cases."}},
        ["measure_name", "checks"]),
        tool("review_interactions", "Review the measured evidence for EVERY test input. "
             "Unresolved is allowed; supported requires observed checks, not proof of the answer.",
             {"audit_id": {"type": "integer"}, "tests": {"type": "array", "items": {
                 "type": "object", "properties": {
                     "index": {"type": "integer", "minimum": 0},
                     "checks": {"type": "object", "additionalProperties": {
                         "type": "string", "enum": ["supported", "unresolved"]}},
                     "explanation": {"type": "string", "description":
                         "Input-specific rule evidence, interactions and remaining unsupported cases."}},
                 "required": ["index", "checks", "explanation"]}}}, ["audit_id", "tests"])]


def _structured(value):
    # Native tool arguments are typed; the Qwen XML tool format may encode a
    # top-level list/object/integer as a JSON string. Never eval model text.
    return json.loads(value) if isinstance(value, str) else value


def _text(value, field, minimum=12, maximum=2000):
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ValueError(f"{field} needs {minimum}..{maximum} characters of specific rule evidence")
    return value.strip()


def _checks(arguments):
    checks = _structured(arguments.get("checks"))
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        raise ValueError("checks must list 1..4 conditions, including coverage and interaction")
    result = []
    for c in checks:
        if not isinstance(c, dict) or c.get("kind") not in ("coverage", "interaction", "assumption"):
            raise ValueError("Each check needs kind=coverage, interaction or assumption")
        result.append(dict(feature=_text(c.get("feature"), "feature", 1, 80), kind=c["kind"],
            claim=_text(c.get("claim"), "claim"),
            training_evidence=_text(c.get("training_evidence"), "training_evidence")))
    if len({c["feature"] for c in result}) != len(result):
        raise ValueError("Check feature names must be distinct")
    if not any(c["kind"] == "coverage" for c in result):
        raise ValueError("Declare at least one coverage check")
    reason = arguments.get("interaction_not_applicable")
    if not any(c["kind"] == "interaction" for c in result):
        reason = _text(reason, "interaction_not_applicable", 40)
    elif reason is not None:
        raise ValueError("Use an interaction check OR interaction_not_applicable, not both")
    return result, reason


def _fail(session, text):
    session.last_tool_failed = True
    return "Interaction review incomplete; saved candidate unchanged. " + text


def _status(row, feature):
    value = (row.get("values") or {}).get(feature)
    if row.get("error") or type(value) is not bool:
        return "unknown"
    return "observed" if value else "contradicted"


async def audit_interactions(session, task, arguments, record):
    session.last_tool_failed = False
    preferred = record.get("accepted")
    if preferred is None:
        return _fail(session, "Save a preferred function with submit first.")
    try:
        name = arguments.get("measure_name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("measure_name must name a function already defined in Python")
        checks, reason = _checks(arguments)
    except (ValueError, TypeError) as exc:
        return _fail(session, str(exc))
    # Once a new audit starts, an older review must not conceal its new problems.
    record["interaction_review_completed_for"] = None
    entry = record["submissions"][preferred]
    result = await session.measure_inputs(name, candidate_name=entry["function"])
    rows = result.get("measurements", [])
    expected = [(split, i) for split in ("train", "test") for i in range(len(task[split]))]
    complete = (not result.get("error") and
                [(r.get("split"), r.get("index")) for r in rows] == expected)
    identity = entry.get("identity") or {}
    matches = bool(identity.get("digest") and identity == result.get("candidate_identity"))
    training = {c["feature"]: bool(task["train"]) and complete and all(
        _status(r, c["feature"]) == "observed" for r in rows if r["split"] == "train") for c in checks}
    for row in rows:
        row["checks"] = {c["feature"]: _status(row, c["feature"]) for c in checks}
    audits = record.setdefault("interaction_audits", [])
    event = dict(audit_id=len(audits), turn=record["turns"], completion_tokens=record["completion_tokens"],
        preferred_submission=preferred, measure_name=name, checks=checks,
        interaction_not_applicable=reason, measurements=rows, report_complete=bool(complete),
        snapshot_matches=matches, candidate_identity=result.get("candidate_identity"),
        training_conditions_observed=training, error=result.get("error"),
        test_answers_used=False, journal_length=len(session.journal), review=None)
    audits.append(event)
    # Keep the full event on disk even if a pathological task/probe exceeds the
    # tool-response cap. Do not permit a complete review of an unseen partial report.
    report = {k: event[k] for k in ("audit_id", "checks", "measurements", "report_complete",
        "snapshot_matches", "training_conditions_observed", "error")}
    text = json.dumps(report, ensure_ascii=False, allow_nan=False)
    if len(text) > 24000:
        event["report_complete"] = False
        text = text[:24000] + "\n[report truncated; use fewer/shorter evidence features and audit again]"
    session.last_tool_failed = not event["report_complete"]
    return ("Model-authored probes executed on original INPUTS only; no test answers.\n" + text +
        "\nOnly observed conditions with observed training support and a matching saved solver can be "
        "marked supported. This is not independent verification of the rule. False/None/error, "
        "missing conditions, or an unlinked solver must stay unresolved. " + (pending_request(record) or ""))


def review_interactions(session, task, arguments, record):
    session.last_tool_failed = False
    try:
        audits = record.get("interaction_audits", [])
        audit_id = _structured(arguments.get("audit_id"))
        if not audits or type(audit_id) is not int or audit_id != audits[-1]["audit_id"]:
            raise ValueError("Use the latest audit_id from audit_interactions")
        event = audits[-1]
        if record.get("accepted") is None or event["preferred_submission"] != record["accepted"]:
            raise ValueError("The preferred candidate changed; audit it again")
        if not event["report_complete"]:
            raise ValueError("Not every input has a complete report; audit again within the remaining budget")
        if record["turns"] <= event["turn"]:
            raise ValueError("Read the audit results first; review_interactions must be in a later assistant turn")
        tests = _structured(arguments.get("tests"))
        if (not isinstance(tests, list) or any(not isinstance(t, dict) or type(t.get("index")) is not int for t in tests)
                or sorted(t["index"] for t in tests) != list(range(len(task["test"])))):
            raise ValueError("Review EVERY test index exactly once; no missing or duplicate entries")
        features = {c["feature"] for c in event["checks"]}
        rows = {r["index"]: r for r in event["measurements"] if r["split"] == "test"}
        review = []
        for t in tests:
            checks = t.get("checks")
            if (not isinstance(checks, dict) or set(checks) != features
                    or any(v not in ("supported", "unresolved") for v in checks.values())):
                raise ValueError("Each test must assess EVERY declared check as supported or unresolved")
            explanation = _text(t.get("explanation"), "explanation")
            for feature, status in checks.items():
                if status == "supported" and (not event["snapshot_matches"]
                        or not event["training_conditions_observed"][feature]
                        or rows[t["index"]]["checks"][feature] != "observed"):
                    raise ValueError(f"test {t['index']}/{feature}: evidence is contradicted, unknown, "
                                     "unsupported on training inputs or stale; mark unresolved")
            review.append(dict(index=t["index"], checks=dict(checks), explanation=explanation))
    except (ValueError, TypeError) as exc:
        return _fail(session, str(exc))
    unresolved = [t["index"] for t in review if "unresolved" in t["checks"].values()]
    event["review"] = dict(tests=review, unresolved_test_indices=unresolved,
                           turn=record["turns"], completion_tokens=record["completion_tokens"])
    record["interaction_review_completed_for"] = record["accepted"]
    return (f"Reviewed every test input ({len(review)}); unresolved test indices={unresolved}. "
            "Protocol complete, NOT a proof of test correctness. Candidate unchanged. "
            "If a concrete training-supported alternative explains the uncertainty, compare_rules "
            "can preserve it for attempt_2. Otherwise finish in text; no extra budget was granted.")
