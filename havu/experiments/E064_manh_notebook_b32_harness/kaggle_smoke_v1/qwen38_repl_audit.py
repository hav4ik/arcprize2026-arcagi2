"""Label-blind train/test input measurements for an opt-in REPL experiment.

Novelty is not an error, and a model-authored measurement is not an independently
proved invariant. Test outputs never enter this module or the sandbox request.
"""
import json
import math


GUIDANCE = """

# Executable train-to-test assumption audit
After saving a training-passing candidate, identify the most consequential INPUT
condition its rule relies on, and measure that condition on ALL train and test
inputs using audit_inputs. This is not another full-grid dump or a comparison
against an expected test grid you invented from your own rule.

Define a named measure(grid) function in Python. Return a dict of 1..8 concise
feature names to scalars (bool, finite number, short string, None for unknown),
or lists of at most 16 such values. The harness calls the SAME function on every
original input and reports per-input values, unseen test values and range changes.
Use solver helpers to measure rule-relevant conditions, not just grid dimensions.
Explain which training observations support the assumption. Useful questions:
- Does a numerical threshold encode an absolute count or a relation to available
  neighbors? Measure relevant boundary/degree cases, not only interior examples.
- Does a detector cover the objects the rule needs? Are there unhandled objects,
  ambiguous matches, ties, changed dependency structures or unreachable targets?
- Does a lookup/fallback handle unseen cases by a demonstrated relation, or merely
  default? Measure coverage and unresolved cases on the actual test inputs too.
- Does a search find a UNIQUE distinct prediction? Duplicate search paths do not
  count as different predictions; a capped search is not a uniqueness proof.
These are possible questions, NOT universal ARC invariances or rules to impose.
New sizes, colors, counts and structures may be intentional. Audit only conditions
your chosen rule depends on. A measured difference is a reason to investigate,
not permission to patch individual test cells. Matching values do not prove the
rule, and self-consistency cannot establish an unknown test answer.

If a consequential ambiguity survives, define a separately named alternative and
use compare_rules: it checks ALL training pairs and saves distinct test predictions
for attempt_2 without replacing the preferred candidate. In this experiment,
check(alt) also triggers an independent validation/snapshot of up to two named,
training-passing alternatives per cell AFTER a preferred candidate exists. Keep
the preferred function intact. Do not manufacture alternatives to fill slots.
Use concise reasoning after acceptance. All auditing uses the EXISTING task budget.
"""

REQUEST = (
    "[executable input audit] The candidate is saved, not verified on test. Define "
    "measure(grid) for the most consequential condition your solver assumes, then call "
    "audit_inputs(measure_name=..., assumption=..., training_evidence=...). The harness "
    "will measure EVERY train and test input with the same code. Use numerical/input "
    "evidence, not an expected test grid computed from your own hypothesis. Investigate "
    "novel or unresolved cases; differences are not automatically errors. If a meaningful "
    "alternative remains, use compare_rules to check and save it without replacing the "
    "preferred candidate. Do not restart broad exploration or force a second answer."
)


def normalize_measurement(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 8:
        raise ValueError("measure(grid) must return a dict with 1..8 feature names")

    def scalar(x):
        # NumPy scalars are converted by the runtime before this validator.
        if x is None or type(x) is bool:
            return x
        if type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1e12:
            return x
        if type(x) is str and len(x) <= 120:
            return x
        raise ValueError("feature values must be finite scalars, short strings, or None")

    result = {}
    for key, item in value.items():
        if type(key) is not str or not key.strip() or len(key) > 80:
            raise ValueError("feature names must be nonempty strings of at most 80 characters")
        if isinstance(item, (list, tuple)):
            if len(item) > 16:
                raise ValueError("feature lists may have at most 16 values; return compact measurements")
            result[key] = [scalar(x) for x in item]
        else:
            result[key] = scalar(item)
    return result


def comparison_report(result):
    """Compare observed feature domains; never score a rule or assume invariance."""
    rows = result["measurements"]
    fields = sorted({k for row in rows for k in (row.get("values") or {})})
    changes = []
    key = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    values = lambda item: item if isinstance(item, list) else [item]
    for name in fields:
        train = [x for row in rows if row["split"] == "train" and name in (row.get("values") or {})
                 for x in values(row["values"][name])]
        known = {key(x) for x in train if x is not None}
        numeric = [x for x in train if type(x) in (int, float)]
        for row in rows:
            if row["split"] != "test" or row.get("error"):
                continue
            vals = values((row.get("values") or {}).get(name))
            unseen = list(dict.fromkeys(key(x) for x in vals if x is not None and key(x) not in known))
            unknown = any(x is None for x in vals) or not known
            outside = [x for x in vals if type(x) in (int, float) and numeric
                       and not min(numeric) <= x <= max(numeric)]
            if unseen or unknown:
                changes.append(dict(test_index=row["index"], feature=name,
                    unseen_values=[json.loads(x) for x in unseen[:8]], unknown=unknown,
                    train_numeric_range=[min(numeric), max(numeric)] if numeric else None,
                    outside_train_range=outside[:8]))
    schemas = [set(row["values"]) for row in rows if row.get("values") is not None]
    consistent_schema = bool(schemas) and all(s == schemas[0] for s in schemas)
    return dict(measurements=rows, differences=changes, consistent_schema=consistent_schema,
                complete=bool(rows) and consistent_schema and all(not r.get("error") for r in rows),
                test_answers_used=False, interpretation=(
                    "Model-authored measurements, not verified invariants. Novel values are not errors; "
                    "equal values do not prove generalization. None means unknown, not pass."))


async def audit_inputs(session, task, arguments, record):
    session.last_tool_failed = False
    preferred = record.get("accepted")
    if preferred is None:
        session.last_tool_failed = True
        return "Save your preferred function with submit before auditing its input assumptions."
    name = arguments.get("measure_name")
    if not isinstance(name, str) or not name.isidentifier():
        session.last_tool_failed = True
        return "measure_name must name a function already defined in Python."
    for field in ("assumption", "training_evidence"):
        text = arguments.get(field)
        if not isinstance(text, str) or not 12 <= len(text.strip()) <= 3000:
            session.last_tool_failed = True
            return f"{field} needs 12..3000 characters explaining the actual rule condition and evidence."
    result = await session.measure_inputs(name)
    if result.get("error"):
        session.last_tool_failed = True
        return "Input audit failed; saved candidate unchanged: " + result["error"]
    rows = result.get("measurements", [])
    expected = [(split, i) for split in ("train", "test") for i in range(len(task[split]))]
    if [(r.get("split"), r.get("index")) for r in rows] != expected:
        session.last_tool_failed = True
        return "Input audit incomplete: not every original train/test input was measured."
    report = comparison_report(result)
    event = dict(turn=record["turns"], completion_tokens=record["completion_tokens"],
                 preferred_submission=preferred, measure_name=name,
                 assumption=arguments["assumption"], training_evidence=arguments["training_evidence"],
                 evidence_source="model-stated, not independently certified", **report)
    record.setdefault("input_audits", []).append(event)
    if report["complete"]:
        record["input_audit_completed_for"] = preferred
    session.last_tool_failed = not report["complete"]
    # All rows remain in the raw record, even when a pathological report is too long for a turn.
    text = json.dumps(report, ensure_ascii=False, allow_nan=False)
    if len(text) > 18000:
        text = text[:18000] + "\n[report truncated; use fewer/shorter features and audit again]"
        record["input_audit_completed_for"] = None
    return ("Executed on every original training and test INPUT (no test answers).\n" + text +
            "\nDo these measurements support the rule on unseen cases? If not, test a separately named "
            "alternative with compare_rules. No difference or a matching prediction is not proof. "
            "The saved preferred candidate remains unchanged.")


def capture_checked_alternatives(task, record, evaluations, journal_length):
    """Independently validated snapshots only; never replace the preferred index."""
    preferred = record.get("accepted")
    if preferred is None or not record.get("candidate_pool_enabled"):
        return ""
    seen = {json.dumps(s["test"]) for s in record["submissions"] if s.get("accepted")}
    saved = []
    for result in evaluations:
        train, test = result.get("train", []), result.get("test", [])
        if (result.get("error") or len(train) != len(task["train"]) or len(test) != len(task["test"])
                or any(c.get("error") or c.get("grid") != p["output"] for c, p in zip(train, task["train"]))
                or any(c.get("error") or c.get("grid") is None for c in test)):
            continue
        predictions = [c["grid"] for c in test]
        identity = json.dumps(predictions)
        if identity in seen:
            continue
        seen.add(identity)
        record["submissions"].append(dict(turn=record["turns"], function=result["function"],
            train_passed=len(task["train"]), test=predictions, source=result.get("source"),
            journal_length=journal_length, accepted=True, preserve_only=True,
            capture="independent-validation-after-check", completion_tokens=record["completion_tokens"]))
        saved.append(result["function"])
    return ("\n[harness] Independently revalidated and saved named alternatives for attempt_2: " +
            ", ".join(saved) + ". Preferred candidate unchanged; test correctness UNKNOWN.") if saved else ""
