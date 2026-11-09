"""Confidence routing helpers for staged ARC model cascades."""

from dataclasses import dataclass
import math

from arc_decoder import getter_kgmon, hashable


CASCADE_MERGE_POLICIES = (
    "one-each",
    "prefer-8b",
    "prefer-4b",
)


@dataclass(frozen=True)
class OutputRoutingConfidence:
    """Confidence statistics for one ARC test output."""

    base_key: str
    kgmon: float
    beam_score: float


@dataclass(frozen=True)
class TaskRoutingDecision:
    """Whole-task routing decision derived from all of its test outputs."""

    task_key: str
    route: bool
    reasons: tuple[str, ...]
    missing_outputs: tuple[str, ...]
    min_kgmon: float | None
    max_beam_score: float | None


@dataclass(frozen=True)
class OutputCascadeMerge:
    """Audit record for one routed test-output merge."""

    base_key: str
    primary_candidates: int
    secondary_candidates: int
    selected_sources: tuple[str, ...]


def output_routing_confidence(base_key, decoded_samples):
    """Return top-KGMon confidence for one output, or ``None`` if missing.

    Identical solution grids are grouped exactly as they are for KGMon
    selection.  The beam statistic is the lowest cumulative NLL among the
    occurrences of the top-ranked grid, matching the evaluation analysis in
    ``docs/routing_4b28b.md``.
    """
    samples = list(decoded_samples)
    if not samples:
        return None

    grouped = {}
    for sample in samples:
        grouped.setdefault(hashable(sample["solution"]), []).append(sample)

    candidates = []
    for guesses in grouped.values():
        kgmon = float(getter_kgmon(guesses))
        beam_score = min(float(guess["beam_score"]) for guess in guesses)
        # Non-finite confidence cannot be trusted.  Treat it as maximally
        # uncertain so a private rerun does not silently retain the output.
        if not math.isfinite(kgmon):
            kgmon = -math.inf
        if not math.isfinite(beam_score):
            beam_score = math.inf
        candidates.append((kgmon, beam_score))

    top_kgmon = max(candidate[0] for candidate in candidates)
    tied_top_beams = [
        beam_score
        for kgmon, beam_score in candidates
        if kgmon == top_kgmon
    ]
    # Exact KGMon ties are rare.  Taking the highest tied beam is conservative
    # for routing and avoids depending on filesystem iteration order.
    return OutputRoutingConfidence(
        base_key=base_key,
        kgmon=top_kgmon,
        beam_score=max(tied_top_beams),
    )


def route_tasks_by_confidence(
    queries,
    decoded_results,
    task_keys,
    *,
    kgmon_threshold=-5.0,
    beam_threshold=0.7,
):
    """Route uncertain ARC tasks using only inference artifacts.

    A whole task is routed when any expected output is missing, its least
    confident output has KGMon at or below ``kgmon_threshold``, or its largest
    top-candidate beam NLL is at or above ``beam_threshold``.
    """
    kgmon_threshold = float(kgmon_threshold)
    beam_threshold = float(beam_threshold)
    if not math.isfinite(kgmon_threshold):
        raise ValueError("kgmon_threshold must be finite")
    if not math.isfinite(beam_threshold) or beam_threshold < 0:
        raise ValueError("beam_threshold must be finite and non-negative")

    decisions = []
    for task_key in task_keys:
        confidences = []
        missing_outputs = []
        for test_index in range(len(queries[task_key]["test"])):
            base_key = f"{task_key}_{test_index}"
            confidence = output_routing_confidence(
                base_key,
                decoded_results.get(base_key, {}).values(),
            )
            if confidence is None:
                missing_outputs.append(base_key)
            else:
                confidences.append(confidence)

        min_kgmon = (
            min(confidence.kgmon for confidence in confidences)
            if confidences
            else None
        )
        max_beam_score = (
            max(confidence.beam_score for confidence in confidences)
            if confidences
            else None
        )
        reasons = []
        if missing_outputs:
            reasons.append("missing-output")
        if min_kgmon is not None and min_kgmon <= kgmon_threshold:
            reasons.append("low-kgmon")
        if (
            max_beam_score is not None
            and max_beam_score >= beam_threshold
        ):
            reasons.append("high-beam")
        decisions.append(
            TaskRoutingDecision(
                task_key=task_key,
                route=bool(reasons),
                reasons=tuple(reasons),
                missing_outputs=tuple(missing_outputs),
                min_kgmon=min_kgmon,
                max_beam_score=max_beam_score,
            )
        )
    return decisions


def zero_routed_submission_tasks(submission, decisions):
    """Replace both attempts for every routed task with ``[[0]]``."""
    routed_keys = []
    for decision in decisions:
        if not decision.route:
            continue
        routed_keys.append(decision.task_key)
        for output in submission[decision.task_key]:
            output["attempt_1"] = [[0]]
            output["attempt_2"] = [[0]]
    return routed_keys


def merge_routed_predictions(
    queries,
    primary_predictions,
    secondary_predictions,
    decisions,
    *,
    policy="one-each",
    n_guesses=2,
):
    """Merge independently ranked 4B and 8B predictions for routed tasks.

    Unrouted tasks retain the primary predictions exactly.  Routed outputs are
    deduplicated by grid.  ``one-each`` gives the top 8B and top 4B grids first,
    then fills any remaining slot from each model's lower-ranked candidates.
    The two prefer-* policies keep both candidates from one model when
    available while retaining the other model as a missing-output fallback.
    """
    if policy not in CASCADE_MERGE_POLICIES:
        supported = ", ".join(CASCADE_MERGE_POLICIES)
        raise ValueError(
            f"Unknown cascade merge policy {policy!r}; expected {supported}"
        )
    n_guesses = int(n_guesses)
    if n_guesses < 1:
        raise ValueError("n_guesses must be positive")

    merged = {
        base_key: list(guesses)
        for base_key, guesses in primary_predictions.items()
    }
    routed_tasks = [
        decision.task_key for decision in decisions if decision.route
    ]
    audit_records = []

    for task_key in routed_tasks:
        for test_index in range(len(queries[task_key]["test"])):
            base_key = f"{task_key}_{test_index}"
            primary = list(primary_predictions.get(base_key, ()))
            secondary = list(secondary_predictions.get(base_key, ()))

            if policy == "one-each":
                ranked_groups = (
                    ("8b", secondary[:1]),
                    ("4b", primary[:1]),
                    ("8b", secondary[1:]),
                    ("4b", primary[1:]),
                )
            elif policy == "prefer-8b":
                ranked_groups = (("8b", secondary), ("4b", primary))
            else:
                ranked_groups = (("4b", primary), ("8b", secondary))

            selected = []
            selected_sources = []
            seen = set()
            for source, guesses in ranked_groups:
                for guess in guesses:
                    solution_key = hashable(guess)
                    if solution_key in seen:
                        continue
                    seen.add(solution_key)
                    selected.append(guess)
                    selected_sources.append(source)
                    if len(selected) >= n_guesses:
                        break
                if len(selected) >= n_guesses:
                    break

            if selected:
                merged[base_key] = selected
            else:
                merged.pop(base_key, None)
            audit_records.append(
                OutputCascadeMerge(
                    base_key=base_key,
                    primary_candidates=len(primary),
                    secondary_candidates=len(secondary),
                    selected_sources=tuple(selected_sources),
                )
            )

    return merged, audit_records


def print_cascade_merge_summary(decisions, audit_records, *, policy):
    """Print compact routing and fallback coverage for a real cascade."""
    routed = [decision for decision in decisions if decision.route]
    outputs_with_8b = sum(
        record.secondary_candidates > 0 for record in audit_records
    )
    outputs_with_4b = sum(
        record.primary_candidates > 0 for record in audit_records
    )
    source_patterns = {}
    for record in audit_records:
        pattern = "+".join(record.selected_sources) or "missing"
        source_patterns[pattern] = source_patterns.get(pattern, 0) + 1
    patterns = ", ".join(
        f"{pattern}={count}"
        for pattern, count in sorted(source_patterns.items())
    )
    print("*** Real 4B -> 8B cascade merge")
    print(
        f" Merge policy: {policy}; routed tasks: {len(routed)}/"
        f"{len(decisions)}; routed outputs: {len(audit_records)}"
    )
    print(
        f" Routed outputs with candidates: 4B={outputs_with_4b}, "
        f"8B={outputs_with_8b}; selected source patterns: "
        f"{patterns or 'none'}"
    )
    return {
        "routed_tasks": len(routed),
        "routed_outputs": len(audit_records),
        "outputs_with_4b": outputs_with_4b,
        "outputs_with_8b": outputs_with_8b,
        "source_patterns": source_patterns,
    }


def print_routing_simulation_summary(
    decisions,
    *,
    kgmon_threshold,
    beam_threshold,
):
    """Print an auditable summary of the zero-output routing simulation."""
    routed = [decision for decision in decisions if decision.route]
    reason_counts = {
        reason: sum(reason in decision.reasons for decision in routed)
        for reason in ("missing-output", "low-kgmon", "high-beam")
    }
    print("*** Simulating 4B -> 8B routing by zeroing routed tasks")
    print(
        " Routing rule: any missing output OR minimum task-output "
        f"KGMon <= {float(kgmon_threshold):g} OR maximum task-output "
        f"beam_score >= {float(beam_threshold):g}"
    )
    print(
        f" Routed tasks: {len(routed)}/{len(decisions)}; "
        f"missing-output={reason_counts['missing-output']}, "
        f"low-kgmon={reason_counts['low-kgmon']}, "
        f"high-beam={reason_counts['high-beam']} "
        "(reason counts overlap)"
    )
    for decision in routed:
        kgmon = (
            "n/a"
            if decision.min_kgmon is None
            else f"{decision.min_kgmon:.6f}"
        )
        beam = (
            "n/a"
            if decision.max_beam_score is None
            else f"{decision.max_beam_score:.6f}"
        )
        print(
            f"  {decision.task_key}: reasons={','.join(decision.reasons)}; "
            f"min_kgmon={kgmon}; max_beam={beam}; "
            f"missing={list(decision.missing_outputs)}"
        )
    return len(routed)
