"""Label-blind budgets and answer preservation for a queued REPL -> raw CoT pass.

CoT outputs are NOT training-verified solver submissions. Keep their provenance
separate and never feed them through the REPL accepted-program selector.
"""
from copy import deepcopy


def allocate(context, test_count):
    """Independent ceilings before prompt counting, NOT shares of a task/TIR budget."""
    return [context - 16] * test_count


def combine(first, requests, *, allowances, context, elapsed_s):
    """Preserve REPL top-1; put a distinct valid CoT grid in slot 2, or fill an empty slot."""
    result = deepcopy(first)
    top, alt = result["final_grids"], result["alternative_grids"]
    sources = [["repl" if g is not None else None, "repl" if a is not None else None]
               for g, a in zip(top, alt)]
    usage = result["usage"]
    cot_usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    complete = True
    for index, request in requests.items():
        complete &= bool(request.get("usage_complete", bool(request.get("usage"))))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            tokens = request.get("usage", {}).get(name, 0)
            cot_usage[name] += tokens
            usage[name] = usage.get(name, 0) + tokens
        grid = request.get("final_grid")
        if grid is None:
            continue
        if top[index] is None:
            top[index], sources[index][0] = deepcopy(grid), "cot"
        elif grid != top[index]:
            alt[index], sources[index][1] = deepcopy(grid), "cot"
        # An agreeing CoT never removes a distinct REPL runner-up.
    for index, pair in enumerate(sources):
        if pair[0] == "cot":
            result["solver_paths"][index] = None  # CoT did not verify or execute a solver.
    completion = usage.get("completion_tokens", 0)
    result.update(
        prediction_available=any(g is not None for g in top), usage_complete=complete,
        streamed_tokens=completion, elapsed_s=first["elapsed_s"] + elapsed_s,
        # The copied task_budget_remaining still describes TIR only. Total usage
        # may exceed that cap; CoT has its own per-request context allowance.
        task_token_budget_scope="repl-only", repl_usage=deepcopy(first["usage"]), cot_usage=cot_usage,
        second_attempt=dict(mode="cot", context=context, allowances=allowances,
                            budget_scope="independent-per-request-context",
                            completed_requests=len(requests), planned_requests=len(allowances),
                            prediction_sources=sources,
                            # Full content/reasoning lives in outputs/<task>_<i>/cot_1/.
                            requests={str(i): {k: r.get(k) for k in (
                                "final_grid", "error", "cancelled", "finish_reason", "usage", "usage_complete",
                                "elapsed_s", "prompt_tokens", "max_tokens", "context", "output")}
                                for i, r in sorted(requests.items())}))
    rate = first.get("decode_tokens_per_s_estimate")
    seconds = first["usage"]["completion_tokens"] / rate if rate else 0
    for request in requests.values():
        rate = request.get("decode_tokens_per_s_estimate")
        if rate:
            seconds += request.get("usage", {}).get("completion_tokens", 0) / rate
    result["decode_tokens_per_s_estimate"] = completion / seconds if seconds > 0 else None
    return result
