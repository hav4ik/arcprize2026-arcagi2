"""Kaggle answer selection (E021, notebook v3): attempt_1 / attempt_2 for one test pair, and which pairs get a
gated second attempt.

Pure functions over attempt records, shared by kaggle/src/solver.py and offline simulations. An attempt record
needs only `pred`, `train_pass`, `train_frac`, `answer_source`, `tokens` and optionally `alternatives` (the
agent_qwen `record_alternatives` list: other test grids of the same attempt, [{'src', 'pred', 'train_frac'}]).

Ranking (tiers, then order inside a tier):
  0. train-passing grids (the attempt's answer or an alternative with train_frac 1.0);
  1. attempts' own answers that are not known to fail train (e.g. submit_answer of `answer`, no check possible);
  2. the rest (answers known to fail train, alternatives below 1.0, unknown alternatives), by train_frac.
Inside tiers 0 and 1: attempt answers before alternatives, then attempt order, then the alternative's order.
With no train checks and no alternatives this is v2's rule: attempt answers in attempt order.
"""
from .grids import valid_grid

# answer sources that come from a fallback rather than a submission the model chose (E012/E019: ~0-20% correct)
WEAK_SOURCES = {"forced", "session", "salvage", "parse", "lenient", "program_best_fail", "session_solve"}


def candidates(attempts):
    """attempts: {attempt_index: record}. Valid grids only; duplicates are kept (select_two dedups)."""
    out = []
    for a in sorted(attempts):
        r = attempts[a] or {}
        if valid_grid(r.get("pred")):
            out.append({"grid": r["pred"], "attempt": a, "rank": 0, "primary": True, "src": r.get("answer_source"),
                        "train_pass": r.get("train_pass"), "train_frac": r.get("train_frac")})
        for i, alt in enumerate(r.get("alternatives") or []):
            if not isinstance(alt, dict) or not valid_grid(alt.get("pred")):
                continue
            f = alt.get("train_frac")
            out.append({"grid": alt["pred"], "attempt": a, "rank": i + 1, "primary": False,
                        "src": f"alt:{alt.get('src')}", "train_pass": (f == 1.0) if f is not None else None,
                        "train_frac": f})
    return out


def tier(c):
    if c.get("train_pass") is True:
        return 0
    if c["primary"] and c.get("train_pass") is not False:
        return 1
    return 2


def rank_key(c):
    t = tier(c)
    return (t, -(c.get("train_frac") or 0.0) if t == 2 else 0.0, not c["primary"], c["attempt"], c["rank"])


def select_two(cands):
    """The top-2 distinct candidates (0, 1 or 2 of them) in rank order."""
    out = []
    for c in sorted(cands, key=rank_key):
        if all(c["grid"] != o["grid"] for o in out):
            out.append(c)
        if len(out) == 2:
            break
    return out


def retry_category(rec, include_unknown=True):
    """Gated second attempt, by expected value per token (E012 full set, v2 config, 2 attempts per pair: a retry
    of a pair whose first attempt submitted a wrong answer solved 10/66 = 15%; of a submitted answer of unknown
    correctness 10/108 = 9%; after no answer or a forced/fallback answer only 2/59 = 3.4%, at 43k vs 34k tokens):
    0 = the model's own submitted answer fails the train check;
    1 = a submitted answer that cannot be train-checked (only with include_unknown);
    2 = no answer, or only a fallback answer (forced / session / salvage / best failing program ...);
    None = no retry (a train-passing grid exists: the answer or one of its alternatives)."""
    alts = [a for a in rec.get("alternatives") or [] if isinstance(a, dict) and valid_grid(a.get("pred"))]
    if rec.get("train_pass") is True or any(a.get("train_frac") == 1.0 for a in alts):
        return None
    if not valid_grid(rec.get("pred")) or rec.get("answer_source") in WEAK_SOURCES:
        return 2
    if rec.get("train_pass") is False:
        return 0
    return 1 if include_unknown else None


def retry_priority(rec, include_unknown=True):
    """Sort key for the gated second attempt (smaller first), or None if the pair should not be retried:
    retry_category first, then higher train fraction (of the answer or an alternative), then fewer tokens spent in
    the first attempt (a cheaper attempt finishes within the leftover time more often)."""
    cat = retry_category(rec, include_unknown)
    if cat is None:
        return None
    fracs = [rec.get("train_frac") or 0.0] + [a.get("train_frac") or 0.0 for a in rec.get("alternatives") or []
                                               if isinstance(a, dict)]
    return (cat, -max(fracs), rec.get("tokens") or 0)

