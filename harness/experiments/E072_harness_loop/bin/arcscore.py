"""E072 scoring: per-task results from a run's attempts.jsonl, Kaggle-style. Per (task, pair): the final record of each attempt
(multi-test records split per pair; an errored final record without a grid falls back to the latest earlier errored record with
one), ranked train_pass first then attempt order, top-2 distinct grids = attempt_1/attempt_2. Task score = mean over pairs of
(either answer correct); Kaggle score = 100 x mean over tasks. Unanswered pairs count wrong."""
import collections, json


def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


def split(rec):
    n = rec.get("multi_test")
    if n:
        preds = rec["pred"] if isinstance(rec.get("pred"), list) and len(rec["pred"]) == n else [None] * n
        return [(q, preds[q]) for q in range(n)]
    return [(rec.get("pair", 0), rec.get("pred"))]


def load_jsonl(path):
    out = []
    try:
        for l in open(path, errors="replace"):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def task_results(records, solutions):
    hist = collections.defaultdict(list)
    for r in records:
        if "task_id" in r:
            hist[(r["task_id"], r.get("pair", 0), r["attempt"])].append(r)
    per = collections.defaultdict(list)  # (task, q) -> [{pred, train_pass, attempt}]
    for (t, p, a), rs in hist.items():
        pp = collections.defaultdict(list)
        for r in rs:
            for q, g in split(r):
                pp[q].append((r, g))
        for q, seq in pp.items():
            r, g = seq[-1]
            if not valid_grid(g):
                prov = [(x, y) for x, y in seq[:-1] if x.get("error") and valid_grid(y)]
                if prov:
                    r, g = prov[-1]
            if valid_grid(g):
                per[(t, q)].append({"pred": g, "train_pass": r.get("train_pass"), "attempt": a})
    res = {}
    for t, sol in solutions.items():
        pairs = []
        for q, s in enumerate(sol):
            ans = per.get((t, q), [])
            order = sorted(ans, key=lambda r: (r["train_pass"] is not True, r["attempt"]))
            top = []
            for r in order:
                if r["pred"] not in top:
                    top.append(r["pred"])
            top = top[:2]
            pairs.append({"p1": bool(top) and top[0] == s, "p2": s in top,
                          "att": {str(r["attempt"]): (r["pred"] == s) for r in ans},
                          "train_pass": {str(r["attempt"]): r["train_pass"] for r in ans}})
        res[t] = pairs
    return res


def summarize(res, only=None):
    ts = [t for t in res if only is None or t in only]
    if not ts:
        return {}
    f = lambda t, k: sum(p[k] for p in res[t]) / len(res[t])
    a0 = [p["att"]["0"] for t in ts for p in res[t] if "0" in p["att"]]
    a1 = [p["att"]["1"] for t in ts for p in res[t] if "1" in p["att"]]
    both = sum(1 for t in ts if all("0" in p["att"] and "1" in p["att"] for p in res[t]))
    return {"tasks": len(ts), "score_pass2": round(100 * sum(f(t, "p2") for t in ts) / len(ts), 2),
            "score_pass1": round(100 * sum(f(t, "p1") for t in ts) / len(ts), 2),
            "tasks_with_both_attempts": both,
            "attempt0_acc": round(100 * sum(a0) / max(1, len(a0)), 1), "attempt1_acc": round(100 * sum(a1) / max(1, len(a1)), 1),
            "n_a0": len(a0), "n_a1": len(a1)}
