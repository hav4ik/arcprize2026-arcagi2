"""Summarize every DeepSeek eval run in evals/results/ as one table (reads result files only).

For each *.jsonl: number of records, errors, accuracy, mean output tokens per call, total output tokens,
and estimated cost at off-peak list prices (Flash $0.60/M, Pro $1.98/M output tokens; input ignored).

    .venv/bin/python evals/report.py            # all runs
    .venv/bin/python evals/report.py arcprize   # runs whose file name contains "arcprize"
"""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT_PRICE = {"deepseek-flash": 0.60, "deepseek-v4-pro": 1.98}


def load(path):
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


def out_tokens(d):
    u = d.get("usage") or {}
    return u.get("completion_tokens", u.get("completion", 0))


def summarize(path):
    recs = load(path)
    ok = [d for d in recs if "error" not in d]
    if not ok:
        return None
    toks = [out_tokens(d) for d in ok]
    model = ok[0].get("model", "?")
    correct = sum(bool(d.get("correct")) for d in ok)
    return {
        "run": path.stem, "model": model, "effort": ok[0].get("effort", "?"),
        "calls": len(ok), "errors": len(recs) - len(ok),
        "acc": correct / len(ok), "unparsed": sum(d.get("pred") is None for d in ok),
        "tok_mean": sum(toks) / len(toks), "tok_total": sum(toks),
        "cost": sum(toks) * OUTPUT_PRICE.get(model, 0) / 1e6,
    }


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = [s for p in sorted(RESULTS.glob("*.jsonl")) if pattern in p.name and (s := summarize(p))]
    print(f"{'run':45s} {'effort':6s} {'calls':>5s} {'err':>4s} {'acc/call':>8s} {'unpars':>6s} "
          f"{'tok/call':>8s} {'tok total':>10s} {'~$':>6s}")
    for r in rows:
        print(f"{r['run']:45s} {r['effort']:6s} {r['calls']:5d} {r['errors']:4d} {r['acc']:8.1%} {r['unparsed']:6d} "
              f"{r['tok_mean'] / 1e3:7.1f}k {r['tok_total'] / 1e6:9.2f}M {r['cost']:6.2f}")
    print("\nacc/call = fraction of individual calls (attempts) that were correct; it is not pass@2.")


if __name__ == "__main__":
    main()
