"""Compare tool-use runs with the E001d no-tool baseline on the same test pairs.

For each tools_persistent_*.jsonl (optionally filtered by a substring): completed attempts, per-attempt accuracy,
mean/median output tokens, accepted submissions and their precision, and the E001d attempt-0 accuracy and
tokens on exactly the same pairs.

    .venv/bin/python evals/compare_runs.py [substring]
"""
import ast
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
BASE = RESULTS / "arcprize_deepseek-flash_max.jsonl"


def load(path):
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def tokens(rec):
    usage = rec.get("usage") or {}
    if isinstance(usage, str):
        usage = ast.literal_eval(usage)
    return usage.get("completion", usage.get("completion_tokens", 0)) or 0


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    base = {(r["task_id"], int(r["pair"])): r for r in load(BASE) if str(r["attempt"]) == "0"}
    print(f"{'run':58s} {'n':>3s} {'err':>3s} {'acc':>6s} {'tok':>7s} {'med':>7s} {'acc#':>4s} {'prec':>5s} "
          f"| {'E001d acc':>9s} {'tok':>6s}")
    for path in sorted(RESULTS.glob("tools_persistent_*.jsonl")):
        if pattern not in path.name:
            continue
        latest = {}
        for r in load(path):
            latest[(r["task_id"], int(r["pair"]), r["attempt"])] = r
        ok = [r for r in latest.values() if "error" not in r]
        if not ok:
            continue
        tok = [tokens(r) for r in ok]
        acc = [r for r in ok if r.get("stop_reason") == "submitted"]
        keys = [(r["task_id"], int(r["pair"])) for r in ok]
        b = [base[k] for k in keys if k in base]
        b_acc = sum(r.get("correct") is True for r in b) / max(len(b), 1)
        b_tok = statistics.mean(tokens(r) for r in b) if b else 0
        print(f"{path.stem[len('tools_persistent_'):]:58s} {len(ok):3d} {len(latest) - len(ok):3d} "
              f"{sum(bool(r.get('correct')) for r in ok) / len(ok):6.1%} {statistics.mean(tok) / 1e3:6.1f}k "
              f"{statistics.median(tok) / 1e3:6.1f}k {len(acc):4d} "
              f"{(sum(bool(r.get('correct')) for r in acc) / len(acc)) if acc else 0:5.0%} "
              f"| {b_acc:9.1%} {b_tok / 1e3:5.1f}k")
    print("\nacc = per attempt; acc# = accepted submissions (code mode); prec = accuracy of accepted submissions.")
    print("E001d columns: attempt 0 of the no-tool baseline on the same pairs (E001d attempt-0 errors count as wrong).")


if __name__ == "__main__":
    main()
