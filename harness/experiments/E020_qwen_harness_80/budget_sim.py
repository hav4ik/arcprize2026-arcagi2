"""E020: Kaggle-budget policies, simulated offline from logged attempts (no tokens).

For a first-attempt file A (1 attempt per pair) and optionally a second-attempt source B (another sample of a config,
e.g. another arm's attempt 0 or an E012 attempt 1):
- `single`: attempt_1 = A's answer, attempt_2 = none (the test-input copy).
- `free2`: attempt_2 = A's best alternative (record_alternatives: highest train fraction, then first listed).
- `gate`: run B only on pairs where A did not pass the train check (no train_pass, or no answer); attempt_2 =
  B's answer if B ran, else A's free alternative. B's answer counts only if B finished within `--b-cap` tokens
  (simulating a cap by truncation: a longer B attempt is scored as no answer at min(tokens, cap)).
- `always`: B on every pair (plain pass@2 of two attempts).
Reports Kaggle pair-weighted pass@1 / pass@2, task-weighted pass@2, tokens per pair mean / p90, gated fraction,
and the paired pass@2 CI vs `single`.

    .venv/bin/python experiments/E020_qwen_harness_80/budget_sim.py --a evals/results/qwen_tools_e020_K0_all.jsonl \
        --b evals/results/qwen_tools_e020_K1_all.jsonl [--b-cap 48000] [--a-attempt 0 --b-attempt 0]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_sim import grid_key, load_truth, paired_ci, tokens  # noqa: E402


def load(path, attempt):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if int(r["attempt"]) == attempt and not ("error" in r and r.get("pred") is None):
                out[(r["task_id"], int(r["pair"]))] = r
    return out


def best_alt(r):
    alts = [a for a in r.get("alternatives") or [] if "pred" in a]
    if not alts:
        return None
    return sorted(alts, key=lambda a: -(a.get("train_frac") or 0))[0]["pred"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--a-attempt", type=int, default=0)
    ap.add_argument("--b-attempt", type=int, default=0)
    ap.add_argument("--b-cap", type=int, default=0, help="simulated hard cap for B (0 = as logged)")
    args = ap.parse_args()
    A = load(args.a, args.a_attempt)
    B = load(args.b, args.b_attempt) if args.b else {}
    keys = sorted(set(A) & set(B)) if B else sorted(A)
    truth = load_truth(keys)
    rows = {}
    for pol in ["single", "free2"] + (["gate", "always"] if B else []):
        p1, p2, toks, gated = {}, {}, [], 0
        for k in keys:
            a, ans = A[k], truth[k]
            first = grid_key(a.get("pred"))
            second = None
            tok = tokens(a)
            if pol in ("free2", "gate", "always"):
                second = grid_key(best_alt(a))
            if pol in ("gate", "always") and (pol == "always" or not a.get("train_pass")):
                b = B[k]
                tb = tokens(b)
                ok = not args.b_cap or tb <= args.b_cap
                tok += min(tb, args.b_cap) if args.b_cap else tb
                gated += 1
                bp = grid_key(b.get("pred")) if ok else None
                if first is None:
                    first, second = bp, second
                elif bp is not None and bp != first:
                    second = bp
            p1[k] = float(first is not None and first == ans)
            p2[k] = float(ans is not None and ans in (first, second))
            toks.append(tok)
        tw = {}
        for (t, _), v in p2.items():
            tw.setdefault(t, []).append(v)
        rows[pol] = (p1, p2, toks, gated, statistics.mean(statistics.mean(v) for v in tw.values()))
    print(f"{len(keys)} pairs; A={Path(args.a).name}" + (f", B={Path(args.b).name}" if args.b else "")
          + (f", B capped at {args.b_cap}" if args.b_cap else ""))
    print("\n| policy | pass@1 pair-w | pass@2 pair-w | pass@2 task-w | tokens/pair mean / p90 | 2nd attempts | "
          "Δ pass@2 vs single [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    for pol, (p1, p2, toks, gated, t2) in rows.items():
        d, lo, hi = paired_ci(rows["single"][1], p2)
        q = sorted(toks)[int(0.9 * (len(toks) - 1))]
        print(f"| {pol} | {statistics.mean(p1.values()):.1%} | {statistics.mean(p2.values()):.1%} | {t2:.1%} | "
              f"{statistics.mean(toks) / 1000:.1f}k / {q / 1000:.1f}k | {gated / len(keys):.0%} | "
              f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")


if __name__ == "__main__":
    main()
