"""E045: score sampled programs. A sample counts only if its program reproduces EVERY training pair.

Per test input, the passing samples' outputs are ranked by votes (number of passing samples that produce
them); the top two distinct outputs are attempt_1 / attempt_2. Task score = mean over its test inputs; the
total is out of 120 (ARC Prize scoring). Also reports pass@1 (top vote only), the oracle (any passing sample
correct), coverage (>= 1 passing sample) and per-arm sample statistics.

Usage: score.py --samples samples.jsonl --results programs.jsonl [--combos code,rulecode,rulecode_loo,all]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from validate import ROOT, code_id, extract


def key(g):
    return json.dumps(g, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--challenges", default=str(ROOT / "data/arc-agi_evaluation_challenges.json"))
    ap.add_argument("--solutions", default=str(ROOT / "data/arc-agi_evaluation_solutions.json"))
    ap.add_argument("--combos", default="code,rulecode,rulecode_loo,code+rulecode,all")
    ap.add_argument("--max-per-arm", type=int, help="use only the first N samples per task and arm")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    tasks = json.loads(Path(args.challenges).read_text())
    sols = json.loads(Path(args.solutions).read_text())
    res = {}
    for line in args.results.open():
        r = json.loads(line)
        res[r["id"]] = r
    samples = defaultdict(list)  # (task, arm) -> [sample info]
    for line in args.samples.open():
        rec = json.loads(line)
        per = rec["completion_tokens"] / max(1, len(rec["samples"]))
        for s in rec["samples"]:
            code, rule = extract(s["text"], rec["arm"])
            r = res.get(code_id(rec["task"], code)) if code else None
            samples[(rec["task"], rec["arm"])].append(dict(
                parsed=code is not None, finish=s["finish"], tokens=per, held_out=rec["held_out"],
                passed=bool(r and not r.get("crash") and r.get("train_ok") and all(r["train_ok"])),
                train_ok=(r or {}).get("train_ok"), test=(r or {}).get("test") or []))
    if args.max_per_arm:
        samples = {k: v[: args.max_per_arm] for k, v in samples.items()}
    arms = sorted({a for _, a in samples})

    print("Per-arm samples:")
    print(f"{'arm':14s} {'samples':>8s} {'parsed':>7s} {'length':>7s} {'train-pass':>10s} {'tok/sample':>10s}")
    for arm in arms:
        ss = [s for (t, a), v in samples.items() if a == arm for s in v]
        print(f"{arm:14s} {len(ss):8d} {sum(s['parsed'] for s in ss) / len(ss):7.1%} "
              f"{sum(s['finish'] == 'length' for s in ss) / len(ss):7.1%} "
              f"{sum(s['passed'] for s in ss) / len(ss):10.1%} {sum(s['tokens'] for s in ss) / len(ss):10.0f}")
    loo = [s for (t, a), v in samples.items() if a == "rulecode_loo" for s in v if s["train_ok"]]
    if loo:
        shown_ok = [s for s in loo if all(ok for i, ok in enumerate(s["train_ok"]) if i != s["held_out"])]
        held = sum(s["train_ok"][s["held_out"]] for s in shown_ok)
        print(f"rulecode_loo: {len(shown_ok)} samples fit every shown pair; {held} of them ({held / max(1, len(shown_ok)):.1%})"
              " also reproduce the held-out pair")

    out = {}
    print(f"\n{'combo':16s} {'pass@2':>8s} {'pass@1':>8s} {'oracle':>8s} {'coverage':>9s} {'P(top1 ok|cov)':>14s}")
    for combo in args.combos.split(","):
        use = arms if combo == "all" else combo.split("+")
        tot2 = tot1 = orc = cov = 0.0
        for tid, task in tasks.items():
            n_test = len(task["test"])
            s2 = s1 = so = sc = 0.0
            for i in range(n_test):
                votes = Counter()
                for arm in use:
                    for s in samples.get((tid, arm), []):
                        if s["passed"] and i < len(s["test"]) and not isinstance(s["test"][i], dict):
                            votes[key(s["test"][i])] += 1
                truth = key(sols[tid][i])
                ranked = [g for g, _ in votes.most_common()]
                s2 += truth in ranked[:2]
                s1 += truth in ranked[:1]
                so += truth in votes
                sc += bool(votes)
            tot2 += s2 / n_test
            tot1 += s1 / n_test
            orc += so / n_test
            cov += sc / n_test
        out[combo] = dict(pass2=tot2, pass1=tot1, oracle=orc, coverage=cov)
        print(f"{combo:16s} {tot2:7.2f}  {tot1:7.2f}  {orc:7.2f}  {cov:8.2f}  {tot1 / max(cov, 1e-9):13.1%}"
              f"   (of {len(tasks)} tasks)")
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
