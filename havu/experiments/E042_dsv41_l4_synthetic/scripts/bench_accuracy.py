#!/usr/bin/env python3
"""E042: per-attempt accuracy of Kaggle/pod eval-bench runs (solver attempts.jsonl) against E128's runs on the 8x box
on the same pairs. A coarse check that a serving change (e.g. the Humming dense kernel) does not cost accuracy; benches
finish the smallest tasks first, so they overlap.

T1 (kaggle/dsv41_tools/next_candidate_fixes.md; the old version produced a false "~3 pp L4 gap"):
- pairs are matched by their test input and expected output, not by (task, pair index): the Kaggle eval json and the
  GitHub eval set the references ran on differ in 5 tasks;
- the reference follows the run's mode: a run with multi-test records (one rollout per task) is compared with E128's
  multi-test runs (E046 runs 7, 3 and 3b), a per-pair run with E043's per-pair E128 run;
- an attempt without an answer counts as wrong on both sides; errored records are left out on both sides (the solver
  re-runs them) and counted;
- attempts cut by the force deadline or the loop guard (`cuts`) are reported separately, outside the comparison.
Usage: bench_accuracy.py [--challenges C] <solutions.json> <name>=<attempts.jsonl> [<name>=<attempts.jsonl> ...]
(--challenges: the bench's challenge file; default: the solutions path with "solutions" -> "challenges").
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = ROOT / "evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_"
REFERENCES = {"pair": ["e128t_w10_u2_gptq_eng64_sub80"],  # E043, 8x box, per-pair attempts
              "multi": ["e128t_final_all", "e128t_mt3_sub80_tx_b05", "e128t_mt3b_sub80_tx_b05"]}  # E046 runs 7, 3, 3b


def pair_key(test_input, output):
    """A test pair independent of its task id and position: hash of its input and expected output."""
    return hashlib.sha1(json.dumps([test_input, output], separators=(",", ":")).encode()).hexdigest()[:16]


def load_set(challenges, solutions=None):
    """{(task, pair index): pair_key} of a challenge set: a directory of task files with test outputs, or a
    challenge json + solutions json."""
    path = Path(challenges)
    if path.is_dir():
        tasks = {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}
        outs = {t: [x.get("output") for x in task["test"]] for t, task in tasks.items()}
    elif solutions is None:
        return {}
    else:
        tasks = json.loads(path.read_text())
        outs = solutions if isinstance(solutions, dict) else json.loads(Path(solutions).read_text())
    return {(t, q): pair_key(x["input"], outs[t][q]) for t, task in tasks.items() if t in outs
            for q, x in enumerate(task["test"]) if q < len(outs[t]) and outs[t][q] is not None}


def load_reference(names, prefix=PREFIX):
    """{pair_key: [correct per attempt]} from harness result files (records are per pair, also for multi-test runs),
    each matched through its own challenge set (config.challenges)."""
    ref, sets = collections.defaultdict(list), {}
    for name in names:
        for r in map(json.loads, open(f"{prefix}{name}.jsonl")):
            if "task_id" not in r or r.get("error"):
                continue
            ch = r["config"]["challenges"]
            if ch not in sets:
                sets[ch] = load_set(ch, r["config"].get("solutions"))
            k = sets[ch].get((r["task_id"], r["pair"]))
            if k:
                ref[k].append(bool(r.get("correct")))
    return ref


def load_bench(path, keys, solutions):
    """A solver attempts.jsonl -> {"att": {(pair_key, attempt): correct}, "cut": {...}, "errors": n, "multi": bool,
    "unmatched": n}. Multi-test records are split per pair; the latest record of an attempt wins."""
    out = {"att": {}, "cut": {}, "errors": 0, "multi": False, "unmatched": 0}
    for r in map(json.loads, open(path)):
        if "task_id" not in r:
            continue
        if r.get("error"):
            out["errors"] += 1
            continue
        t, n = r["task_id"], r.get("multi_test")
        out["multi"] |= bool(n)
        pred = r.get("pred")
        preds = (pred if isinstance(pred, list) and len(pred) == n else [None] * n) if n else [pred]
        for q, g in (enumerate(preds) if n else [(r["pair"], pred)]):
            k = keys.get((t, q))
            if k is None:
                out["unmatched"] += 1
                continue
            ok = g is not None and g == solutions[t][q]
            out["att"].pop((k, r["attempt"]), None)
            out["cut"].pop((k, r["attempt"]), None)
            out["cut" if r.get("cuts") else "att"][(k, r["attempt"])] = ok
    return out


def compare(bench, ref):
    """Bench accuracy vs the reference's per-attempt rate on the same pairs (pairs the reference lacks are left out)."""
    att = {k: v for k, v in bench["att"].items() if ref.get(k[0])}
    pairs = {k for k, _ in att}
    exp = [sum(ref[k]) / len(ref[k]) for k, _ in att]
    cut = {k: v for k, v in bench["cut"].items() if ref.get(k[0])}
    return {"attempts": len(att), "pairs": len(pairs), "correct": sum(att.values()),
            "acc": sum(att.values()) / max(1, len(att)), "ref_acc": sum(exp) / max(1, len(exp)),
            "cut": len(cut), "cut_correct": sum(cut.values()), "errors": bench["errors"],
            "no_ref": len(bench["att"]) - len(att), "unmatched": bench["unmatched"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--challenges")
    ap.add_argument("solutions")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    solutions = json.loads(Path(a.solutions).read_text())
    keys = load_set(a.challenges or a.solutions.replace("solutions", "challenges"), solutions)
    refs = {}
    runs = {}
    for arg in a.runs:
        name, path = arg.split("=", 1)
        runs[name] = load_bench(path, keys, solutions)
    for name, bench in runs.items():
        mode = "multi" if bench["multi"] else "pair"
        if mode not in refs:
            refs[mode] = load_reference(REFERENCES[mode])
        s = compare(bench, refs[mode])
        print(f"{name:12s} {s['attempts']:3d} attempts on {s['pairs']:3d} pairs: correct {s['correct']:3d} "
              f"({100 * s['acc']:.1f}%); E128 {mode} reference on these pairs {100 * s['ref_acc']:.1f}% "
              f"(diff {100 * (s['acc'] - s['ref_acc']):+.1f}); cut {s['cut']} ({s['cut_correct']} correct), "
              f"errors {s['errors']}, no reference {s['no_ref']}, unmatched {s['unmatched']}")
    names = list(runs)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            common = set(runs[x]["att"]) & set(runs[y]["att"])
            if common:
                cx, cy = sum(runs[x]["att"][k] for k in common), sum(runs[y]["att"][k] for k in common)
                print(f"{x} vs {y}: {len(common)} shared attempts, correct {cx} vs {cy}")


if __name__ == "__main__":
    main()
