#!/usr/bin/env python3
"""When evals/auto_repair.py's rewrite differs from the model's own next-call fix: by error kind, does the retried
cell still complete (a silent wrong rename) or fail again, and how did those attempts end in the recorded run?

The recorded outcome is the model's own path (no auto-repair ran), so it is context, not the effect of the repair.
Output: out/repair_disagreements.json and a printed table.
"""
import collections
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_auto_repair as v  # noqa: E402
from auto_repair import propose  # noqa: E402

FIX_SRC = ("fix=vocab", "fix", "fix-far", "fix-span")


def main():
    out, examples = {}, collections.defaultdict(list)
    for m in ["E128", "E112", "E96"]:
        A = v.attempts(m)
        ie = {(r["task_id"], int(r["pair"]), int(r["attempt"]), int(r["call_seq"])): r
              for r in pickle.load(open(os.path.join(HERE, "out", f"ident_errors_{m}.pkl"), "rb"))}
        rows = [r for r in pickle.load(open(os.path.join(HERE, "out", f"replay_{m}.pkl"), "rb"))
                if "fail" not in r and r.get("replay_matches")]
        c = collections.defaultdict(collections.Counter)
        for r in rows:
            a = A[r["key"]]
            seq = [x for t in a["turns"] for x in t["calls"]]
            call = seq[r["call_seq"]]
            sess = set(v.PRELOADED)
            for x in seq[: r["call_seq"]]:
                if x["name"] == "run_python":
                    sess |= v.bound(x["code"] or "")
            p = propose(call["code"] or "", call["output"] or "", sess)
            if not p:
                continue
            typo, repl = p[1][len("[auto-corrected: "):-1].split(" -> ")
            e = ie.get((*r["key"], r["call_seq"]))
            kind = e["kind"] if e else "?"
            k = c[kind]
            k["proposed"] += 1
            done = bool(r["final"] and r["final"][0] == "completed") and r["rewrites"] and r["rewrites"][0][1] == repl
            k["retry_completes"] += done
            if not e or e.get("intended_src") not in FIX_SRC or e["typo"].split(".")[-1] != typo:
                k["no_model_fix"] += 1
                continue
            fix = e["intended"].split(".")[-1]
            if "=" in fix:  # dbool -> dtype=bool style
                fix = fix
            correct = bool(a["correct"])
            if fix == repl:
                k["agree"] += 1
                k["agree_attempt_correct"] += correct
            else:
                k["differ"] += 1
                k["differ_retry_completes"] += done
                k["differ_attempt_correct"] += correct
                examples[m].append({"kind": kind, "typo": typo, "rewrite": repl, "model_fix": e["intended"],
                                    "retry_completes": done, "attempt_correct": correct, "key": list(r["key"])})
        out[m] = {k: dict(x) for k, x in c.items()}
        print(f"\n{m}")
        print(f"{'kind':8s} {'prop':>5s} {'retry ok':>8s} {'no fix':>6s} {'agree':>5s} {'differ':>6s} "
              f"{'differ: retry ok':>16s} {'differ: att correct':>19s} {'agree: att correct':>18s}")
        for kind, x in sorted(c.items(), key=lambda t: -t[1]["proposed"]):
            print(f"{kind:8s} {x['proposed']:5d} {x['retry_completes']:8d} {x['no_model_fix']:6d} {x['agree']:5d} "
                  f"{x['differ']:6d} {x['differ_retry_completes']:16d} {x['differ_attempt_correct']:19d} "
                  f"{x['agree_attempt_correct']:18d}")
        for ex in examples[m][:12]:
            print("   ", ex["kind"], f"{ex['typo']} -> {ex['rewrite']} (model: {ex['model_fix']})",
                  "retry ok" if ex["retry_completes"] else "retry fails", "| attempt", "correct" if ex["attempt_correct"] else "wrong")
    json.dump({"counts": out, "examples": examples}, open(os.path.join(HERE, "out", "repair_disagreements.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
