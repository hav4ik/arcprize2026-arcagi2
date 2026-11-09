#!/usr/bin/env python3
"""Check evals/auto_repair.py against the replay's proposals (out/replay_<model>.pkl) on the recorded error calls.

For every replayed identifier-error call: suggest() vs the replay's first rewrite (same typo -> same name, or both
none), and how often the side-effect check refuses. Needs the attempts pickles (typos_cache in the scratchpad or out/).
Writes out/auto_repair_validation.json and, with --fixture, a test fixture of sampled cases.
"""
import ast
import collections
import json
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/evals")
from auto_repair import _bound_names, propose, ran_side_effects, suggest  # noqa: E402

CACHE = [os.path.join(HERE, "out"), "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/"
         "5b3f7cae-743a-4aac-ba8b-b255a5f73280/scratchpad/typos_cache"]
PRELOADED = {"train", "test_input", "np", "json"}


def attempts(model):
    for d in CACHE:
        p = os.path.join(d, f"attempts_{model}.pkl")
        if os.path.exists(p):
            return {(a["task_id"], a["pair"], a["attempt"]): a for a in pickle.load(open(p, "rb"))}
    raise SystemExit(f"no attempts pickle for {model}")


def bound(code):
    try:
        return _bound_names(ast.parse(code))
    except SyntaxError:
        return set()


def main():
    models = ["E128", "E112", "E96", "DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95"]
    fixture = "--fixture" in sys.argv
    res, cases = {}, []
    for m in models:
        A = attempts(m)
        rows = [r for r in pickle.load(open(os.path.join(HERE, "out", f"replay_{m}.pkl"), "rb"))
                if "fail" not in r and r.get("replay_matches")]
        c = collections.Counter()
        for r in rows:
            a = A[r["key"]]
            seq = [x for t in a["turns"] for x in t["calls"]]
            call = seq[r["call_seq"]]
            session = set(PRELOADED)
            for x in seq[: r["call_seq"]]:
                if x["name"] == "run_python" and x["status"] != "error":
                    session |= bound(x["code"] or "")
                elif x["name"] == "run_python":
                    session |= bound(x["code"] or "")  # partial runs bind names too (the replay used the same)
            want = tuple(r["rewrites"][0][:2]) if r["rewrites"] else None
            s = suggest(call["code"] or "", call["output"] or "", session, min_len=1, dtype_rule=False)
            got = tuple(s[:2]) if s else None
            c["calls"] += 1
            c["agree"] += got == want
            c["both_none"] += got is None and want is None
            c["replay_only"] += got is None and want is not None
            c["module_only"] += got is not None and want is None
            c["differ"] += got is not None and want is not None and got != want
            p = propose(call["code"] or "", call["output"] or "", session)
            c["proposed"] += p is not None
            if s and p is None:
                short = len(s[0].rpartition(".")[2]) < 4 and s[2] != "kwarg"
                c["not_proposed_short_name" if short else "not_proposed_side_effects_or_other"] += 1
            done = r["final"] and r["final"][0] == "completed"
            c["replay_completed"] += bool(done)
            c["completed_and_proposed"] += bool(done and p)
            if fixture and want is not None or (fixture and random.Random(hash(r["key"]) % 1000).random() < 0.05):
                cases.append({"model": m, "key": list(r["key"]), "call_seq": r["call_seq"], "code": call["code"],
                              "error_text": call["output"], "session_names": sorted(session),
                              "replay": list(want) if want else None, "proposed": list(p) if p else None})
        res[m] = dict(c)
        print(m, dict(c), flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "auto_repair_validation.json"), "w"), indent=1)
    if fixture:
        random.Random(0).shuffle(cases)
        cases = [x for x in cases if len(x["code"] or "") < 2500]
        pick = ([x for x in cases if x["model"].startswith(("E128", "DEV")) and x["replay"]][:20]
                + [x for x in cases if x["model"] == "E112" and x["replay"]][:10]
                + [x for x in cases if not x["replay"]][:8])
        json.dump(pick, open("/home/vu/Workspace/ARC-AGI-2/evals/tests/fixtures/auto_repair_cases.json", "w"), indent=1)
        print("fixture cases", len(pick))


if __name__ == "__main__":
    main()
