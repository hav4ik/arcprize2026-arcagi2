#!/usr/bin/env python3
"""E047: Kaggle 12-h estimate (E042/E046 multi_sim.py, unchanged model) with attempt 1 as a verifycode check.
The E128 records have no informed attempts, so attempt 1 is transformed with Flash-1017 ratios (E047 OpenRouter runs):
tokens x TOK, turns x TURNS; where attempt 0 is wrong on a pair, attempt 1's (fresh E128) right answer is kept with
probability RHO = P(fix | verifycode) / P(fix | fresh); where attempt 0 is right, attempt 1 resubmits it.
Usage (repo root): kaggle_sim.py --multi <E128 multi jsonl> [--configs 36] [--gate 0.6] [--thrash 0.88:0.6]
                   [--tok 0.575 --turns 0.68 --rho 0.6] [--reps 80]"""
import inspect, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/E042_dsv41_l4_synthetic/scripts"))
import multi_sim as ms

VC = {"on": False, "tok": 0.575, "turns": 0.68, "rho": 0.6}
# --split: per-class ratios by attempt 0's train check (E047 OpenRouter runs; fail = verifycode, pass = verify)
SPLIT = None  # {False: (tok, turns, rho), True: (tok, turns, rho)}


def transform(att, rng):
    if not VC["on"]:
        return att
    out = {}
    for i, (r0, r1) in att.items():
        tok, turns, rho = SPLIT[bool(r0["train_pass"])] if SPLIT else (VC["tok"], VC["turns"], VC["rho"])
        c = [bool(a or (b and rng.random() < rho)) for a, b in zip(r0["correct"], r1["correct"])]
        out[i] = (r0, dict(r1, tokens=max(1, int(r1["tokens"] * tok)),
                           turns=max(1, round(r1["turns"] * turns)), correct=c))
    return out


src = inspect.getsource(ms.simulate)
line = "    att = {i: (rng.sample(jobs_pool[k], 2) if len(jobs_pool[k]) >= 2 else jobs_pool[k] * 2) for i, k in enumerate(keys)}\n"
assert line in src
src = src.replace(line, line + "    att = TRANSFORM(att, rng)\n")
ns = dict(vars(ms), TRANSFORM=transform)
exec(src, ns)
ms.simulate = ns["simulate"]

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--split" in args:  # --split fail_tok,fail_turns,fail_rho:pass_tok,pass_turns,pass_rho
        j = args.index("--split"); f, g = args[j + 1].split(":"); del args[j:j + 2]
        SPLIT = {False: tuple(map(float, f.split(","))), True: tuple(map(float, g.split(",")))}
    for k in ("tok", "turns", "rho"):
        if f"--{k}" in args:
            j = args.index(f"--{k}"); VC[k] = float(args[j + 1]); del args[j:j + 2]
    for on in (False, True):
        VC["on"] = on
        print(f"=== attempt 1 = {('split ' + str(SPLIT)) if on and SPLIT else ''}{'verifycode check (tok x%.3f, turns x%.2f, rho %.2f)' % (VC['tok'], VC['turns'], VC['rho']) if on else 'fresh (as submitted)'}")
        sys.argv = [sys.argv[0]] + args
        ms.main()
