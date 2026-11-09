#!/usr/bin/env python3
"""Step 1: compact every attempt of the analysed runs into out/attempts_<model>.pkl (and the training rollouts into
out/attempts_TRAIN.pkl). Read-only over evals/results."""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, RUNS, TRAIN_TAGS, compact_attempt, iter_records  # noqa: E402


def job(args):
    rec, model, tag = args
    a = compact_attempt(rec, model, tag)
    if a is not None and tag.endswith("_r2"):
        a["attempt"] = 1  # E112 r2 = its logical second attempt (main attempt 1 is excluded)
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {}
    for model, tags in RUNS.items():
        jobs[model] = [(r, model, tag) for tag, att in tags for r in iter_records(tag, att)]
    jobs["TRAIN"] = [(r, "TRAIN", tag) for tag in TRAIN_TAGS for r in iter_records(tag)]
    with ProcessPoolExecutor(16) as ex:
        for model, js in jobs.items():
            res = [a for a in ex.map(job, js, chunksize=8) if a is not None]
            pickle.dump(res, open(f"{OUT}/attempts_{model}.pkl", "wb"))
            n_calls = sum(len(c["calls"]) for a in res for c in a["turns"])
            print(f"{model}: {len(res)}/{len(js)} attempts, {n_calls} calls", flush=True)


if __name__ == "__main__":
    main()
