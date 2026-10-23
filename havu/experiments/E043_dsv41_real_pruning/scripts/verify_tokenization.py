#!/usr/bin/env python3
"""E043: check rebuilt token sequences against the API's per-turn usage.

For assistant turn k (k-th run of role-1 tokens): the API's prompt_tokens must equal the run's start offset
(everything before it, including '<｜Assistant｜><think>'), and completion_tokens must match the run length
(the API does not count the closing EOS, so run_len - 1 is expected). Prints the distribution of mismatches.

  verify_tokenization.py --tok DIR [DIR ...] [--show 5]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def runs(roles):
    d = np.diff(np.concatenate([[0], roles.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    dp, dc = collections.Counter(), collections.Counter()
    n_roll = n_turn = bad_roll = turn_count_mismatch = 0
    shown = 0
    for d in a.tok:
        roles = np.fromfile(Path(d) / "roles.u8", dtype=np.uint8)
        for line in open(Path(d) / "index.jsonl"):
            r = json.loads(line)
            usage = r["api_usage"]
            rr = runs(roles[r["offset"]: r["offset"] + r["length"]])
            n_roll += 1
            if len(rr) != len(usage):
                turn_count_mismatch += 1
            bad = False
            for (s, ln), (p, c) in zip(rr, usage):
                n_turn += 1
                dp[p - s] += 1
                dc[c - (ln - 1)] += 1
                bad |= (p != s) or (c != ln - 1)
            if bad:
                bad_roll += 1
                if shown < a.show:
                    shown += 1
                    print("MISMATCH", r["split"], r["task_id"], [(s, ln) for s, ln in rr][:6], usage[:6])
    print(json.dumps({"rollouts": n_roll, "turns": n_turn, "rollouts_with_mismatch": bad_roll,
                      "turn_count_mismatch": turn_count_mismatch,
                      "prompt_delta": dict(dp.most_common(8)), "completion_delta": dict(dc.most_common(8))}))


if __name__ == "__main__":
    main()
