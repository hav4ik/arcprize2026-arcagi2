#!/usr/bin/env python3
"""E054 diagnostic: is torch.ops._C.persistent_topk deterministic against itself on a fixed logits buffer?

Rebuilds indexer_ab.py's inputs for one dense decode case (same seeds), computes OLD and NEW logits, then runs
persistent_topk REPS times on OLD's buffer and on NEW's buffer. Reports:
  - logits bitwise OLD == NEW;
  - persistent_topk OLD-vs-OLD (self) and OLD-vs-NEW set equality (indexer_ab's fin_set comparison);
  - for rows whose sets differ: the k-th largest value, how many columns hold exactly that value, and whether every
    differing index sits at that tied value (=> tie-breaking, not a logits difference);
  - top-512 VALUE multisets equal (tie-insensitive).

  python topk_selfcheck.py --rows 36 --width 81920 --page 64 --ctx full [--reps 5]
"""
import argparse
import importlib.util
import json
import sys

import torch

AB = "/root/e054/src/E054_exact_cherrypicks/gpu/indexer_ab.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--page", type=int, required=True)
    p.add_argument("--ctx", default="full")
    p.add_argument("--seed", type=int, default=0, help="ctx_rows seed (only for mix / fixed kinds)")
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("iab", AB)
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401

    class A:
        old = None
        kit_tgz = "/root/kit/vllm_overlay_rw.tgz"
    OLD = ab.load(ab.find_old(A), "o_idx")
    NEW = ab.load(ab.Path(AB).parent.parent / "overlay" / ab.REL, "n_idx",
                  {"DSV41_INDEXER_SKIP": "1", "DSV41_INDEXER_SKIP_CANDIDATES": "1"})
    pool = ab.Pool(21_490)
    rows, width, page = a.rows, a.width, a.page
    ratio = 2 if width == 81_920 else 1
    q, qs, w = ab.make_q(rows, 1000 + rows)
    bt = ab.make_bt(rows, 21_490, 2000 + rows)
    ends = ab.ctx_rows(a.ctx, rows, ratio, width, a.seed)
    if rows == 40:
        ends = ends[:36] + [0, 0, 0, 0]
    c = ab.Case("diag", "dense", rows, width, page, ends, q, qs, w, bt)
    c.k = pool.view(page)
    out = torch.empty((rows, width), dtype=torch.float32, device="cuda")
    o1 = c.call(OLD, c.ends_t, out).clone()
    n1 = c.call(NEW, c.ends_t, torch.empty_like(out)).clone()
    lens = c.ends_t.clamp_min(0)

    def fin_set(idx, lg):
        safe = idx.clamp(0, lg.shape[1] - 1).long()
        keep = (idx >= 0) & torch.isfinite(torch.gather(lg, 1, safe))
        return torch.sort(torch.where(keep, idx, -1), dim=1)[0]

    old_sets = [fin_set(ab.prod_topk(o1, lens), o1) for _ in range(a.reps)]
    new_sets = [fin_set(ab.prod_topk(n1, lens), n1) for _ in range(a.reps)]
    self_old = [torch.equal(old_sets[0], s) for s in old_sets[1:]]
    self_new = [torch.equal(new_sets[0], s) for s in new_sets[1:]]
    across = [torch.equal(old_sets[0], s) for s in new_sets]
    # tie analysis on OLD rep 0 vs every other OLD rep / NEW rep
    ties = []
    for name, other in [("old_rep%d" % i, s) for i, s in enumerate(old_sets[1:], 1)] + \
                       [("new_rep%d" % i, s) for i, s in enumerate(new_sets)]:
        for r in range(rows):
            x, y = old_sets[0][r], other[r]
            if torch.equal(x, y):
                continue
            n = int(lens[r])
            kth = torch.topk(o1[r, :max(n, 1)], min(512, n)).values[-1]
            sx, sy = set(x.tolist()) - {-1}, set(y.tolist()) - {-1}
            diff = sorted((sx ^ sy))
            dvals = o1[r, torch.tensor(diff, device="cuda", dtype=torch.long)]
            ties.append({"vs": name, "row": r, "n_diff": len(diff), "kth": float(kth),
                         "n_at_kth": int((o1[r, :n] == kth).sum()), "n_above_kth": int((o1[r, :n] > kth).sum()),
                         "all_diff_at_kth": bool(torch.all(dvals == kth))})
    vals_equal = all(torch.equal(torch.sort(torch.gather(o1, 1, old_sets[0].clamp_min(0).long()))[0],
                                 torch.sort(torch.gather(n1, 1, s.clamp_min(0).long()))[0]) for s in new_sets)
    rec = {"rows": rows, "width": width, "ctx": a.ctx, "logits_bit_identical": ab.bits_equal(o1, n1),
           "persistent_topk_old_self_equal": self_old, "persistent_topk_new_self_equal": self_new,
           "persistent_topk_old_vs_new_equal": across, "topk_value_multisets_equal_old_vs_new": vals_equal,
           "diff_rows": len(ties), "all_diffs_are_ties_at_kth": all(t["all_diff_at_kth"] for t in ties),
           "ties_sample": ties[:6]}
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    sys.exit(main())
