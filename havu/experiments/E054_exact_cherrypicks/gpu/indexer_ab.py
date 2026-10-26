#!/usr/bin/env python3
"""E054 (a): fork PR #156 part 1 kernel A/B at E128 decode shapes, on one GPU (SM89 4090 / L4).

OLD = the kit overlay's dsv41_indexer.py (sha256 3da076e1), NEW = E054's, loaded side by side
from their files (no install needed). Inputs are generated once per shape from a fixed seed
on the GPU (MXFP4 q / K bytes, ue8m0 scales 2^-9..2^-2, fp32 head weights) in the real page
pool geometry: --pool-blocks 21490 pages of 113,920 B (Kaggle's pool; the indexer K view is
as_strided [blocks, page, 68] with that block stride), block tables [reqs, 1280].
Per shape it checks, in order:
  1. OLD is deterministic against itself (two eager calls, bitwise);
  2. NEW == OLD bitwise over the whole [rows, width] buffer, eager and under CUDA-graph
     replay (graphs captured with different row_ends than the ones replayed: the grid must not
     depend on them), plus the split: scores below each row's context / -inf tail;
  3. identical top-512 sets: torch.topk on [0, ctx) and the production torch.ops._C.persistent_topk;
  4. microseconds per call under graph replay (the production path: FULL decode graphs),
     q predecode inside the graph as in production (DSV41_INDEXER_Q_PREDECODE=1 on SM89).
Shapes: dense decode rows 36 / 40 (36 + 4 padded rows with end 0, as a 36-stream batch replays
the 40-graph) / 48 / 64 x width 81,920 (ratio-2 sources 2/8/14, page 64) and 163,840 (ratio-1
source 20, page 128) x contexts 8K / 32K / 64K / 128K raw tokens, a mixed 4K-160K draw and the
full width; candidate calls (16,384 wide, 2048 blocks) for layers 24-36; two prefill chunks
(dense packed rows). A NEW-only sweep of DSV41_INDEXER_TARGET_PROGRAMS checks bit-identity
and speed at other program counts (pick the best for L4 later).
Output: one JSON line per case, then a summary with the per-step indexer time per GPU
(3 x dense 81,920 + 1 x dense 163,840 + 4 x candidates) for the mixed contexts.

  /root/dsv41/venv/bin/python gpu/indexer_ab.py --json runs/indexer_ab.json            # ~10 min
  ... --quick                                                                          # ~2 min
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DSV41_INDEXER_Q_PREDECODE", "1")  # production env on SM89 (server_cmd.json)
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REL = "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py"
OLD_SHA = "3da076e17cb043c57c75f660b0f1b1d2fd15a5d431fb1d7730d192520e629d2f"
MAX_MODEL_LEN = 163_840
BT_WIDTH = MAX_MODEL_LEN // 128
BLOCK_BYTES = 113_920
TOPK = 512


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_old(a) -> Path:
    cands = [Path(a.old)] if a.old else []
    venv = Path(sys.prefix)
    cands += [venv / "e054_backup" / REL]
    for c in cands:
        if c.is_file() and sha(c) == OLD_SHA:
            return c
    kit = Path(a.kit_tgz)
    if kit.is_file():
        with tarfile.open(kit) as tf:
            data = tf.extractfile(REL).read()
        if hashlib.sha256(data).hexdigest() == OLD_SHA:
            d = Path(tempfile.mkdtemp(prefix="e054_old_"))
            (d / "dsv41_indexer.py").write_bytes(data)
            return d / "dsv41_indexer.py"
    sys.exit(f"no OLD indexer with sha {OLD_SHA[:8]}: pass --old or --kit-tgz")


def load(path: Path, name: str, env=None):
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return mod


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


class Pool:
    """K pages in Kaggle's packed-pool geometry, shared by every shape."""

    def __init__(self, blocks, seed=54):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.blocks = blocks
        self.buf = torch.randint(0, 256, (blocks, BLOCK_BYTES), dtype=torch.uint8, device="cuda", generator=g)
        # ue8m0 scale bytes 118..125 for both page sizes' scale areas (64 or 128 tokens/page)
        for page in (64, 128):
            o = page * 64
            self.buf[:, o:o + page * 4] = torch.randint(118, 126, (blocks, page * 4), dtype=torch.uint8,
                                                        device="cuda", generator=g)

    def view(self, page):
        return torch.as_strided(self.buf, (self.blocks, page, 68), (BLOCK_BYTES, 68, 1))


def make_q(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda", generator=g)
    qs = torch.randint(118, 126, (rows, 32, 4), dtype=torch.uint8, device="cuda", generator=g)
    w = torch.randn(rows, 32, device="cuda", generator=g)
    return q, qs, w


def make_bt(reqs, blocks, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, blocks, (reqs, BT_WIDTH), dtype=torch.int32, device="cuda", generator=g)


def ctx_rows(kind, rows, ratio, width, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "full":
        raw = [width * ratio] * rows
    elif kind == "mix":
        raw = torch.randint(4096, 160_000, (rows,), generator=g).tolist()
    else:
        raw = [int(kind)] * rows
    return [min(width, r // ratio) for r in raw]


def time_graph(graph, iters, reps=5):
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(ts)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()  # warm (compile) outside the graph
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def topk_sets(logits, ends):
    out = []
    for r in range(logits.shape[0]):
        n = int(ends[r])
        k = min(TOPK, max(n, 0))
        out.append(torch.sort(torch.topk(logits[r, :max(n, 1)], k).indices)[0] if k else None)
    return out


def prod_topk(logits, ends):
    try:
        idx = torch.empty((logits.shape[0], TOPK), dtype=torch.int32, device="cuda")
        ws = torch.empty((1024 * 1024,), dtype=torch.uint8, device="cuda")
        torch.ops._C.persistent_topk(logits, ends, idx, ws, TOPK, logits.shape[1])
        return idx
    except Exception as exc:  # the op needs vllm._C loaded (import vllm first)
        return repr(exc)[:200]


class Case:
    def __init__(self, name, kind, rows, width, page, ends, q, qs, w, bt, candidates=None, prefill=None,
                 row_repeat=1):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.ends_t = torch.tensor(ends, dtype=torch.int32, device="cuda")

    def call(self, mod, ends_t, out):
        if self.candidates is not None:
            kw = dict(row_starts=self.prefill[0], cu_seq_lens=self.prefill[1], token_to_seq=self.prefill[2]) \
                if self.prefill else {}
            return mod.dsv41_mxfp4_candidate_logits(self.q, self.qs, self.k, self.w, self.bt, ends_t,
                                                    self.candidates, row_repeat=self.row_repeat, out=out, **kw)
        kw = dict(row_starts=self.prefill[0], cu_seq_lens=self.prefill[1], token_to_seq=self.prefill[2]) \
            if self.prefill else {}
        return mod.dsv41_mxfp4_dense_logits(self.q, self.qs, self.k, self.w, self.bt, ends_t, self.width,
                                            row_repeat=self.row_repeat, out=out, **kw)


def run_case(c: Case, OLD, NEW, a, extra_mods=()):
    width = c.width if c.candidates is None else c.candidates.shape[1] * 8
    out_old = torch.empty((c.rows, width), dtype=torch.float32, device="cuda")
    out_new = torch.empty_like(out_old)
    rec = {"case": c.name, "kind": c.kind, "rows": c.rows, "width": width, "page": c.page,
           "ctx_mean": round(sum(max(e, 0) for e in c.ends) / len(c.ends), 1)}
    # 1. OLD self-determinism, eager
    o1 = c.call(OLD, c.ends_t, out_old).clone()
    o2 = c.call(OLD, c.ends_t, out_old).clone()
    rec["old_deterministic"] = bits_equal(o1, o2)
    # 2. NEW vs OLD eager
    n1 = c.call(NEW, c.ends_t, out_new).clone()
    rec["eager_bit_identical"] = bits_equal(o1, n1)
    if c.candidates is None and c.prefill is None:
        lt = torch.arange(width, device="cuda")[None, :] < c.ends_t.clamp_min(0)[:, None]
        rec["scores_below_ctx_identical"] = torch.equal(o1[lt].view(torch.int32), n1[lt].view(torch.int32))
        rec["tail_all_neginf"] = bool(torch.all(torch.isneginf(n1[~lt])))
    for name, mod in extra_mods:
        rec[f"{name}_bit_identical"] = bits_equal(o1, c.call(mod, c.ends_t, torch.empty_like(out_new)))
    # 3. top-k sets (dense decode / candidates; prefill rows are checked bitwise only)
    if c.prefill is None:
        lens = c.ends_t.clamp_min(0) if c.candidates is None else torch.full_like(c.ends_t, width)
        rec["topk_sets_identical"] = all(
            (x is None and y is None) or (x is not None and torch.equal(x, y))
            for x, y in zip(topk_sets(o1, lens), topk_sets(n1, lens)))
        po, pn = (prod_topk(o1, lens), prod_topk(n1, lens)) if width >= TOPK else ("width < 512: skipped", None)
        if isinstance(po, torch.Tensor):
            # compare as sets of indices of finite logits (ties among -inf may be broken arbitrarily)
            def fin_set(idx, lg):
                safe = idx.clamp(0, lg.shape[1] - 1).long()
                keep = (idx >= 0) & torch.isfinite(torch.gather(lg, 1, safe))
                return torch.sort(torch.where(keep, idx, -1), dim=1)[0]
            rec["persistent_topk_identical"] = torch.equal(fin_set(po, o1), fin_set(pn, n1))
        else:
            rec["persistent_topk_identical"] = po
    # 4. graphs: capture with other row_ends than replayed, replay, compare, time
    if a.graphs and c.prefill is None:
        static = c.ends_t.clone()
        static.fill_(1 if c.candidates is None else width)  # capture-time contexts differ from the replayed ones
        g_old = capture(lambda: c.call(OLD, static, out_old))
        g_new = capture(lambda: c.call(NEW, static, out_new))
        static.copy_(c.ends_t)
        out_old.fill_(float("nan"))
        out_new.fill_(float("nan"))
        g_old.replay()
        g_new.replay()
        torch.cuda.synchronize()
        rec["graph_old_eq_eager"] = bits_equal(out_old, o1)
        rec["graph_bit_identical"] = bits_equal(out_old, out_new)
        rec["us_old"] = round(time_graph(g_old, a.iters), 2)
        rec["us_new"] = round(time_graph(g_new, a.iters), 2)
        rec["speedup"] = round(rec["us_old"] / rec["us_new"], 3)
        del g_old, g_new
    else:
        for tag, mod, out in (("us_old", OLD, out_old), ("us_new", NEW, out_new)):
            c.call(mod, c.ends_t, out)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(max(1, a.iters // 20)):
                    c.call(mod, c.ends_t, out)
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / max(1, a.iters // 20))
            rec[tag] = round(statistics.median(ts), 2)
        rec["speedup"] = round(rec["us_old"] / rec["us_new"], 3)
    rec["ok"] = all(v for k, v in rec.items() if k.endswith(("_identical", "_deterministic", "_eq_eager",
                                                              "_neginf")) and not isinstance(v, str))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", help="old dsv41_indexer.py (default: <venv>/e054_backup/..., else --kit-tgz)")
    p.add_argument("--kit-tgz", default="/root/kit/vllm_overlay_rw.tgz")
    p.add_argument("--new", default=str(HERE.parent / "overlay" / REL))
    p.add_argument("--pool-blocks", type=int, default=21_490)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    p.add_argument("--quick", action="store_true", help="rows 36/48, contexts 32K + mix only")
    p.add_argument("--json")
    a = p.parse_args()

    import vllm  # noqa: F401  (vllm.triton_utils for both kernel files)
    try:
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.*, incl. persistent_topk)
    except Exception as exc:
        print(json.dumps({"warning": f"vllm._custom_ops: {exc!r}"[:200]}), flush=True)

    old_path, new_path = find_old(a), Path(a.new)
    OLD = load(old_path, "e054_old_indexer")
    NEW = load(new_path, "e054_new_indexer", {"DSV41_INDEXER_SKIP": "1", "DSV41_INDEXER_SKIP_CANDIDATES": "1"})
    OFF = load(new_path, "e054_new_indexer_off", {"DSV41_INDEXER_SKIP": "0"})
    assert hasattr(NEW, "_mxfp4_indexer_logits_stride_kernel") and NEW._INDEXER_SKIP and not OFF._INDEXER_SKIP
    dev = torch.cuda.get_device_name()
    head = {"device": dev, "torch": torch.__version__, "old": str(old_path), "old_sha": sha(old_path)[:8],
            "new": str(new_path), "new_sha": sha(new_path)[:8], "pool_blocks": a.pool_blocks,
            "q_predecode_env": os.environ.get("DSV41_INDEXER_Q_PREDECODE"),
            "target_programs": NEW._LOGITS_TARGET_PROGRAMS}
    print(json.dumps(head), flush=True)

    pool = Pool(a.pool_blocks)
    rows_list = (36, 48) if a.quick else (36, 40, 48, 64)
    ctx_list = ("32768", "mix") if a.quick else ("8192", "32768", "65536", "131072", "mix", "full")
    results, t0 = [], time.time()
    seed = 0
    for rows in rows_list:
        q, qs, w = make_q(rows, 1000 + rows)
        bt = make_bt(rows, a.pool_blocks, 2000 + rows)
        for width, page, ratio in ((81_920, 64, 2), (163_840, 128, 1)):
            for kind in ctx_list:
                seed += 1
                ends = ctx_rows(kind, rows, ratio, width, seed)
                if rows == 40:
                    ends = ends[:36] + [0, 0, 0, 0]  # 36 live streams in the 40-row graph
                c = Case(f"dense_r{rows}_w{width}_{kind}", "dense", rows, width, page, ends, q, qs, w, bt)
                c.k = pool.view(page)
                r = run_case(c, OLD, NEW, a, [("killswitch", OFF)] if kind == "mix" else [])
                results.append(r)
                print(json.dumps(r), flush=True)
        # candidates: 2048 blocks of 8 compressed tokens (layers 24-36 read layer 20's selection)
        ends = ctx_rows("mix", rows, 1, 163_840, 3000 + rows)
        g = torch.Generator(device="cuda").manual_seed(4000 + rows)
        cand = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
        for r_, e in enumerate(ends):
            nblk = max(1, e // 8)
            take = min(2048, nblk)
            cand[r_, :take] = torch.sort(torch.randperm(nblk, device="cuda", generator=g)[:take])[0].to(torch.int32)
        c = Case(f"cand_r{rows}_mix", "candidates", rows, 16_384, 128, ends, q, qs, w, bt, candidates=cand)
        c.k = pool.view(128)
        r = run_case(c, OLD, NEW, a)
        results.append(r)
        print(json.dumps(r), flush=True)

    # eager decode at small / odd widths (mixed steps run eager with width = the batch's real max context;
    # a width of 1 is specialized as a constexpr by Triton)
    for rows, width in ((3, 1), (3, 17), (5, 255), (36, 4097), (36, 20_001)):
        q, qs, w = make_q(rows, 8000 + width)
        bt = make_bt(rows, a.pool_blocks, 9000 + width)
        g = torch.Generator().manual_seed(width)
        ends = torch.randint(0, width + 1, (rows,), generator=g).tolist()
        ends[0] = width
        c = Case(f"eager_r{rows}_w{width}", "eager", rows, width, 64, ends, q, qs, w, bt)
        c.k = pool.view(64)
        saved_graphs, a.graphs = a.graphs, False
        r = run_case(c, OLD, NEW, a)
        a.graphs = saved_graphs
        results.append(r)
        print(json.dumps(r), flush=True)

    # prefill: one 2048-token chunk after a 30K prefix; two requests (60K + 1024 new each) packed
    for name, reqs in (("prefill_1x30k", [(30_720, 2048)]), ("prefill_2x60k", [(61_440, 1024), (61_440, 1024)])):
        cu, t2s, starts, ends_l, total = [0], [], [], [], 0
        for ri, (prefix, new) in enumerate(reqs):
            n = prefix + new
            for i in range(new):
                starts.append(total)
                ends_l.append(total + prefix + i + 1)
            t2s += [ri] * n
            total += n
            cu.append(total)
        rows = len(starts)
        q, qs, w = make_q(rows, 5000 + rows)
        bt = make_bt(len(reqs), a.pool_blocks, 6000 + rows)
        pre = (torch.tensor(starts, dtype=torch.int32, device="cuda"), torch.tensor(cu, dtype=torch.int32, device="cuda"),
               torch.tensor(t2s, dtype=torch.int32, device="cuda"))
        c = Case(name, "prefill", rows, total, 128, ends_l, q, qs, w, bt, prefill=pre)
        c.k = pool.view(128)
        r = run_case(c, OLD, NEW, a)
        results.append(r)
        print(json.dumps(r), flush=True)

    # NEW-only program-count sweep on the mixed decode shapes (bit-identity + speed)
    sweep = []
    for target in (4096, 8192, 16_384, 32_768):
        NEW._LOGITS_TARGET_PROGRAMS = target
        for rows in (36, 48):
            q, qs, w = make_q(rows, 1000 + rows)
            bt = make_bt(rows, a.pool_blocks, 2000 + rows)
            for width, page, ratio in ((81_920, 64, 2), (163_840, 128, 1)):
                ends = ctx_rows("mix", rows, ratio, width, 7000 + rows)
                c = Case(f"sweep_t{target}_r{rows}_w{width}", "sweep", rows, width, page, ends, q, qs, w, bt)
                c.k = pool.view(page)
                r = run_case(c, OLD, NEW, a)
                r["target_programs"] = target
                sweep.append(r)
                print(json.dumps(r), flush=True)
    NEW._LOGITS_TARGET_PROGRAMS = 16_384

    def us(name, key):
        m = [r for r in results if r["case"] == name]
        return m[0][key] if m else None

    step = {}
    for rows in rows_list:
        if rows == 40:
            continue
        o = n = 0.0
        for nm, mult in ((f"dense_r{rows}_w81920_mix", 3), (f"dense_r{rows}_w163840_mix", 1), (f"cand_r{rows}_mix", 4)):
            o += mult * (us(nm, "us_old") or 0)
            n += mult * (us(nm, "us_new") or 0)
        step[rows] = {"indexer_us_per_step_old": round(o, 1), "indexer_us_per_step_new": round(n, 1),
                      "saved_ms_per_step": round((o - n) / 1000, 3)}
    summary = {"summary": True, "all_ok": all(r["ok"] for r in results + sweep),
               "failed": [r["case"] for r in results + sweep if not r["ok"]],
               "per_step_mix": step, "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(summary), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({**head, "cases": results, "sweep": sweep, **summary}, open(a.json, "w"), indent=1)
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
