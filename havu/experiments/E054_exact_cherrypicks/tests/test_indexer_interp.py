# SPDX-License-Identifier: Apache-2.0
"""E054 CPU check for fork PR #156 part 1 (grid-stride context skip), Triton interpreter.

Loads two copies of dsv41_indexer.py side by side:
  OLD = our 12-file overlay's version (sha256 3da076e1..., the kit file),
  NEW = experiments/E054_exact_cherrypicks/overlay/... (stride kernel),
plus NEW with DSV41_INDEXER_SKIP=0 and with DSV41_INDEXER_SKIP_CANDIDATES=0, and runs
the real @triton.jit kernels on CPU tensors (TRITON_INTERPRET=1). Checks:
  1. NEW == OLD bit for bit over the whole [rows, width] logits buffer (scores below the
     context and the -inf tail), on decode / prefill / candidate paths, edge contexts
     (0, 1, 15, 16, 17, width-1, width, > width, negative), tiny and odd widths, several
     tile-program counts (so tiles per program and tail fill loops > 1), row_repeat 2,
     unallocated (-1) and out-of-range block-table entries, a strided page pool, and an
     output buffer pre-filled with NaN (every column must be written);
  2. the kill switches reproduce OLD bit for bit;
  3. both agree with an independent PyTorch reference of the scoring (fp32 tolerance;
     identical -inf masks; identical top-k sets);
  4. the stride launch grid depends only on (rows, width), never on row_ends, and stays
     under the CUDA grid-y limit; no SWAP_GRID_AXES on the stride path.
The interpreter's tl.dot is numpy fp32, not the GPU MMA, so (1) proves the indexing,
masking and fill logic, not the GPU bits; gpu/indexer_ab.py checks those on SM89.

Run: tests/run_cpu_checks.sh (sets CUDA_VISIBLE_DEVICES="" TRITON_INTERPRET=1 PYTHONPATH).
"""

import hashlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import cpu_harness  # noqa: F401  (must precede torch/vllm imports)
import torch

HERE = Path(__file__).resolve().parent
E054 = HERE.parent
ROOT = E054.parents[1]
REL = "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py"
NEW_PATH = Path(os.environ.get("E054_NEW_INDEXER", E054 / "overlay" / REL))
OLD_SHA = "3da076e17cb043c57c75f660b0f1b1d2fd15a5d431fb1d7730d192520e629d2f"


def find_old() -> Path:
    """The kit version: $E054_OLD_INDEXER, the repo's readable overlay copy, or extracted from the kit tarball
    (repo kit, then /root/kit on a box)."""
    for c in (os.environ.get("E054_OLD_INDEXER"), ROOT / "kaggle/dsv41_tools/vllm_overlay_src" / REL):
        if c and Path(c).is_file():
            return Path(c)
    import tarfile
    import tempfile
    for kit in (ROOT / "kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz", Path("/root/kit/vllm_overlay_rw.tgz")):
        if kit.is_file():
            with tarfile.open(kit) as tf:
                data = tf.extractfile(REL).read()
            out = Path(tempfile.mkdtemp(prefix="e054_old_")) / "dsv41_indexer.py"
            out.write_bytes(data)
            return out
    raise SystemExit("no kit dsv41_indexer.py found: set E054_OLD_INDEXER")


OLD_PATH = find_old()


def load(path: Path, name: str, env: dict[str, str] | None = None):
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    mod._validate_inputs = lambda *a, **k: None  # its first check requires CUDA tensors
    return mod


assert hashlib.sha256(OLD_PATH.read_bytes()).hexdigest() == OLD_SHA, f"{OLD_PATH} is not the kit version"
OLD = load(OLD_PATH, "e054_old_indexer")
NEW = load(NEW_PATH, "e054_new_indexer")
NEW_OFF = load(NEW_PATH, "e054_new_indexer_off", {"DSV41_INDEXER_SKIP": "0"})
NEW_NOCAND = load(NEW_PATH, "e054_new_indexer_nocand", {"DSV41_INDEXER_SKIP_CANDIDATES": "0"})
assert NEW._INDEXER_SKIP and NEW._INDEXER_SKIP_CANDIDATES
assert not NEW_OFF._INDEXER_SKIP and not NEW_OFF._INDEXER_SKIP_CANDIDATES
assert NEW_NOCAND._INDEXER_SKIP and not NEW_NOCAND._INDEXER_SKIP_CANDIDATES

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


class Inputs:
    """Random MXFP4 indexer inputs in the kernel's segregated page layout."""

    def __init__(self, seed, rows, reqs, page, blocks_per_req, pool_blocks, stride_pad=40,
                 hole_frac=0.05, oob_frac=0.02):
        g = torch.Generator().manual_seed(seed)
        self.page = page
        self.q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, generator=g)
        self.qs = torch.randint(118, 126, (rows, 32, 4), dtype=torch.uint8, generator=g)
        self.stride = page * 68 + stride_pad
        buf = torch.randint(0, 256, (pool_blocks * self.stride,), dtype=torch.uint8, generator=g)
        for b in range(pool_blocks):  # scale bytes for the page's tokens, 2^-9 .. 2^-2
            o = b * self.stride + page * 64
            buf[o:o + page * 4] = torch.randint(118, 126, (page * 4,), dtype=torch.uint8, generator=g)
        self.buf = buf
        self.k = torch.as_strided(buf, (pool_blocks, page, 68), (self.stride, 68, 1))
        self.w = torch.randn(rows, 32, generator=g, dtype=torch.float32)
        bt = torch.randperm(pool_blocks, generator=g)[: reqs * blocks_per_req]
        bt = bt.to(torch.int32).view(reqs, blocks_per_req).clone()
        r = torch.rand(bt.shape, generator=g)
        bt[r < hole_frac] = -1
        bt[(r >= hole_frac) & (r < hole_frac + oob_frac)] = pool_blocks + 3  # >= num_cache_blocks
        self.bt = bt

    # ---- independent PyTorch reference -------------------------------------------------
    def _decode(self, nib, scale_byte):
        mag = E2M1[(nib & 7).long()]
        sign = torch.where((nib & 8) != 0, -1.0, 1.0).to(torch.float64)
        return sign * mag * torch.pow(2.0, scale_byte.double() - 127.0)

    def q_bf16(self, row):
        d = torch.arange(128)
        byte = self.q[row][:, d // 2].long()
        nib = torch.where(d % 2 == 0, byte & 0xF, byte >> 4)
        sc = self.qs[row][:, d // 32]
        return self._decode(nib, sc).float().to(torch.bfloat16).double()  # [32, 128]

    def k_bf16(self, phys, off):
        d = torch.arange(128)
        base = phys * self.stride
        byte = self.buf[base + off * 64 + d // 2].long()
        nib = torch.where(d % 2 == 0, byte & 0xF, byte >> 4)
        sc = self.buf[base + self.page * 64 + off * 4 + d // 32]
        return self._decode(nib, sc).float().to(torch.bfloat16).double()  # [128]

    def score(self, row, req, token):
        lb, off = token // self.page, token % self.page
        if token < 0 or lb >= self.bt.shape[1]:
            return None
        phys = int(self.bt[req, lb])
        if phys < 0 or phys >= self.k.shape[0]:
            return None
        hs = self.q_bf16(row) @ self.k_bf16(phys, off)  # [32]
        return float((hs.clamp_min(0.0) * self.w[row].double()).sum())


def ref_dense_decode(x: Inputs, ends, width, row_repeat=1):
    out = torch.full((len(ends), width), float("-inf"), dtype=torch.float64)
    for r, e in enumerate(ends):
        for c in range(max(0, min(int(e), width))):
            s = x.score(r, r // row_repeat, c)
            if s is not None:
                out[r, c] = s
    return out


def ref_dense_prefill(x: Inputs, starts, ends, cu, t2s, width):
    out = torch.full((len(ends), width), float("-inf"), dtype=torch.float64)
    for r, (s0, e) in enumerate(zip(starts, ends)):
        for c in range(max(0, int(s0)), min(int(e), width)):
            req = int(t2s[c])
            s = x.score(r, req, c - int(cu[req]))
            if s is not None:
                out[r, c] = s
    return out


def ref_candidates(x: Inputs, cand, ends, starts=None, req_of_row=None):
    rows, nblk = cand.shape
    out = torch.full((rows, nblk * 8), float("-inf"), dtype=torch.float64)
    for r in range(rows):
        lim = int(ends[r]) - (int(starts[r]) if starts is not None else 0)
        if starts is not None and int(ends[r]) <= int(starts[r]):
            continue
        for c in range(nblk * 8):
            b = int(cand[r, c // 8])
            tok = b * 8 + c % 8
            if b < 0 or tok >= lim:
                continue
            s = x.score(r, req_of_row(r), tok)
            if s is not None:
                out[r, c] = s
    return out


def check_against_ref(tc, got, ref, k=8):
    fin = torch.isfinite(ref)
    tc.assertTrue(torch.equal(fin, torch.isfinite(got)), "finite masks differ from the reference")
    tc.assertTrue(bool(torch.all(torch.isneginf(got[~fin]))), "non-scored columns must be -inf")
    if fin.any():
        diff = (got.double()[fin] - ref[fin]).abs()
        tol = 1e-5 * ref[fin].abs().max().clamp_min(1e-30)
        tc.assertLessEqual(float(diff.max()), float(tol))
    for r in range(ref.shape[0]):
        n = int(fin[r].sum())
        kk = min(k, n)
        if kk == 0:
            continue
        a = set(torch.topk(ref[r], kk).indices.tolist())
        b = set(torch.topk(got[r].double(), kk).indices.tolist())
        tc.assertEqual(a, b, f"top-{kk} set differs on row {r}")


def run_dense(mod, x: Inputs, ends, width, predecode=False, prefill=None, row_repeat=1):
    mod._use_q_predecode = lambda device: predecode
    out = torch.full((len(ends), width), float("nan"), dtype=torch.float32)
    kw = {}
    if prefill is not None:
        starts, cu, t2s = prefill
        kw = dict(row_starts=starts, cu_seq_lens=cu, token_to_seq=t2s)
    mod.dsv41_mxfp4_dense_logits(x.q, x.qs, x.k, x.w, x.bt, ends, width, row_repeat=row_repeat,
                                 out=out, **kw)
    return out


def run_cand(mod, x: Inputs, ends, cand, predecode=False, prefill=None, row_repeat=1):
    mod._use_q_predecode = lambda device: predecode
    out = torch.full((len(ends), cand.shape[1] * 8), float("nan"), dtype=torch.float32)
    kw = {}
    if prefill is not None:
        starts, cu, t2s = prefill
        kw = dict(row_starts=starts, cu_seq_lens=cu, token_to_seq=t2s)
    mod.dsv41_mxfp4_candidate_logits(x.q, x.qs, x.k, x.w, x.bt, ends, cand, row_repeat=row_repeat,
                                     out=out, **kw)
    return out


class TargetPrograms:
    """Temporarily change the stride kernel's total program target (bit-identity must hold)."""

    def __init__(self, mod, target, minimum=8):
        self.mod, self.target, self.minimum = mod, target, minimum

    def __enter__(self):
        self.saved = (self.mod._LOGITS_TARGET_PROGRAMS, self.mod._LOGITS_MIN_TILE_PROGRAMS)
        self.mod._LOGITS_TARGET_PROGRAMS, self.mod._LOGITS_MIN_TILE_PROGRAMS = self.target, self.minimum

    def __exit__(self, *exc):
        self.mod._LOGITS_TARGET_PROGRAMS, self.mod._LOGITS_MIN_TILE_PROGRAMS = self.saved


I32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731


class DenseDecode(unittest.TestCase):
    def test_edge_contexts_bitwise_and_reference(self):
        width, page = 1000, 64
        ends_l = [0, 1, 15, 16, 17, 999, 1000, 5000, -3, 640, 257]
        rows = len(ends_l)
        x = Inputs(1, rows, rows, page, 16, 200)
        ends = I32(ends_l)
        for pred in (False, True):
            old = run_dense(OLD, x, ends, width, pred)
            self.assertFalse(bool(torch.isnan(old).any()))
            for mod in (NEW, NEW_OFF, NEW_NOCAND):
                got = run_dense(mod, x, ends, width, pred)
                self.assertFalse(bool(torch.isnan(got).any()), f"{mod.__name__}: unwritten columns")
                self.assertTrue(same_bits(old, got), f"{mod.__name__} pred={pred} differs from OLD")
            for r, e in enumerate(ends_l):  # the tail past the context is -inf in both
                self.assertTrue(bool(torch.all(torch.isneginf(got[r, max(0, min(e, width)):]))))
        check_against_ref(self, got, ref_dense_decode(x, ends_l, width))

    def test_tile_program_counts_and_tail_loops(self):
        width, page = 4096 + 7, 128
        ends_l = [4103, 4000, 2049, 300, 0, 17]
        x = Inputs(2, len(ends_l), len(ends_l), page, 33, 300)
        ends = I32(ends_l)
        old = run_dense(OLD, x, ends, width)
        for target, minimum in ((1, 1), (5, 1), (64, 8), (7000, 8), (10**9, 8)):
            with TargetPrograms(NEW, target, minimum):
                got = run_dense(NEW, x, ends, width)
            self.assertTrue(same_bits(old, got), f"target={target} min={minimum}")
        check_against_ref(self, got, ref_dense_decode(x, ends_l, width))

    def test_row_repeat_and_tiny_widths(self):
        page = 64
        x = Inputs(3, 6, 3, page, 4, 40)
        ends = I32([200, 199, 1, 0, 256, 255])
        for width in (1, 2, 16, 17, 255, 256):
            old = run_dense(OLD, x, ends, width, row_repeat=2)
            got = run_dense(NEW, x, ends, width, row_repeat=2)
            self.assertTrue(same_bits(old, got), f"width={width}")
        check_against_ref(self, got, ref_dense_decode(x, ends.tolist(), 256, row_repeat=2))

    def test_grid_ignores_row_ends(self):
        # The launch grid is all a CUDA graph freezes: it must not depend on row_ends.
        launches = []

        class Fake:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append((grid, kwargs))
                return launch

        x = Inputs(4, 3, 3, 64, 4, 20)
        saved = NEW._mxfp4_indexer_logits_stride_kernel, OLD._mxfp4_indexer_logits_kernel
        NEW._mxfp4_indexer_logits_stride_kernel = Fake()
        try:
            for ends in ([0, 0, 0], [100, 7, 50], [5000, 5000, 5000]):
                run_dense(NEW, x, I32(ends), 81_920)
        finally:
            NEW._mxfp4_indexer_logits_stride_kernel = saved[0]
        grids = {g for g, _ in launches}
        self.assertEqual(len(grids), 1)
        self.assertEqual(grids.pop(), (3, 5120))  # min(5120 tiles, 16384 // 3)
        self.assertTrue(all("SWAP_GRID_AXES" not in kw for _, kw in launches))


class StrideGrid(unittest.TestCase):
    def test_pr_grid_expectations(self):  # the PR's test_dense_launcher_grid_is_rows_by_tile_programs
        for width, grid in ((100, (3, 7)), (65_535 * 16 + 1, (3, 16_384 // 3)), (1024 * 1024, (3, 16_384 // 3))):
            self.assertEqual(NEW._stride_launch_grid(3, width), grid)

    def test_e128_shapes(self):
        for rows in (1, 36, 40, 48, 64, 2048, 100_000):
            for width in (81_920, 163_840, 16_384, 1):
                r, p = NEW._stride_launch_grid(rows, width)
                self.assertEqual(r, rows)
                self.assertTrue(1 <= p <= 65_535)
                self.assertLessEqual(p, -(-width // 16))
        self.assertEqual(NEW._stride_launch_grid(36, 81_920), (36, 455))
        self.assertEqual(NEW._stride_launch_grid(64, 163_840), (64, 256))

    def test_old_path_keeps_swap(self):
        launches = []

        class Fake:
            def __getitem__(self, grid):
                return lambda *a, **kw: launches.append((grid, kw))

        x = Inputs(5, 2, 2, 64, 4, 10)
        saved = NEW_OFF._mxfp4_indexer_logits_kernel
        NEW_OFF._mxfp4_indexer_logits_kernel = Fake()
        try:
            run_dense(NEW_OFF, x, I32([10, 10]), 65_535 * 16 + 1)
            run_dense(NEW_OFF, x, I32([10, 10]), 1000)
        finally:
            NEW_OFF._mxfp4_indexer_logits_kernel = saved
        self.assertEqual(launches[0][0], (65_536, 2))
        self.assertIs(launches[0][1]["SWAP_GRID_AXES"], True)
        self.assertEqual(launches[1][0], (2, 63))
        self.assertIs(launches[1][1]["SWAP_GRID_AXES"], False)


class DensePrefill(unittest.TestCase):
    def test_packed_two_requests(self):
        page = 64
        lens = [250, 350]
        width = sum(lens)  # packed: req0 at [0, 250), req1 at [250, 600)
        cu = I32([0, 250, 600])
        t2s = I32([0] * 250 + [1] * 350)
        starts_l = [0, 0, 0, 250, 250, 250, 250, 0, 250]
        ends_l = [1, 130, 250, 251, 333, 600, 250, 0, 700]  # incl. empty and > width rows
        x = Inputs(6, len(ends_l), 2, page, 8, 50)
        prefill = (I32(starts_l), cu, t2s)
        ends = I32(ends_l)
        for pred in (False, True):
            old = run_dense(OLD, x, ends, width, pred, prefill)
            for mod in (NEW, NEW_OFF):
                got = run_dense(mod, x, ends, width, pred, prefill)
                self.assertFalse(bool(torch.isnan(got).any()))
                self.assertTrue(same_bits(old, got), f"{mod.__name__} pred={pred}")
        for target in (1, 3, 40):
            with TargetPrograms(NEW, target, 1):
                self.assertTrue(same_bits(old, run_dense(NEW, x, ends, width, True, prefill)))
        check_against_ref(self, got, ref_dense_prefill(x, starts_l, ends_l, cu, t2s, width))


class Candidates(unittest.TestCase):
    def _cand(self, seed, rows, nblk, hi):
        g = torch.Generator().manual_seed(seed)
        c = torch.randint(0, hi, (rows, nblk), dtype=torch.int32, generator=g)
        c[torch.rand(c.shape, generator=g) < 0.2] = -1
        return c

    def test_decode_candidates(self):
        page = 64
        ends_l = [300, 64, 0, 1, 511, 512]
        x = Inputs(7, len(ends_l), len(ends_l), page, 8, 60)
        cand = self._cand(8, len(ends_l), 20, 64)  # 8-token candidate blocks, some past the end
        ends = I32(ends_l)
        old = run_cand(OLD, x, ends, cand)
        for mod in (NEW, NEW_OFF, NEW_NOCAND):
            got = run_cand(mod, x, ends, cand)
            self.assertFalse(bool(torch.isnan(got).any()))
            self.assertTrue(same_bits(old, got), mod.__name__)
        with TargetPrograms(NEW, 2, 1):
            self.assertTrue(same_bits(old, run_cand(NEW, x, ends, cand)))
        check_against_ref(self, got, ref_candidates(x, cand, ends_l, req_of_row=lambda r: r))

    def test_prefill_candidates(self):
        page = 64
        cu = I32([0, 100, 400])
        t2s = I32([0] * 100 + [1] * 300)
        starts_l = [0, 0, 100, 100, 100]
        ends_l = [50, 100, 101, 400, 100]
        x = Inputs(9, len(ends_l), 2, page, 8, 30)
        cand = self._cand(10, len(ends_l), 12, 40)
        prefill = (I32(starts_l), cu, t2s)
        ends = I32(ends_l)
        old = run_cand(OLD, x, ends, cand, True, prefill)
        got = run_cand(NEW, x, ends, cand, True, prefill)
        self.assertTrue(same_bits(old, got))
        req_of_row = lambda r: int(t2s[starts_l[r]]) if ends_l[r] > starts_l[r] else 0  # noqa: E731
        check_against_ref(self, got, ref_candidates(x, cand, ends_l, starts_l, req_of_row))


if __name__ == "__main__":
    unittest.main(verbosity=2)
