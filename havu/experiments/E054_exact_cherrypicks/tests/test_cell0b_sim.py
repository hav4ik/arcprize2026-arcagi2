# SPDX-License-Identifier: Apache-2.0
"""E054 CPU simulation of the Kaggle packaging: cell 0b puts the E054 files into the kit overlay, then cell 2's
own install code (copied verbatim from the V5/V6 notebook: extract KIT/vllm_overlay_rw.tgz into site-packages,
check every sha line) must pass and leave the E054 versions installed. Covers both packaging routes:
  A. the pinned-file route: the E054 kit pair (kit/vllm_overlay_rw.{tgz,sha256}) replaces the kit's pair;
  B. scripts/cell0b_overlay.py: the kit's tarball is rebuilt with the 2 members of the addition tarball;
and the rerun fallback (the *.v3orig copies restored -> cell 2 installs the kit's own versions).
Stdlib + no vllm import; runs in a few seconds.
"""

import base64
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
E054 = HERE.parent
ROOT = E054.parents[1]
KIT_SRC = next((d for d in (Path(os.environ.get("E054_KIT_DIR", ROOT / "kaggle/dsv41_tools/kit")), Path("/root/kit"))
                if (d / "vllm_overlay_rw.tgz").is_file()), None)
sys.path.insert(0, str(E054 / "scripts"))
import cell0b_overlay  # noqa: E402

TARGETS = ("vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py",
           "vllm/models/deepseek_v4_1/common/engram.py")


def cell2_install(KIT, SP):
    """Verbatim from the notebook's cell 2 (the overlay part)."""
    with tarfile.open(f"{KIT}/vllm_overlay_rw.tgz") as tf:
        tf.extractall(SP)
    for line in open(f"{KIT}/vllm_overlay_rw.sha256"):
        h, rel = line.split()
        assert hashlib.sha256(open(f"{SP}/{rel}", "rb").read()).hexdigest() == h, f"overlay mismatch: {rel}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@unittest.skipIf(KIT_SRC is None, "no kit vllm_overlay_rw.tgz (set E054_KIT_DIR)")
class Cell0bSim(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="e054_cell0b_"))
        self.kit = self.tmp / "kit"
        self.kit.mkdir()
        for f in ("vllm_overlay_rw.tgz", "vllm_overlay_rw.sha256"):
            shutil.copy2(KIT_SRC / f, self.kit / f)
        self.e054 = {rel: sha(E054 / "overlay" / rel) for rel in TARGETS}
        self.base = {ln.split()[1]: ln.split()[0] for ln in (KIT_SRC / "vllm_overlay_rw.sha256").read_text().splitlines()}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def installed(self, sp):
        return {rel: sha(sp / rel) for rel in TARGETS}

    def test_route_a_pinned_pair(self):
        for f in ("vllm_overlay_rw.tgz", "vllm_overlay_rw.sha256"):
            shutil.copy2(self.kit / f, self.kit / f"{f}.v3orig")  # cell 0b's backup
            shutil.copy2(E054 / "kit" / f, self.kit / f)
        sp = self.tmp / "sp"
        cell2_install(self.kit, sp)
        self.assertEqual(self.installed(sp), self.e054)
        self.assertEqual(len(list(sp.rglob("*.py"))), 12)

    def test_route_b_rebuild_from_addition(self):
        add = (E054 / "kit/vllm_overlay_e054_add.tgz").read_bytes()
        add_sha = {ln.split()[1]: ln.split()[0]
                   for ln in (E054 / "kit/vllm_overlay_e054_add.sha256").read_text().splitlines()}
        # as embedded in a notebook cell: base64 round trip
        got = cell0b_overlay.e054_overlay_into_kit(str(self.kit), base64.b64decode(base64.b64encode(add)), add_sha)
        self.assertEqual(set(got), set(TARGETS))
        sp = self.tmp / "sp"
        cell2_install(self.kit, sp)
        self.assertEqual(self.installed(sp), self.e054)
        others = [ln.split()[1] for ln in (self.kit / "vllm_overlay_rw.sha256").read_text().splitlines()]
        self.assertEqual(others, list(self.base))  # same 12 paths, same order
        for rel in others:
            if rel not in TARGETS:
                self.assertEqual(sha(sp / rel), self.base[rel])
        # a second run is refused (the kit tarball is no longer the base version), state unchanged
        with self.assertRaises(AssertionError):
            cell0b_overlay.e054_overlay_into_kit(str(self.kit), add, add_sha)
        # rerun fallback: restore, then cell 2 installs the kit's own files
        cell0b_overlay.restore_kit_overlay(str(self.kit))
        sp2 = self.tmp / "sp2"
        cell2_install(self.kit, sp2)
        self.assertEqual(self.installed(sp2), {rel: self.base[rel] for rel in TARGETS})
        self.assertEqual(sha(self.kit / "vllm_overlay_rw.tgz"), cell0b_overlay.KIT_OVERLAY_SHA)

    def test_route_b_refuses_corrupt_member(self):
        add = (E054 / "kit/vllm_overlay_e054_add.tgz").read_bytes()
        bad = {rel: "0" * 64 for rel in TARGETS}
        with self.assertRaises(AssertionError):
            cell0b_overlay.e054_overlay_into_kit(str(self.kit), add, bad)
        self.assertEqual(sha(self.kit / "vllm_overlay_rw.tgz"), cell0b_overlay.KIT_OVERLAY_SHA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
