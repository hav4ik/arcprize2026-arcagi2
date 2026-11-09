"""gpu_hold.py without a GPU: the calibration maths (B32-M numbers), cap parsing, and the hold with a fake
libcudart."""
import io
import unittest
from contextlib import redirect_stdout

from kit_helpers import TmpCase, fixture
import gpu_hold


class FakeCudart:
    def __init__(self, n=4):
        self.n, self.dev, self.mallocs, self.contexts = n, None, [], set()

    def cudaGetDeviceCount(self, p):
        p._obj.value = self.n
        return 0

    def cudaSetDevice(self, d):
        self.dev = d
        return 0

    def cudaFree(self, p):
        self.contexts.add(self.dev)
        return 0

    def cudaMalloc(self, pp, size):
        self.mallocs.append((self.dev, size.value >> 20))
        pp._obj.value = 0x1000 + len(self.mallocs)
        return 0

    def cudaMemset(self, p, v, n):
        return 0


class Calibrate(TmpCase):
    def test_b32m_try1_suggests_about_350(self):
        """Try 1 held 410 MiB: pod cap - reserved 0.66/0.64/0.61/0.61 vs Kaggle 0.73/0.71/0.67/0.67; try 2's 350
        MiB then matched Kaggle within 0.01 GiB."""
        pod = {"TP0": 0.66, "TP1": 0.64, "TP2": 0.61, "TP3": 0.61}
        kag = {"TP0": 0.73, "TP1": 0.71, "TP2": 0.67, "TP3": 0.67}
        new, mean, rows = gpu_hold.suggest(410, pod, kag)
        self.assertAlmostEqual(mean, -0.065, places=3)
        self.assertEqual(new, 340)
        self.assertEqual(len(rows), 4)
        new, mean, _ = gpu_hold.suggest(0, {"TP0": 1.2, "TP1": 1.16}, gpu_hold.parse_kaggle("0.67-0.73"))
        self.assertEqual(new, 490)  # an unheld pod with ~0.48 GiB more headroom than Kaggle

    def test_parse_kaggle(self):
        self.assertEqual(gpu_hold.parse_kaggle("0.67-0.73"), {"*": 0.7})
        self.assertEqual(gpu_hold.parse_kaggle("0.70"), {"*": 0.7})
        self.assertEqual(gpu_hold.parse_kaggle("TP0=0.73, TP1=0.71"), {"TP0": 0.73, "TP1": 0.71})

    def test_parse_caps_and_main(self):
        caps = gpu_hold.parse_caps(fixture("server_ansi.log"))
        self.assertEqual(caps, {"TP0": 0.72, "TP1": 0.7, "TP2": 0.66, "TP3": 0.66})
        with redirect_stdout(io.StringIO()) as out:
            gpu_hold.main(["calibrate", "--hold-mib", "350", "--pod-log", fixture("server_ansi.log"),
                           "--kaggle", "TP0=0.73,TP1=0.71,TP2=0.67,TP3=0.67"])
        self.assertIn("-> hold 340 MiB/GPU (was 350); within 0.015 GiB: keep the current hold", out.getvalue())

    def test_no_common_rank(self):
        with self.assertRaises(SystemExit):
            gpu_hold.suggest(350, {"TP0": 0.7}, {"TP5": 0.7})


class Hold(unittest.TestCase):
    def test_tops_up_to_the_target(self):
        cu = FakeCudart()
        msgs = []
        alloc, ptrs, ctx = gpu_hold.hold(cu, 350, None, lambda: {0: 186, 1: 186, 2: 186, 3: 186}, None,
                                         msgs.append, settle_s=0)
        self.assertEqual(alloc, {0: 164, 1: 164, 2: 164, 3: 164})
        self.assertEqual(cu.contexts, {0, 1, 2, 3})
        self.assertEqual(len(ptrs), 4)
        self.assertIn("nvidia-smi compute apps", msgs[0])

    def test_board_delta_fallback_and_subset(self):
        cu = FakeCudart()
        boards = iter([{0: 100, 1: 100}, {0: 300, 1: 290}])
        alloc, _, ctx = gpu_hold.hold(cu, 350, [0, 1], lambda: {}, lambda: next(boards), lambda m: None,
                                      settle_s=0)
        self.assertEqual(ctx, {0: 200, 1: 190})
        self.assertEqual(alloc, {0: 150, 1: 160})

    def test_missing_gpu(self):
        with self.assertRaises(SystemExit):
            gpu_hold.hold(FakeCudart(2), 350, [3], None, None, lambda m: None, settle_s=0)


if __name__ == "__main__":
    unittest.main()
