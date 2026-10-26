"""Tests for evals/auto_repair.py (retry-once repair of identifier typos, E043 typo analysis).

  .venv/bin/python -m unittest evals/tests/test_auto_repair.py

The fixture holds real failing cells of the E128 / E112 / E96 tool evals and dev runs:
- what the replay study (experiments/E043_dsv41_real_pruning/analysis/typos/replay.py) proposed for each;
- the module's output when the fixture was made (validate_auto_repair.py --fixture).
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from auto_repair import parse_error, propose, ran_side_effects, rewrite, suggest  # noqa: E402
from arc_harness.executor import LocalExecutor  # noqa: E402


def tb(line, err):
    """A tool output in the sandbox's IPython plain style (cell frames: source line only)."""
    return f"[error]\nTraceback (most recent call last):\n    {line}\n{err}\n"


NAME_ERR = "NameError: name 'coms' is not defined"


class ParseTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(parse_error(tb("x", NAME_ERR)), ("coms", "name", None))
        self.assertEqual(parse_error(tb("x", "AttributeError: module 'numpy' has no attribute 'arnge'")),
                         ("arnge", "attr", ("module", "numpy")))
        self.assertEqual(parse_error(tb("x", "AttributeError: 'collections.deque' object has no attribute 'popleift'")),
                         ("popleift", "attr", ("type", "collections.deque")))
        self.assertEqual(parse_error(tb("x", "ModuleNotFoundError: No module named 'scpy'")), ("scpy", "module", None))
        self.assertEqual(parse_error(tb("x", "ImportError: cannot import name 'lable' from 'scipy.ndimage' (/x.py)")),
                         ("lable", "import", "scipy.ndimage"))
        self.assertEqual(parse_error(tb("x", "TypeError: zeros_like() got an unexpected keyword argument 'd'")),
                         ("d", "kwarg", "zeros_like"))

    def test_other_errors_ignored(self):
        for err in ("IndexError: index 5 is out of bounds", "TypeError: 'int' object is not iterable",
                    "ValueError: x", "SyntaxError: invalid syntax", ""):
            self.assertIsNone(parse_error(tb("x", err)), err)


class ProposeTest(unittest.TestCase):
    def test_name(self):
        code = "comps = [1, 2]\nfor c in coms:\n    print(c)\n"
        new, note = propose(code, tb("for c in coms:", NAME_ERR), {"np"})
        self.assertEqual(new, "comps = [1, 2]\nfor c in comps:\n    print(c)\n")
        self.assertEqual(note, "[auto-corrected: coms -> comps]")

    def test_session_name_and_builtin(self):
        self.assertEqual(propose("print(coms)\n", tb("print(coms)", NAME_ERR), {"comps"})[0], "print(comps)\n")
        self.assertEqual(propose("for a, b in zzip(x, y): pass\n",
                                 tb("for a, b in zzip(x, y): pass", "NameError: name 'zzip' is not defined"),
                                 {"x", "y"})[0], "for a, b in zip(x, y): pass\n")

    def test_strings_and_comments_untouched(self):
        new, _ = propose("print('coms', coms)  # coms\n", tb("print('coms', coms)  # coms", NAME_ERR), {"comps"})
        self.assertEqual(new, "print('coms', comps)  # coms\n")

    def test_module_type_import_kwarg(self):
        self.assertEqual(propose("x = np.arnge(5)\n",
                                 tb("x = np.arnge(5)", "AttributeError: module 'numpy' has no attribute 'arnge'"),
                                 {"np"})[0], "x = np.arange(5)\n")
        code = "from collections import Counter\ncnt = Counter([1, 1, 2])\nB = cnt.mmost_common(1)\n"
        new, note = propose(code, tb("B = cnt.mmost_common(1)",
                                     "AttributeError: 'Counter' object has no attribute 'mmost_common'"), set())
        self.assertIn("cnt.most_common(1)", new)
        self.assertEqual(note, "[auto-corrected: mmost_common -> most_common]")
        self.assertEqual(propose("import scpy.ndimage as nd\n", tb("import scpy.ndimage as nd",
                                                                   "ModuleNotFoundError: No module named 'scpy'"),
                                 set())[0], "import scipy.ndimage as nd\n")
        self.assertEqual(propose("from collections import dequeue\n",
                                 tb("from collections import dequeue",
                                    "ImportError: cannot import name 'dequeue' from 'collections' (/x.py)"),
                                 set())[0], "from collections import deque\n")
        self.assertEqual(propose("seen = np.zeros_like(a, d=bool)\n",
                                 tb("seen = np.zeros_like(a, d=bool)",
                                    "TypeError: zeros_like() got an unexpected keyword argument 'd'"),
                                 {"a", "np"})[0], "seen = np.zeros_like(a, dtype=bool)\n")

    def test_no_unique_match(self):
        # two candidates at the same distance and rank: ambiguous
        self.assertIsNone(propose("print(cat)\n", tb("print(cat)", "NameError: name 'cat' is not defined"),
                                  {"cab", "car"}))
        # names of <= 4 characters: distance 1 only
        self.assertIsNone(propose("print(abcd)\n", tb("print(abcd)", "NameError: name 'abcd' is not defined"),
                                  {"axcy"}))
        # too far
        self.assertIsNone(propose("print(components)\n", tb("print(components)",
                                                            "NameError: name 'components' is not defined"),
                                  {"comps"}))

    def test_typo_not_in_cell(self):
        # the misspelled name is inside a function defined in an earlier cell
        self.assertIsNone(propose("print(f(1))\n", tb("print(f(1))", NAME_ERR), {"f", "comps"}))

    def test_rewrite_kinds(self):
        self.assertEqual(rewrite("a.coms + coms", "coms", "comps", "name"), "a.coms + comps")
        self.assertEqual(rewrite("a.coms + coms", "coms", "comps", "attr"), "a.comps + coms")
        self.assertEqual(rewrite("f(d=1, x=d)", "d", "dtype", "kwarg"), "f(dtype=1, x=d)")

    def test_dtype_restart(self):
        # `,d` + `bool`: the model restarted `dtype=bool` after the comma
        err = tb("m = np.zeros((3, 3), dbool)", "NameError: name 'dbool' is not defined")
        self.assertEqual(propose("m = np.zeros((3, 3), dbool)\n", err, {"np"}),
                         ("m = np.zeros((3, 3), dtype=bool)\n", "[auto-corrected: dbool -> dtype=bool]"))
        err = tb("m = np.full((3, 3), 0, dint)", "NameError: name 'dint' is not defined")
        self.assertEqual(propose("m = np.full((3, 3), 0, dint)\n", err, {"np"})[0], "m = np.full((3, 3), 0, dtype=int)\n")
        # not a positional argument: the plain close-name rule
        self.assertEqual(propose("x = dbool\n", tb("x = dbool", "NameError: name 'dbool' is not defined"), set())[0],
                         "x = bool\n")

    def test_short_names_skipped(self):
        err = tb("print(deq)", "NameError: name 'deq' is not defined")
        self.assertIsNone(propose("print(deq)\n", err, {"dq"}))
        self.assertEqual(propose("print(deq)\n", err, {"dq"}, min_len=1)[0], "print(dq)\n")


class SideEffectTest(unittest.TestCase):
    def blocked(self, code, line, session=("comps",), strict=False):
        return propose(code, tb(line, NAME_ERR), set(session), strict=strict) is None

    def test_mutating_session_objects(self):
        self.assertTrue(self.blocked("res.append(1)\nprint(coms)\n", "print(coms)", ("comps", "res")))
        self.assertTrue(self.blocked("grid[0][0] = 5\nprint(coms)\n", "print(coms)", ("comps", "grid")))
        self.assertTrue(self.blocked("total += 1\nprint(coms)\n", "print(coms)", ("comps", "total")))
        self.assertTrue(self.blocked("x = x + 1\nprint(coms)\n", "print(coms)", ("comps", "x")))
        self.assertTrue(self.blocked("del cache[3]\nprint(coms)\n", "print(coms)", ("comps", "cache")))
        self.assertTrue(self.blocked("heapq.heappush(h, 1)\nprint(coms)\n", "print(coms)", ("comps", "h")))
        self.assertTrue(self.blocked("for p in train:\n    g = p['input']\n    g[0][0] = 5\n    print(coms)\n",
                                     "print(coms)", ("comps", "train")))

    def test_objects_created_in_the_cell(self):
        self.assertFalse(self.blocked("res = []\nres.append(1)\nprint(coms)\n", "print(coms)"))
        self.assertFalse(self.blocked("g = np.array(train[0]['input'])\ng[0, 0] = 5\nprint(coms)\n", "print(coms)",
                                      ("comps", "train", "np")))
        self.assertFalse(self.blocked("total = 0\nfor v in [1, 2]:\n    total += v\nprint(coms)\n", "print(coms)"))
        self.assertFalse(self.blocked("q = dequeue if False else deque([1])\nq.append(2)\nprint(coms)\n",
                                      "print(coms)", ("comps", "deque")))

    def test_only_the_part_that_ran(self):
        # the mutation comes after the failing statement: it never ran
        self.assertFalse(self.blocked("print(coms)\nres.append(1)\n", "print(coms)", ("comps", "res")))
        # plain-style traceback with a line number
        err = ("Traceback (most recent call last):\n  File \"<cell>\", line 1, in <module>\n    print(coms)\n"
               + NAME_ERR + "\n")
        self.assertIsNotNone(propose("print(coms)\nres.append(1)\n", err, {"comps", "res"}))
        # not located: the whole cell counts as executed
        self.assertIsNone(propose("print(coms)\nres.append(1)\n", NAME_ERR, {"comps", "res"}))

    def test_cell_functions(self):
        self.assertTrue(self.blocked("def f():\n    res.append(1)\nf()\nprint(coms)\n", "print(coms)",
                                     ("comps", "res")))
        self.assertFalse(self.blocked("res = []\ndef f():\n    res.append(1)\nf()\nprint(coms)\n", "print(coms)"))
        self.assertTrue(self.blocked("def paint(g):\n    g[0][0] = 1\npaint(test_input)\nprint(coms)\n",
                                     "print(coms)", ("comps", "test_input")))
        self.assertFalse(self.blocked("def paint(g):\n    g[0][0] = 1\npaint([[0]])\nprint(coms)\n", "print(coms)"))
        self.assertTrue(self.blocked("def f():\n    global n\n    n = 1\nf()\nprint(coms)\n", "print(coms)"))

    def test_os_and_files(self):
        self.assertTrue(self.blocked("open('x.txt', 'w').write('a')\nprint(coms)\n", "print(coms)"))
        self.assertTrue(self.blocked("os.remove('x')\nprint(coms)\n", "print(coms)"))
        self.assertTrue(self.blocked("exec('a = 1')\nprint(coms)\n", "print(coms)"))
        self.assertFalse(self.blocked("t = open('x.txt').read()\nprint(coms)\n", "print(coms)"))

    def test_strict_session_functions(self):
        code = "out = solve(test_input)\nprint(coms)\n"
        self.assertFalse(self.blocked(code, "print(coms)", ("comps", "solve", "test_input")))
        self.assertTrue(self.blocked(code, "print(coms)", ("comps", "solve", "test_input"), strict=True))

    def test_api(self):
        self.assertTrue(ran_side_effects("res.append(1)\n", "", {"res"}))
        self.assertFalse(ran_side_effects("res = [1]\n", "", {"res"}))


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "fixtures", "auto_repair_cases.json")) as f:
            cls.cases = json.load(f)

    def test_matches_replay_study(self):
        n = 0
        for c in self.cases:
            if c["replay"]:
                s = suggest(c["code"], c["error_text"], c["session_names"], min_len=1, dtype_rule=False)
                self.assertEqual(list(s[:2]) if s else None, c["replay"], c["key"])
                n += 1
        self.assertGreaterEqual(n, 25)

    def test_regression(self):
        for c in self.cases:
            p = propose(c["code"], c["error_text"], c["session_names"])
            self.assertEqual(list(p) if p else None, c["proposed"], c["key"])


class ExecutorTest(unittest.TestCase):
    """End to end: the cell fails in a real session, the repaired cell runs once and completes."""

    def run_style(self, style):
        code = "comps = [3, 4]\nres = []\nfor c in coms:\n    res.append(c)\nprint(sum(res))\n"
        with LocalExecutor([{"input": [[0]], "output": [[0]]}], [[0]], timeout=20, style=style) as ex:
            r = ex.execute(code)
            out = (r.get("stdout") or "") + "\n" + (r.get("stderr") or "")
            self.assertIn("NameError", out)
            new, note = propose(code, out, {"train", "test_input", "np"})
            self.assertEqual(note, "[auto-corrected: coms -> comps]")
            r2 = ex.execute(new)
            self.assertIn("7", r2.get("stdout") or "")
            self.assertNotIn("Error", (r2.get("stdout") or "") + (r2.get("stderr") or ""))

    def test_plain(self):
        self.run_style("plain")

    def test_ipython(self):
        self.run_style("ipython")


if __name__ == "__main__":
    unittest.main()
