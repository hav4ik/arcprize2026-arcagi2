import json, sys
REPO = "/home/vu/Workspace/ARC-AGI-2"
sys.path[:0] = ["/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_answers/kit/repo/evals"]
import run_deepseek_tools as h
from arc_harness.executor import LocalExecutor
train = [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}]
tis = [[[1, 2], [3, 4]], [[5]], [[1, 1, 1]]]
pre = f"\nimport json as _arc_json\ntest_inputs = _arc_json.loads({json.dumps(tis)!r})\n"
cases_multi = [
    ("list of np arrays", "answer = [np.array(t) for t in test_inputs]", "answer"),
    ("list of lists of np ints", "answer = [[[np.int64(v) for v in r] for r in t] for t in test_inputs]", "answer"),
    ("tuple of tuples", "answer = tuple(tuple(tuple(r) for r in t) for t in test_inputs)", "answer"),
    ("np 3D object array", "answer = np.array([np.array(t) for t in test_inputs], dtype=object)", "answer"),
    ("list of np rows", "answer = [[np.array(r) for r in t] for t in test_inputs]", "answer"),
    ("n-1 grids", "answer = [t for t in test_inputs[:2]]", "answer"),
    ("n+1 grids", "answer = test_inputs + [[[0]]]", "answer"),
    ("one grid", "answer = test_inputs[0]", "answer"),
    ("2 of 3 valid (None)", "answer = [test_inputs[0], None, test_inputs[2]]", "answer"),
    ("2 of 3 valid (empty)", "answer = [test_inputs[0], [], test_inputs[2]]", "answer"),
    ("float grid", "answer = [np.array(t, dtype=float) for t in test_inputs]", "answer"),
    ("bool grid", "answer = [np.array(t) > 0 for t in test_inputs]", "answer"),
    ("value 10", "answer = [[[10]], [[5]], [[1]]]", "answer"),
    ("order kept", "answer = [[[7]], [[8]], [[9]]]", "answer"),
    ("expr list comp", "def solve(g):\n    return np.array(g)[::-1]\n", "[solve(t) for t in test_inputs]"),
]
with LocalExecutor(train, tis[0], 30, 16000, memory_mb=2048, python=REPO + "/.venv/bin/python", prelude_extra=pre,
                   style="ipython", cpu_seconds=600) as s:
    for name, setup, expr in cases_multi:
        s.execute("for _k in ('answer','solve'):\n    globals().pop(_k, None)")
        s.execute(setup)
        out, grid = h.submit_answer(s, expr, 3)
        print(f"MULTI {name:28s} -> {'ACCEPT ' + json.dumps(grid) if grid is not None else 'reject'} | {out[:110]}")
    for name, setup in [("1x1", "answer = [[3]]"), ("empty []", "answer = []"), ("[[]]", "answer = [[]]"),
                        ("np int64 2D", "answer = np.array([[1,2],[3,4]], dtype=np.int64)"),
                        ("list of np rows", "answer = [np.array([1,2]), np.array([3,4])]"),
                        ("list of np int scalars", "answer = [[np.int8(1), np.uint8(2)]]"),
                        ("ragged", "answer = [[1,2],[3]]"), ("31 rows", "answer = [[0]]*31"),
                        ("list of one grid", "answer = [[[1,2],[3,4]]]")]:
        s.execute("globals().pop('answer', None)")
        s.execute(setup)
        out, grid = h.submit_answer(s, "answer")
        print(f"SINGLE {name:27s} -> {'ACCEPT ' + json.dumps(grid) if grid is not None else 'reject'} | {out[:110]}")
