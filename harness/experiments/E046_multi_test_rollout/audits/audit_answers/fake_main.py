"""Drive solver_dsv41.main() (pinned v6) with a fake run_attempt and a fake server process: no GPU, no network."""
import json
import os
import sys
import threading
from pathlib import Path

SCR = Path(__file__).resolve().parent
W = SCR / "fake_run"
os.system(f"rm -rf {W}")
(W / "model").mkdir(parents=True)
(W / "model" / "config.json").write_text("{}")

G = lambda v: [[v]]  # 1x1 grids
CH = {
    "m3": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}, {"input": [[2]]}, {"input": [[3]]}]},
    "m2": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[4]]}, {"input": [[5]]}]},
    "s1": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[6]]}]},
    "rq": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[7]]}]},
    "er": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[8]]}]},
    "ep": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[9]]}]},
    "un": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[0]]}, {"input": [[1, 1]]}]},
}
SOL = {"m3": [G(1), G(2), G(3)], "m2": [G(4), G(5)], "s1": [G(6)], "rq": [G(7)], "er": [G(8)], "ep": [G(9)],
       "un": [G(0), [[1, 1]]]}
# make solutions distinct from the test inputs (so a fallback copy is never correct)
SOL = {t: [[[v + 1 if isinstance(v, int) else v for v in row] for row in g] for g in gs] for t, gs in SOL.items()}
json.dump(CH, open(W / "ch.json", "w"))
json.dump(SOL, open(W / "sol.json", "w"))
os.environ.update(ARC_OUT=str(W / "out"), ARC_CHALLENGES=str(W / "ch.json"), ARC_SOLUTIONS=str(W / "sol.json"),
                  ARC_KIT=str(SCR / "kit"), ARC_MODEL_DIR=str(W / "model"), ARC_WORKERS="3", ARC_MULTI_TEST="1",
                  ARC_MULTI_BUDGET="0.5", ARC_ATTEMPTS="2", ARC_TASK_IDS="m3,m2,s1,rq,er,ep", ARC_T0="")
sys.path.insert(0, str(SCR / "kit"))
import solver_dsv41 as S  # noqa: E402

S.LOG.mkdir(parents=True, exist_ok=True)


class FakeProc:
    def __init__(self):
        self.dead = False
        self.returncode = None
        self.pid = 999999

    def poll(self):
        return -9 if self.dead else None


STARTS = []


def start_server():
    STARTS.append(S.elapsed())
    S.STATE["server"] = FakeProc()
    S.STATE["server_ok"] = True
    return True


def stop_server():
    S.STATE["server_ok"] = False


CALLS = {}
LOCK = threading.Lock()


def wrong(g):
    return [[(g[0][0] + 5) % 10]]


def run_attempt(args, task, t, p, a):
    with LOCK:
        k = (t, p, a)
        CALLS[k] = CALLS.get(k, 0) + 1
        n = CALLS[k]
    base = {"task_id": t, "pair": p, "attempt": a, "wall_s": 1.0, "cuts": {}, "usage": {"completion": 100},
            "stop_reason": "submitted", "turns": 3}
    sol = SOL[t]
    if t == "m3":
        # pair 1 correct only in attempt 1; pair 0 correct only in attempt 0; pair 2 correct in both
        pred = [sol[0], wrong(sol[1]), sol[2]] if a == 0 else [wrong(sol[0]), sol[1], sol[2]]
        return dict(base, pred=pred, multi_test=3, train_pass=False)
    if t == "m2":
        if a == 0:  # cut attempt, session answer for both pairs
            return dict(base, pred=[sol[0], sol[1]], multi_test=2, train_pass=None, cuts={"deadline": 1},
                        stop_reason="output_budget", answer_source="session")
        if n == 1:  # server death: a connection error, the server is marked dead now
            S.STATE["server"].dead = True
            return dict(base, pred=None, multi_test=2, error="TurnError: APIConnectionError: Connection error.",
                        stop_reason="error")
        return dict(base, pred=None, multi_test=2, train_pass=None, stop_reason="max_turns")
    if t == "s1":
        return dict(base, pred=wrong(sol[0]), train_pass=False) if a == 0 else dict(base, pred=sol[0], train_pass=True)
    if t == "rq":
        if a == 0 and n == 1:
            return dict(base, pred=None, error="ConnectError: [Errno 111] Connection refused", stop_reason="error")
        return dict(base, pred=sol[0] if a == 0 else wrong(sol[0]), train_pass=a == 0)
    if t == "er":
        return dict(base, pred=None, error="RuntimeError: Provider interrupted generation: abort", stop_reason="error")
    if t == "ep":
        if a == 0:
            return dict(base, pred=sol[0], error="OSError: session close failed", stop_reason="error")
        raise ValueError("driver boom")  # fut.result() raises: driver error record
    raise AssertionError(t)


S.start_server, S.stop_server, S.run_attempt = start_server, stop_server, run_attempt
S.server_metrics = lambda: None
S.os._exit = lambda code: (_ for _ in ()).throw(SystemExit(code))
try:
    S.main()
except SystemExit:
    pass
sub = json.load(open(W / "out" / "submission.json"))
print("server starts:", len(STARTS), "| calls:", {f"{k[0]}.{k[1]}#{k[2]}": v for k, v in sorted(CALLS.items())})
print("RECORDS per pair:", {f"{t}.{p}": [(r["attempt"], r.get("pred"), r.get("train_pass")) for r in rs]
                             for (t, p), rs in sorted(S.RECORDS.items())})
ok = 0
for t in CH:
    for p, row in enumerate(sub[t]):
        a1, a2 = row["attempt_1"] == SOL[t][p], row["attempt_2"] == SOL[t][p]
        fb = CH[t]["test"][p]["input"]
        ok += a1 or a2
        print(f"  {t}.{p}: a1={row['attempt_1']} a2={row['attempt_2']} sol={SOL[t][p]} -> a1_ok={a1} a2_ok={a2} "
              f"a2_fallback={row['attempt_2'] == fb}")
print("score", ok, "/", sum(len(x["test"]) for x in CH.values()))
print(open(W / "out" / "dsv41_logs" / "solver.log").read()[-2500:])
