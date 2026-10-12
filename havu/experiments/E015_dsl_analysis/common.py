"""Shared helpers for E015. Imports the vendored arc-dsl from an explicit path (run scripts with python -I)."""
import glob
import json
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VENDOR = os.path.join(HERE, "vendor", "arc-dsl")
ARC1_DIR = "/home/vu/Workspace/auto-neurogolf/ARC-AGI/data"  # fchollet/ARC-AGI @ 399030444e0a (400 training + 400 eval)
ARC2_TRAIN_CH = f"{ROOT}/data/arc-agi_training_challenges.json"
ARC2_TRAIN_SOL = f"{ROOT}/data/arc-agi_training_solutions.json"
ARC2_EVAL_DIR = f"{ROOT}/data/arc-agi-2-github/data/evaluation"


def import_dsl():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import dsl  # noqa: E402
    import solvers  # noqa: E402
    import constants  # noqa: E402
    return dsl, solvers, constants


def load_arc1(split):
    out = {}
    for f in sorted(glob.glob(f"{ARC1_DIR}/{split}/*.json")):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def load_arc2_train():
    ch = json.load(open(ARC2_TRAIN_CH))
    sol = json.load(open(ARC2_TRAIN_SOL))
    out = {}
    for tid, t in ch.items():
        test = [{"input": p["input"], "output": o} for p, o in zip(t["test"], sol[tid])]
        out[tid] = {"train": t["train"], "test": test}
    return out


def load_arc2_eval():
    out = {}
    for f in sorted(glob.glob(f"{ARC2_EVAL_DIR}/*.json")):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def as_tuple(g):
    return tuple(tuple(r) for r in g)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def call_with_timeout(fn, arg, seconds=5):
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(arg)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
