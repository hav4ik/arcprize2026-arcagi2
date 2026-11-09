"""Shared setup for the dsv41 solver and notebook-cell tests (ported from the 2026-10-26 deep audit,
../tests_mtf_v6/common.py). The runtime is Kaggle's for the next candidate: the Kaggle kit of submit Version 5
(../kaggle_kit_v5/dsv41_kit.tgz: harness 1ad5ee24, train_check 44c4d210, solver 62144e4a) with the LIVE solver and
harness copied over it, exactly as the notebook's cell 0b installs them. No GPU, no network, no Kaggle."""
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DSV41 = HERE.parent
ROOT = DSV41.parents[1]
KIT_TGZ = DSV41 / "kaggle_kit_v5/dsv41_kit.tgz"
SOLVER = DSV41 / "solver_dsv41.py"
# the candidate's pinned harness (candidate B, 882c817f), not the live evals/ copy: other sessions keep extending the
# live harness (E058 imports grid_image.deepseek_cell, which the Kaggle kit lacks); ARC_TEST_HARNESS overrides
HARNESS = Path(os.environ.get("ARC_TEST_HARNESS") or DSV41 / "pinned_candb/run_deepseek_tools.py")
SCRATCH = Path(tempfile.mkdtemp(prefix="dsv41_tests_"))


def extract_kit(dest):
    """The Kaggle kit of Version 5 (v3 code) at `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(KIT_TGZ) as tf:
        tf.extractall(dest, filter="data")
    return dest


V3 = extract_kit(SCRATCH / "v3")  # the kit as Kaggle unpacks it (cell 0)
KIT = extract_kit(SCRATCH / "kit")  # + the live solver and harness (cell 0b)
shutil.copy2(SOLVER, KIT / "solver_dsv41.py")
shutil.copy2(HARNESS, KIT / "repo/evals/run_deepseek_tools.py")
EVALS = KIT / "repo/evals"
sys.path.insert(0, str(EVALS))

import run_deepseek_tools as H  # noqa: E402  (live harness, kit modules around it)
from arc_harness.executor import LocalExecutor  # noqa: E402  (kit)


def load_solver(env, name=None, path=SOLVER):
    """Import the live solver as a fresh module with `env` (dict; None values = unset) applied at import."""
    base = {"ARC_OUT": tempfile.mkdtemp(prefix="solver_out_", dir=SCRATCH), "ARC_KIT": str(KIT)}
    saved = {k: os.environ.get(k) for k in set(base) | set(env) | {k for k in os.environ if k.startswith("ARC_")}}
    try:
        for k in list(os.environ):
            if k.startswith("ARC_"):
                del os.environ[k]
        for k, v in {**base, **env}.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        spec = importlib.util.spec_from_file_location(name or f"solver_{abs(hash(json.dumps(env, sort_keys=True)))}",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        (Path(os.environ["ARC_OUT"]) / "dsv41_logs" / "traces").mkdir(parents=True, exist_ok=True)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def local_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    """The solver's session_factory (LocalExecutor, ipython style), with a short timeout for tests."""
    return LocalExecutor(train, test_input, min(timeout, 20), max_output, memory_mb=2048, python=sys.executable,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=600)


def fake_bin(names=("nvidia-smi",)):
    """A PATH directory with a fake nvidia-smi (4 GPUs, 0 MiB used), so stop_server's GPU check returns at once."""
    d = Path(tempfile.mkdtemp(prefix="fakebin_", dir=SCRATCH))
    for n in names:
        (d / n).write_text("#!/bin/sh\nprintf '0\\n0\\n0\\n0\\n'\n")
        (d / n).chmod(0o755)
    return d


def ns(**kw):
    return SimpleNamespace(**kw)
