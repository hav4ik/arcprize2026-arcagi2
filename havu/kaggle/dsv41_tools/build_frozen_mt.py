#!/usr/bin/env python3
"""E046: build a frozen-input submission notebook: submit Version 5's exact notebook and inputs (model v1,
chankhavu/dsv41-tools-kit, the wheels kernel, competition data) + a settings cell + cell 0b, which installs a pinned
solver and harness over the kit (sha256, feature and setting checks; a failure raises in the commit run, falls back to v3 in the rerun). No new
dataset version is mounted (Kaggle mount errors cluster on fresh cache entries, discussion 752147).

Usage: build_frozen_mt.py --slug dsv41-tools-submit-mtf --workers 36 --budget 0.5 [--gate 0.6]
       [--solver PATH] [--harness PATH] [--out DIR]
The commit run ("Save & Run All") is a smoke test: server start + 3 tasks (2 multi-test); the competition rerun runs
every task with the settings below.
"""
import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = HERE / "out/nb_submit_v4"  # submit Version 5's notebook (v3's cells + the v4 settings cell)


def settings_cell(a):
    env = [("ARC_WORKERS", str(a.workers)), ("ARC_MULTI_TEST", "1"), ("ARC_MULTI_BUDGET", str(a.budget)),
           ("ARC_CONTEXT_LIMIT", "163840"), ("ARC_MAX_NUM_SEQS", "64"), ("ARC_GPU_UTIL", "0.945")]
    if a.gate:
        env += [("ARC_KV_ADMIT_MAX", str(a.gate)), ("ARC_KV_ADMIT_MIN_WORKERS", "16")]
    pairs = ",\n               ".join(f'("{k}", "{v}")' for k, v in env)
    return f'''# Cell 00 (E046 frozen-input candidate): submit Version 5's settings (Humming dense, CUDA graphs <= 64, max-num-seqs
# 64, util 0.945; Kaggle KV 1.55M) + one rollout per task for tasks with several test inputs (ARC_MULTI_TEST) with a
# (1 + {a.budget} (n - 1)) x output budget / turn cap (ARC_MULTI_BUDGET); {a.workers} workers{f", KV gate {a.gate}" if a.gate else ""}.
# E128 on the 45 multi-pair eval tasks (E046 run 3): pass@2 83.1% vs 77.5% per pair at -29% tokens per task.
# ARC_CONTEXT_LIMIT = the server's max-model-len: turns never request past it (a 3-input task overflowed in run 3).
import json, os
for _k, _v in ({pairs},
               ("ARC_SERVER_EXTRA_ARGS", json.dumps(["--linear-backend", "humming", "--compilation-config",
                                                     json.dumps({{"cudagraph_mode": "FULL",
                                                                 "max_cudagraph_capture_size": 64}})]))):
    os.environ.setdefault(_k, _v)
# The commit run ("Save & Run All") is a smoke test: server start + 3 tasks (2 multi-test). The competition rerun
# ignores ARC_COMMIT_MODE (cell 0 always runs "submit") and never sees the task list.
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    os.environ.setdefault("ARC_COMMIT_MODE", "smoke")
    os.environ.setdefault("ARC_TASK_IDS", "25ff71a9,27a28665,00576224")
print("E046 settings:", {{k: os.environ.get(k) for k in ("ARC_WORKERS", "ARC_MULTI_TEST", "ARC_MULTI_BUDGET",
                                                     "ARC_CONTEXT_LIMIT", "ARC_KV_ADMIT_MAX", "ARC_MAX_NUM_SEQS", "ARC_GPU_UTIL")}}, flush=True)
'''


def code_cell(files, expect):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for arc, src in files.items():
            tf.add(src, arcname=arc)
    b64 = base64.b64encode(buf.getvalue()).decode()
    sha = {arc: hashlib.sha256(Path(src).read_bytes()).hexdigest() for arc, src in files.items()}
    return f'''# Cell 0b (E046, frozen inputs): the pinned solver + harness ride in the notebook, so the inputs stay exactly
# Version 5's (no new dataset version to mount). Overwrites the kit's two files after cell 0 unpacked and checked it,
# then checks hashes, required features and settings.
# Commit run ("Save & Run All"): any failure RAISES, so the commit errors and the log shows "E046 CHECK FAILED".
# Competition rerun: the same failure restores the kit's original v3 files and runs per pair at 36 workers (a scored
# submission beats an error); the log line is printed either way.
import base64, hashlib, io, shutil, tarfile
_MT = "{b64}"
_MT_SHA = {json.dumps(sha)}
_MT_EXPECT = {json.dumps(expect)}
if MODE != "fallback":
    _bak, _why = {{}}, ""
    try:
        assert KIT, "kit not unpacked"
        for _f in _MT_SHA:
            _bak[_f] = f"{{KIT}}/{{_f}}.v3orig"
            shutil.copy2(f"{{KIT}}/{{_f}}", _bak[_f])
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_MT))) as _tf:
            _tf.extractall(KIT)
        for _f, _h in _MT_SHA.items():
            assert hashlib.sha256(open(f"{{KIT}}/{{_f}}", "rb").read()).hexdigest() == _h, f"overlay mismatch: {{_f}}"
        _sv = open(f"{{KIT}}/solver_dsv41.py").read()
        _hv = open(f"{{KIT}}/repo/evals/run_deepseek_tools.py").read()
        for _need in ("MULTI_TEST", "MULTI_BUDGET", "CONTEXT_LIMIT", "KV_ADMIT_MAX", "split_multi"):
            assert _need in _sv, f"solver lacks {{_need}}"
        for _need in ("--multi-test", "--multi-budget", "--context-limit", "class ContextFull", "import builtins as _b"):
            assert _need in _hv, f"harness lacks {{_need}}"
        for _k, _v in _MT_EXPECT.items():
            assert os.environ.get(_k) == _v, f"{{_k}}={{os.environ.get(_k)!r}}, expected {{_v!r}}"
    except Exception as exc:
        _why = f"{{type(exc).__name__}}: {{exc}}"
    if not _why:
        print("E046 CHECK OK: multi-test code installed and checked:", {{f: h[:8] for f, h in _MT_SHA.items()}},
              _MT_EXPECT, flush=True)
    else:
        print("=" * 100, f"E046 CHECK FAILED: {{_why}}", "=" * 100, sep=chr(10), flush=True)
        if not RERUN:
            raise RuntimeError(f"E046 CHECK FAILED (commit run): {{_why}}")
        for _f, _b in _bak.items():  # rerun: back to the kit's v3 files, per pair, 36 workers
            shutil.copy2(_b, f"{{KIT}}/{{_f}}")
        for _k in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT"):
            os.environ.pop(_k, None)
        os.environ["ARC_WORKERS"] = "36"
        print("E046 FALLBACK: the kit's v3 code, per pair, 36 workers", flush=True)
''', sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--solver", type=Path, default=HERE / "solver_dsv41.py")
    ap.add_argument("--harness", type=Path, default=ROOT / "evals/run_deepseek_tools.py")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    nb = json.load(open(V5 / "dsv41-tools-submit.ipynb"))
    meta = json.load(open(V5 / "kernel-metadata.json"))
    assert "".join(nb["cells"][1]["source"]).startswith("# Cell 0: clock, paths"), "unexpected cell order"
    expect = {"ARC_WORKERS": str(a.workers), "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": str(a.budget),
              "ARC_CONTEXT_LIMIT": "163840"}
    if a.gate:
        expect["ARC_KV_ADMIT_MAX"] = str(a.gate)
    cell0b, sha = code_cell({"solver_dsv41.py": a.solver, "repo/evals/run_deepseek_tools.py": a.harness}, expect)
    nb["cells"][0]["source"] = settings_cell(a).splitlines(keepends=True)
    nb["cells"].insert(2, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": cell0b.splitlines(keepends=True)})
    out = a.out or HERE / f"out/nb_{a.slug}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / f"{a.slug}.ipynb", "w"), indent=1, ensure_ascii=False)
    meta.update(id=f"chankhavu/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb")
    meta.pop("id_no", None)
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
    print(out, {k: meta[k] for k in ("dataset_sources", "model_sources", "kernel_sources", "is_private")},
          {f: h[:8] for f, h in sha.items()})


if __name__ == "__main__":
    main()
