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



def opt_env(a):
    """Settings cell / expected env for the opt-in levers of the next candidate (empty for V6's settings)."""
    env, g = [], lambda k: getattr(a, k, 0)
    if g("effort"):
        env.append(("ARC_REASONING_EFFORT", str(g("effort"))))
    if g("auto_repair"):
        env.append(("ARC_AUTO_REPAIR", "1"))
    if g("best_guess_at"):
        env.append(("ARC_BEST_GUESS_AT", str(g("best_guess_at"))))
    if g("best_guess_turns"):
        env.append(("ARC_BEST_GUESS_TURNS", str(g("best_guess_turns"))))
    if g("post_pass_note"):
        env.append(("ARC_POST_PASS_NOTE", "1"))
    return env


def opt_need(a):
    """Cell 0b's feature check for the opt-in levers: {pinned file: strings it must contain}."""
    need, g = {}, lambda k: getattr(a, k, 0)
    harness, solver = "repo/evals/run_deepseek_tools.py", "solver_dsv41.py"
    if g("auto_repair"):
        need.setdefault(harness, []).append("--auto-repair")
    if g("best_guess_at") or g("best_guess_turns"):
        need.setdefault(harness, []).append("--best-guess-at")
    if g("post_pass_note"):
        need.setdefault(harness, []).append("--post-pass-note")
        need.setdefault(solver, []).append("ARC_POST_PASS_NOTE")
    if g("effort"):
        need.setdefault(solver, []).append("ARC_REASONING_EFFORT")
    return need


def retention_args(a):
    r = getattr(a, "retention", 0)
    return ["--prefix-cache-retention-interval", str(r)] if r else []


def settings_cell(a):
    env = [("ARC_WORKERS", str(a.workers)), ("ARC_MULTI_TEST", "1"), ("ARC_MULTI_BUDGET", str(a.budget)),
           ("ARC_CONTEXT_LIMIT", "163840"), ("ARC_MAX_NUM_SEQS", "64"), ("ARC_GPU_UTIL", "0.945")]
    if a.gate:
        env += [("ARC_KV_ADMIT_MAX", str(a.gate)), ("ARC_KV_ADMIT_MIN_WORKERS", "16")]
    env += opt_env(a)
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
                                                                 "max_cudagraph_capture_size": 64}})]
                                                    + {json.dumps(retention_args(a))}))):
    os.environ.setdefault(_k, _v)
# The commit run ("Save & Run All") is a smoke test: server start + 3 tasks (2 multi-test). The competition rerun
# ignores ARC_COMMIT_MODE (cell 0 always runs "submit") and never sees the task list.
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    os.environ.setdefault("ARC_COMMIT_MODE", "smoke")
    os.environ.setdefault("ARC_TASK_IDS", "25ff71a9,27a28665,00576224")
print("E046 settings:", {{k: os.environ.get(k) for k in ("ARC_WORKERS", "ARC_MULTI_TEST", "ARC_MULTI_BUDGET",
                                                     "ARC_CONTEXT_LIMIT", "ARC_KV_ADMIT_MAX", "ARC_MAX_NUM_SEQS", "ARC_GPU_UTIL",
                                                     "ARC_REASONING_EFFORT", "ARC_AUTO_REPAIR", "ARC_BEST_GUESS_AT",
                                                     "ARC_BEST_GUESS_TURNS", "ARC_POST_PASS_NOTE",
                                                     "ARC_SERVER_EXTRA_ARGS")}}, flush=True)
'''


def code_cell(files, expect, new=(), need=None):
    need = need or {}
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
# submission beats an error); the log line is printed either way. The rerun path never raises (S10): a file whose
# backup is missing or cannot be copied back keeps what is in the kit (the pinned file, or the kit's own if the
# overlay never ran), and the notebook still runs.
import base64, hashlib, io, shutil, tarfile
_MT = "{b64}"
_MT_SHA = {json.dumps(sha)}
_MT_EXPECT = {json.dumps(expect)}
_MT_NEW = {json.dumps(sorted(new))}
_MT_NEED = {json.dumps(need)}
if MODE != "fallback":
    _bak, _why = {{}}, ""
    try:
        assert KIT, "kit not unpacked"
        for _f in _MT_SHA:
            if _f in _MT_NEW:
                continue  # new to the kit (no v3 original); the v3 code never imports it
            shutil.copy2(f"{{KIT}}/{{_f}}", f"{{KIT}}/{{_f}}.v3orig")
            _bak[_f] = f"{{KIT}}/{{_f}}.v3orig"  # S10: only once the backup exists
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_MT))) as _tf:
            _tf.extractall(KIT)
        for _f, _h in _MT_SHA.items():
            assert hashlib.sha256(open(f"{{KIT}}/{{_f}}", "rb").read()).hexdigest() == _h, f"overlay mismatch: {{_f}}"
        _sv = open(f"{{KIT}}/solver_dsv41.py").read()
        _hv = open(f"{{KIT}}/repo/evals/run_deepseek_tools.py").read()
        for _need in ("MULTI_TEST", "MULTI_BUDGET", "CONTEXT_LIMIT", "KV_ADMIT_MAX", "split_multi", "restart_allowed",
                      "shutdown_socket"):
            assert _need in _sv, f"solver lacks {{_need}}"
        for _need in ("--multi-test", "--multi-budget", "--context-limit", "class ContextFull", "import builtins as _b",
                      "def answer_items", "def context_budget"):
            assert _need in _hv, f"harness lacks {{_need}}"
        for _f, _needs in _MT_NEED.items():  # features the chosen opt-in settings rely on
            _txt = open(f"{{KIT}}/{{_f}}").read()
            for _need in _needs:
                assert _need in _txt, f"{{_f}} lacks {{_need}}"
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
        _kept = []
        try:
            for _f in _MT_SHA:  # rerun: back to the kit's v3 files, per pair, 36 workers
                if _f in _MT_NEW:
                    continue
                try:
                    shutil.copy2(_bak[_f], f"{{KIT}}/{{_f}}.restore")
                    os.replace(f"{{KIT}}/{{_f}}.restore", f"{{KIT}}/{{_f}}")
                except Exception as exc:
                    _kept.append(f"{{_f}} ({{type(exc).__name__}})")
            # (a kept pinned solver would pass the opt-in harness flags to the v3 harness, whose parser exits)
            for _k in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT", "ARC_AUTO_REPAIR",
                       "ARC_BEST_GUESS_AT", "ARC_BEST_GUESS_TURNS", "ARC_POST_PASS_NOTE", "ARC_REASONING_EFFORT"):
                os.environ.pop(_k, None)
            os.environ["ARC_WORKERS"] = "36"
        except Exception as exc:
            _kept.append(f"{{type(exc).__name__}}: {{exc}}")
        print("E046 FALLBACK: the kit's v3 code, per pair, 36 workers"
              + (f"; not restored, kept as is: {{', '.join(_kept)}}" if _kept else ""), flush=True)
''', sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--solver", type=Path, default=HERE / "solver_dsv41.py")
    ap.add_argument("--harness", type=Path, default=ROOT / "evals/run_deepseek_tools.py")
    ap.add_argument("--retention", type=int, default=0, help="--prefix-cache-retention-interval (0: the base 128)")
    ap.add_argument("--effort", type=int, default=0, help="ARC_REASONING_EFFORT (0: the solver's 50)")
    ap.add_argument("--auto-repair", action="store_true", help="ARC_AUTO_REPAIR=1 + embed evals/auto_repair*.")
    ap.add_argument("--best-guess-at", type=float, default=0.0)
    ap.add_argument("--best-guess-turns", type=int, default=0)
    ap.add_argument("--post-pass-note", action="store_true", help="ARC_POST_PASS_NOTE=1 (I8)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    nb = json.load(open(V5 / "dsv41-tools-submit.ipynb"))
    meta = json.load(open(V5 / "kernel-metadata.json"))
    assert "".join(nb["cells"][1]["source"]).startswith("# Cell 0: clock, paths"), "unexpected cell order"
    expect = {"ARC_WORKERS": str(a.workers), "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": str(a.budget),
              "ARC_CONTEXT_LIMIT": "163840"}
    if a.gate:
        expect["ARC_KV_ADMIT_MAX"] = str(a.gate)
    expect.update(opt_env(a))
    files = {"solver_dsv41.py": a.solver, "repo/evals/run_deepseek_tools.py": a.harness}
    new, need = [], opt_need(a)
    if a.auto_repair:
        for name in ("auto_repair.py", "auto_repair_names.json"):
            files[f"repo/evals/{name}"] = ROOT / "evals" / name
            new.append(f"repo/evals/{name}")
    cell0b, sha = code_cell(files, expect, new, need)
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
