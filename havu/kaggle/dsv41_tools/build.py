#!/usr/bin/env python3
"""Build the DeepSeek-V4.1 tool-solver kit dataset and its Kaggle submission notebook (E042 RW / E043 tool models).

    .venv/bin/python kaggle/dsv41_tools/build.py            # -> kaggle/dsv41_tools/out/{kit,nb,local_run.py}

- out/kit/ (Kaggle dataset chankhavu/dsv41-tools-kit): an exact snapshot of the harness files the solver imports
  (evals/run_deepseek_tools.py and its imports, the ARC Prize prompt), solver_dsv41.py, stage_model.py, the vLLM
  overlay (every file of the pod's patched vLLM that differs from the fork wheel, with sha256), MANIFEST.json (sha256
  of every kit file).
- out/nb/ (notebook chankhavu/dsv41-tools-submit): plain-Python cells, no shell magics:
  0. T0 and paths; the fallback submission.json is written before anything else.
  1. Model staging (stage_model.py, detached): copy the checkpoint to local disk while vLLM installs.
  2. Offline install of the wheel set (E042's chankhavu/dsv41-sm89-fork-py312-cu130 output), the overlay (sha256
     checked), CUDA 13 shim for FlashInfer's JIT, host libcuda.
  3. solver_dsv41.py with the venv's Python (the tool code runs on the image's Python, which has scipy).
  4. Final check of submission.json (rewritten from the fallback if invalid) and a log archive.
- out/local_run.py: the same cells as one script, for the Runpod pods (paths from ARC_* environment variables).
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
USER = "chankhavu"
KIT_SLUG = "dsv41-tools-kit"
NB_SLUG = "dsv41-tools-submit"
WHEELS_KERNEL = "chankhavu/dsv41-sm89-fork-py312-cu130"
COMPETITION = "arc-prize-2026-arc-agi-2"
DOCKER = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"
HARNESS_FILES = [
    "evals/run_deepseek_tools.py", "evals/run_deepseek_arc2.py", "evals/run_deepseek_arcprize.py",
    "evals/sandbox_client.py", "evals/task_summary.py", "evals/train_check.py", "evals/arc_helpers.py",
    "notes/prompts/arcprize_official_system_prompt.txt",
] + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "evals/arc_harness").glob("*")) if p.suffix in (".py", ".txt")]

CELL_SETUP = r'''
# Cell 0: clock, paths, fallback submission (before anything that can fail).
import glob, json, os, subprocess, sys, time
T0 = float(os.environ.get("ARC_T0") or time.time()); os.environ["ARC_T0"] = str(T0)
INPUT = os.environ.get("ARC_INPUT_ROOT", "/kaggle/input")
WORK = os.environ.get("ARC_OUT", "/kaggle/working"); os.environ["ARC_OUT"] = WORK
SCRATCH = os.environ.get("ARC_SCRATCH", "/tmp/dsv41"); os.makedirs(SCRATCH, exist_ok=True); os.makedirs(WORK, exist_ok=True)
RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
# Commit runs ("Save & Run All") only check the pipeline: "fallback" writes the fallback and stops; "smoke" runs
# ARC_TASK_LIMIT tasks with short deadlines. The competition rerun always runs "submit".
COMMIT_MODE = os.environ.get("ARC_COMMIT_MODE", "fallback")
MODE = "submit" if RERUN else os.environ.get("ARC_MODE", COMMIT_MODE)

def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

CHALLENGES = os.environ.get("ARC_CHALLENGES") or find_one(f"{INPUT}/**/arc-agi_test_challenges.json")
os.environ["ARC_CHALLENGES"] = CHALLENGES
_tasks = json.load(open(CHALLENGES))
_fb = {t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in v["test"]] for t, v in _tasks.items()}
with open(f"{WORK}/submission.json.tmp", "w") as f:
    json.dump(_fb, f)
os.replace(f"{WORK}/submission.json.tmp", f"{WORK}/submission.json")
print(f"mode {MODE}; fallback submission for {len(_tasks)} tasks written at +{time.time() - T0:.0f}s", flush=True)
KIT = os.environ.get("ARC_KIT") or os.path.dirname(find_one(f"{INPUT}/**/solver_dsv41.py"))
MODEL_SRC = os.environ.get("ARC_MODEL_SRC") or (os.path.dirname(find_one(f"{INPUT}/**/e043_manifest.json") or "") or None)
print("kit", KIT, "| model", MODEL_SRC, flush=True)
'''

CELL_STAGE = r'''
# Cell 1: stage the checkpoint to local disk in the background (stage_model.py), while vLLM installs.
STAGE_MODE = os.environ.get("ARC_STAGE_MODE", "copy")
STAGE_DONE = f"{SCRATCH}/stage_done"
if MODE != "fallback" and MODEL_SRC:
    if os.path.exists(STAGE_DONE):
        os.remove(STAGE_DONE)
    _stage = subprocess.Popen([sys.executable, "-I", f"{KIT}/stage_model.py", MODEL_SRC, f"{SCRATCH}/model", STAGE_DONE,
                               STAGE_MODE, os.environ.get("ARC_STAGE_THREADS", "16")],
                              stdout=open(f"{SCRATCH}/stage.log", "w"), stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"staging ({STAGE_MODE}) started, pid {_stage.pid}", flush=True)
'''

CELL_INSTALL = r'''
# Cell 2: offline install of the fork wheel set, the vLLM overlay, the CUDA 13 shim (E042's Kaggle recipe).
import hashlib, shutil, tarfile
V = os.environ.get("ARC_VENV", f"{SCRATCH}/venv"); PY = f"{V}/bin/python"; SP = f"{V}/lib/python3.12/site-packages"
BASE_ENV = dict(os.environ)
if MODE != "fallback" and not os.environ.get("ARC_SKIP_INSTALL"):
    def sh(cmd, timeout=None, env=None):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    real = "/usr/local/nvidia/lib64"  # host driver libcuda first (the image also has an older compat libcuda)
    if os.path.exists(real + "/libcuda.so"):
        os.environ["LD_LIBRARY_PATH"] = real + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_PRELOAD"] = real + "/libcuda.so"
    WHEELS = os.path.dirname(find_one(f"{INPUT}/**/vllm-0.28.1rc1*.whl"))
    t = time.time()
    if shutil.which("uv"):
        rc, out = sh(f"uv venv --python {sys.executable} {V} && uv pip install --python {PY} --no-index --no-deps "
                     f"--find-links {WHEELS} -r {WHEELS}/requirements.lock", timeout=3600)
    else:
        rc, out = sh(f"{sys.executable} -m venv {V} && {PY} -m pip install --no-index --no-deps --find-links {WHEELS} "
                     f"-r {WHEELS}/requirements.lock", timeout=3600)
    print(f"offline install rc {rc} in {time.time() - t:.0f}s", out[-800:], flush=True)
    assert rc == 0, "vLLM install failed"
    with tarfile.open(f"{KIT}/vllm_overlay_rw.tgz") as tf:
        tf.extractall(SP)
    for line in open(f"{KIT}/vllm_overlay_rw.sha256"):
        h, rel = line.split()
        assert hashlib.sha256(open(f"{SP}/{rel}", "rb").read()).hexdigest() == h, f"overlay mismatch: {rel}"
    print("vLLM overlay installed and checked", flush=True)
    cu = f"{SP}/nvidia/cu13"; shim = f"{SCRATCH}/cuda13"
    os.makedirs(f"{shim}/lib64/stubs", exist_ok=True)
    for d in ("bin", "include", "nvvm", "lib", "share"):
        if os.path.exists(f"{cu}/{d}") and not os.path.lexists(f"{shim}/{d}"):
            os.symlink(f"{cu}/{d}", f"{shim}/{d}")
    for f in os.listdir(f"{cu}/lib"):
        if not os.path.lexists(f"{shim}/lib64/{f}"):
            os.symlink(f"{cu}/lib/{f}", f"{shim}/lib64/{f}")
    for name in ("cudart", "nvrtc"):
        r = sorted(glob.glob(f"{cu}/lib/lib{name}.so.*"))
        if r and not os.path.lexists(f"{shim}/lib64/lib{name}.so"):
            os.symlink(r[0], f"{shim}/lib64/lib{name}.so")
    drv = next(iter(sorted(glob.glob("/usr/local/nvidia/lib64/libcuda.so*")) + sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*"))), None)
    for dst in (f"{shim}/lib64/stubs/libcuda.so", f"{shim}/lib64/libcuda.so"):
        if drv and not os.path.lexists(dst):
            os.symlink(drv, dst)
    BASE_ENV = dict(os.environ)
    BASE_ENV.update(CUDA_HOME=shim, CUDA_PATH=shim, PATH=f"{V}/bin:{shim}/bin:" + os.environ["PATH"],
                    LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", "") + f":{shim}/lib64")
BASE_ENV.pop("DSV41_SYNTH_INIT", None)  # the overlay's synthetic init must stay off for real weights
# FlashInfer's sampling module cannot JIT on the Kaggle image (E042 K-runs); vLLM's PyTorch sampler instead.
BASE_ENV["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
'''

CELL_RUN = r'''
# Cell 3: the solver (vLLM server, tool harness, deadlines, watchdog, incremental submission).
if MODE != "fallback" and MODEL_SRC:
    env = dict(BASE_ENV)
    env.update(ARC_KIT=KIT, ARC_MODEL_DIR=MODEL_SRC, ARC_STAGE_DONE=STAGE_DONE, ARC_VLLM=f"{V}/bin/vllm",
               ARC_TOOL_PYTHON=os.environ.get("ARC_TOOL_PYTHON", sys.executable))
    if MODE == "smoke":
        env.setdefault("ARC_TASK_LIMIT", "8")
        for k, v in (("ARC_STOP_NEW_S", "4800"), ("ARC_FORCE_AT_S", "5400"), ("ARC_HARD_STOP_S", "5700")):
            env.setdefault(k, v)
    p = subprocess.Popen([PY, "-u", f"{KIT}/solver_dsv41.py"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print("solver exit code", p.returncode, flush=True)
'''

CELL_FINISH = r'''
# Cell 4: final check of submission.json; log archive.
import tarfile
def _ok(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))
try:
    sub = json.load(open(f"{WORK}/submission.json"))
    assert set(sub) == set(_tasks)
    for t, v in _tasks.items():
        assert len(sub[t]) == len(v["test"]) and all(_ok(r["attempt_1"]) and _ok(r["attempt_2"]) for r in sub[t])
    print("submission.json valid", flush=True)
except Exception as exc:
    print("submission.json invalid, writing the fallback:", exc, flush=True)
    json.dump(_fb, open(f"{WORK}/submission.json", "w"))
logs = f"{WORK}/dsv41_logs"
if os.path.isdir(f"{logs}/traces"):
    with tarfile.open(f"{WORK}/dsv41_traces.tar.gz", "w:gz") as tf:
        tf.add(f"{logs}/traces", arcname="traces")
    shutil.rmtree(f"{logs}/traces", ignore_errors=True)
for f in ("stage.log",):
    if os.path.exists(f"{SCRATCH}/{f}"):
        shutil.copy(f"{SCRATCH}/{f}", f"{logs}/{f}" if os.path.isdir(logs) else f"{WORK}/{f}")
print(f"done at +{(time.time() - T0) / 3600:.2f}h", flush=True)
'''

CELLS = [CELL_SETUP, CELL_STAGE, CELL_INSTALL, CELL_RUN, CELL_FINISH]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_kit():
    kit = OUT / "kit"
    shutil.rmtree(kit, ignore_errors=True)
    for rel in HARNESS_FILES:
        dst = kit / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for name in ("solver_dsv41.py", "stage_model.py"):
        shutil.copy2(HERE / name, kit / name)
    for name in ("vllm_overlay_rw.tgz", "vllm_overlay_rw.sha256"):
        shutil.copy2(HERE / "kit" / name, kit / name)
    manifest = {str(p.relative_to(kit)): sha(p) for p in sorted(kit.rglob("*")) if p.is_file()}
    (kit / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (kit / "dataset-metadata.json").write_text(json.dumps({
        "title": KIT_SLUG, "id": f"{USER}/{KIT_SLUG}", "licenses": [{"name": "other"}]}, indent=1))
    return manifest


def build_notebook():
    nb = OUT / "nb"
    shutil.rmtree(nb, ignore_errors=True)
    nb.mkdir(parents=True)
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": c.strip("\n").splitlines(keepends=True)} for c in CELLS]
    (nb / f"{NB_SLUG}.ipynb").write_text(json.dumps({
        "cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1))
    (nb / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{NB_SLUG}", "title": NB_SLUG, "code_file": f"{NB_SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [f"{USER}/{KIT_SLUG}"], "kernel_sources": [WHEELS_KERNEL],
        "competition_sources": [COMPETITION], "model_sources": [], "docker_image": DOCKER,
        "docker_image_pinning_type": "original", "machine_shape": "NvidiaL4"}, indent=1))
    (OUT / "local_run.py").write_text(
        '"""The notebook cells as one script, for the Runpod pods (ARC_* environment variables set the paths)."""\n'
        + "\n".join(c.strip("\n") + "\n" for c in CELLS))


def main():
    manifest = build_kit()
    build_notebook()
    print(f"kit: {len(manifest)} files -> {OUT / 'kit'}; notebook -> {OUT / 'nb'}; local runner -> {OUT / 'local_run.py'}")
    print("NOTE: the model dataset is not attached yet (not uploaded); add it to kernel-metadata model/dataset sources.")


if __name__ == "__main__":
    main()
