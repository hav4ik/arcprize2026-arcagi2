"""The notebook cells as one script, for the Runpod pods (ARC_* environment variables set the paths)."""
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

# Every cache under SCRATCH: /root/.cache was read-only in smoke v2 (uv could not unpack wheels there).
_C = f"{SCRATCH}/cache"
for _k, _v in (("UV_CACHE_DIR", f"{_C}/uv"), ("XDG_CACHE_HOME", _C), ("VLLM_CACHE_ROOT", f"{_C}/vllm"),
               ("TRITON_CACHE_DIR", f"{_C}/triton"), ("TORCHINDUCTOR_CACHE_DIR", f"{_C}/inductor"),
               ("FLASHINFER_WORKSPACE_BASE", _C), ("TORCH_EXTENSIONS_DIR", f"{_C}/torch_ext"),
               ("HF_HOME", f"{_C}/hf"), ("CUDA_CACHE_PATH", f"{_C}/nv")):
    os.environ.setdefault(_k, _v)
    os.makedirs(os.environ[_k], exist_ok=True)
CHALLENGES = os.environ.get("ARC_CHALLENGES") or find_one(f"{INPUT}/**/arc-agi_test_challenges.json")
os.environ["ARC_CHALLENGES"] = CHALLENGES
_tasks = json.load(open(CHALLENGES))
_fb = {t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in v["test"]] for t, v in _tasks.items()}
with open(f"{WORK}/submission.json.tmp", "w") as f:
    json.dump(_fb, f)
os.replace(f"{WORK}/submission.json.tmp", f"{WORK}/submission.json")
print(f"mode {MODE}; fallback submission for {len(_tasks)} tasks written at +{time.time() - T0:.0f}s", flush=True)
# The kit dataset is one tarball (Kaggle datasets do not keep nested folders reliably): unpack and check sha256.
KIT = os.environ.get("ARC_KIT")
if not KIT:
    try:
        import hashlib, tarfile
        _tgz = find_one(f"{INPUT}/**/dsv41_kit.tgz")
        KIT = f"{SCRATCH}/kit"
        with tarfile.open(_tgz) as tf:
            tf.extractall(KIT)
        _man = json.load(open(f"{KIT}/MANIFEST.json"))
        _bad = [f for f, h in _man.items() if hashlib.sha256(open(f"{KIT}/{f}", "rb").read()).hexdigest() != h]
        assert not _bad, f"kit files differ from MANIFEST: {_bad[:5]}"
        print(f"kit unpacked from {_tgz}: {len(_man)} files, sha256 ok", flush=True)
    except Exception as exc:  # without the kit nothing else can run: keep the fallback submission
        print("kit unavailable, fallback only:", type(exc).__name__, exc, flush=True)
        MODE = "fallback"
MODEL_SRC = os.environ.get("ARC_MODEL_SRC") or (os.path.dirname(find_one(f"{INPUT}/**/e043_manifest.json") or "") or None)
print("kit", KIT, "| model", MODEL_SRC, flush=True)

# Cell 1: stage the checkpoint to local disk in the background (stage_model.py), while vLLM installs.
# "warm" (read the mount once, no writes): smoke v2 (2026-10-24) with "copy" hit "Read-only file system" on the
# root overlay during the install while 131 GB were being written to /tmp.
STAGE_MODE = os.environ.get("ARC_STAGE_MODE", "warm")
STAGE_DONE = f"{SCRATCH}/stage_done"
if MODE != "fallback" and MODEL_SRC:
  try:
    if os.path.exists(STAGE_DONE):
        os.remove(STAGE_DONE)
    _stage = subprocess.Popen([sys.executable, "-I", f"{KIT}/stage_model.py", MODEL_SRC, f"{SCRATCH}/model", STAGE_DONE,
                               STAGE_MODE, os.environ.get("ARC_STAGE_THREADS", "16")],
                              stdout=open(f"{SCRATCH}/stage.log", "w"), stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"staging ({STAGE_MODE}) started, pid {_stage.pid}", flush=True)
  except Exception as exc:  # the solver then loads from MODEL_SRC (no staging marker)
    STAGE_DONE = ""
    print("staging failed to start:", type(exc).__name__, exc, flush=True)

# Cell 2: offline install of the fork wheel set, the vLLM overlay, the CUDA 13 shim (E042's Kaggle recipe).
import hashlib, shutil, tarfile
V = os.environ.get("ARC_VENV", f"{SCRATCH}/venv"); PY = f"{V}/bin/python"; SP = f"{V}/lib/python3.12/site-packages"
BASE_ENV = dict(os.environ)
def sh(cmd, timeout=None, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")
if MODE != "fallback" and not os.environ.get("ARC_SKIP_INSTALL"):
  try:
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
    rc, out = sh(f'nvcc --version | tail -1; {PY} -c "import torch, vllm, flashinfer, importlib.metadata as m; '
                 f'print(torch.__version__, torch.cuda.device_count(), vllm.__version__, flashinfer.__version__, '
                 f'm.version(\'humming-kernels\'))"', timeout=900, env=BASE_ENV)
    print("toolchain / imports:", rc, out[-600:], flush=True)
    rc, out = sh("free -g | head -2; df -h /tmp /kaggle/working | tail -2; nproc; cat /sys/fs/cgroup/memory.max "
                 "/sys/fs/cgroup/memory.oom.group 2>/dev/null; grep -m1 'model name' /proc/cpuinfo", timeout=60)
    print(out, flush=True)
  except Exception as exc:  # no working vLLM: keep the fallback submission
    print("install failed, fallback only:", type(exc).__name__, exc, flush=True)
    MODE = "fallback"
# The tool code runs on the image's Python (scipy etc.); if that cannot import numpy/scipy in the executor's
# environment (-I, PATH only), use the venv's Python instead (numpy only).
TOOL_PY = os.environ.get("ARC_TOOL_PYTHON", sys.executable)
if MODE != "fallback":
    try:
        r = subprocess.run([TOOL_PY, "-I", "-c", "import numpy, scipy.ndimage; print(numpy.__version__)"],
                           env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1"}, capture_output=True,
                           text=True, timeout=300)
        print("tool python", TOOL_PY, "->", r.returncode, (r.stdout + r.stderr).strip()[-300:], flush=True)
        if r.returncode != 0 and os.path.exists(PY):
            TOOL_PY = PY
    except Exception as exc:
        print("tool python check failed:", exc, flush=True)
BASE_ENV.pop("DSV41_SYNTH_INIT", None)  # the overlay's synthetic init must stay off for real weights
# FlashInfer's sampling module cannot JIT on the Kaggle image (E042 K-runs); vLLM's PyTorch sampler instead.
BASE_ENV["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Cell 3: the solver (vLLM server, tool harness, deadlines, watchdog, incremental submission).
if MODE != "fallback" and MODEL_SRC:
  try:
    import signal, threading
    env = dict(BASE_ENV)
    env.update(ARC_KIT=KIT, ARC_MODEL_DIR=MODEL_SRC, ARC_STAGE_DONE=STAGE_DONE, ARC_VLLM=f"{V}/bin/vllm",
               ARC_TOOL_PYTHON=TOOL_PY, ARC_SCRATCH=SCRATCH)
    if MODE == "smoke":
        env.setdefault("ARC_TASK_LIMIT", "8")
        for k, v in (("ARC_STOP_NEW_S", "4800"), ("ARC_FORCE_AT_S", "5400"), ("ARC_HARD_STOP_S", "5700")):
            env.setdefault(k, v)
    p = subprocess.Popen([PY, "-u", f"{KIT}/solver_dsv41.py"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    # Independent kill switch: whatever hangs inside the solver, its whole session dies at T0 + ARC_KILL_AT_S
    # (default 11h45; the solver's own hard stop is 11h30), so cell 4 and the notebook end before Kaggle's 12 h.
    def _kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
            print("kill switch: solver session killed", flush=True)
        except Exception:
            pass
    _timer = threading.Timer(max(0.0, T0 + float(os.environ.get("ARC_KILL_AT_S", 42300)) - time.time()), _kill)
    _timer.daemon = True
    _timer.start()
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    _timer.cancel()
    print("solver exit code", p.returncode, flush=True)
  except Exception as exc:
    print("solver cell failed:", type(exc).__name__, exc, flush=True)

# Cell 4: final check of submission.json; log archive.
import tarfile
def _ok(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))
try:
    sub = json.load(open(f"{WORK}/submission.json"))
    assert set(sub) == set(_tasks), f"task ids differ ({len(sub)} vs {len(_tasks)})"
    for t, v in _tasks.items():
        assert len(sub[t]) == len(v["test"]), f"{t}: {len(sub[t])} rows for {len(v['test'])} test inputs"
        assert all(_ok(r["attempt_1"]) and _ok(r["attempt_2"]) for r in sub[t]), f"{t}: invalid grid"
    print("submission.json valid", flush=True)
except Exception as exc:
    print("submission.json invalid, writing the fallback:", exc, flush=True)
    json.dump(_fb, open(f"{WORK}/submission.json", "w"))
try:
    logs = f"{WORK}/dsv41_logs"
    if os.path.isdir(f"{logs}/traces"):
        with tarfile.open(f"{WORK}/dsv41_traces.tar.gz", "w:gz") as tf:
            tf.add(f"{logs}/traces", arcname="traces")
        shutil.rmtree(f"{logs}/traces", ignore_errors=True)
    for f in ("stage.log",):
        if os.path.exists(f"{SCRATCH}/{f}"):
            shutil.copy(f"{SCRATCH}/{f}", f"{logs}/{f}" if os.path.isdir(logs) else f"{WORK}/{f}")
except Exception as exc:
    print("log archive failed:", exc, flush=True)
print(f"done at +{(time.time() - T0) / 3600:.2f}h", flush=True)
