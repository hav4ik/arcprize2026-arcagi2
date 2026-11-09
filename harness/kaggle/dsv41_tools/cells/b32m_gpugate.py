# Cell 0a (B32-M GPU gate): fail fast when the session has no usable GPU or the wrong Python, before Manh's install.
# Smoke V1 (2026-10-28): every nvidia-smi sample said "couldn't find libnvidia-ml.so" and torch saw 0 GPUs. Records
# /dev/nvidia*, nvidia-smi -L, ldconfig's NVIDIA libraries, NVIDIA_* / LD_LIBRARY_PATH, every libnvidia-ml.so* /
# libcuda.so* under /usr and /opt, and the Python version (b32m_gpu_gate.json). If the device nodes exist but NVML is
# not found while a libnvidia-ml.so.1 exists, its folder goes on LD_LIBRARY_PATH (os.environ: subprocesses inherit
# it). It retries for up to ARC_GPU_GATE_S (90 s; NVML has appeared ~30 s after start before). No 4 GPUs, or a Python
# other than 3.13 (his wheels are cp313): commit runs RAISE at once (no quota burnt); the competition rerun keeps
# B32's fallback submission and turns every later cell of ours off (MODE = "fallback").
import json, os, subprocess, sys, time


def _gate_sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _gate_gpus():
    rc, out = _gate_sh("nvidia-smi -L", 30)
    return sum(1 for l in out.splitlines() if l.startswith("GPU ")), rc, out[-600:]


GPU_GATE = {"t0_s": round(time.time() - T0), "python": sys.version.split()[0],
            "dev_nvidia": _gate_sh("ls -l /dev/nvidia* 2>&1")[1][-1500:],
            "ldconfig_nvidia": _gate_sh("ldconfig -p | grep -i nvidia")[1][-2500:],
            "env": {k: v for k, v in os.environ.items() if k.startswith("NVIDIA_") or k == "LD_LIBRARY_PATH"},
            "libs": _gate_sh("find /usr /opt \\( -name 'libnvidia-ml.so*' -o -name 'libcuda.so*' \\) 2>/dev/null "
                             "| sort", 120)[1].splitlines()[:60], "tries": []}
_need = int(os.environ.get("ARC_GPU_GATE_N", "4"))
_t = time.time()
while True:
    _n, _rc, _out = _gate_gpus()
    GPU_GATE["tries"].append({"t_s": round(time.time() - T0), "gpus": _n, "rc": _rc, "out": _out[-300:]})
    if _n >= _need:
        break
    if ("libnvidia-ml" in _out or "NVML" in _out) and GPU_GATE["dev_nvidia"].count("/dev/nvidia") \
            and not GPU_GATE.get("ld_added"):
        _ml = [p for p in GPU_GATE["libs"] if os.path.basename(p) == "libnvidia-ml.so.1"]
        if _ml:  # NVML exists but is not on the loader path: put its folder first, for this kernel and subprocesses
            _d = os.path.dirname(_ml[0])
            os.environ["LD_LIBRARY_PATH"] = _d + (":" + os.environ["LD_LIBRARY_PATH"]
                                                  if os.environ.get("LD_LIBRARY_PATH") else "")
            GPU_GATE["ld_added"] = _d
            continue
    if time.time() - _t > float(os.environ.get("ARC_GPU_GATE_S", "90")):
        break
    time.sleep(10)
GPU_GATE["gpus"] = _n
_why = []
if _n < _need:
    _why.append(f"{_n} GPUs visible to nvidia-smi after {time.time() - _t:.0f}s (need {_need}): {_out[-200:]!r}")
_py = os.environ.get("ARC_GPU_GATE_PY", "3.13")  # his wheels are cp313
if ".".join(map(str, sys.version_info[:2])) != _py:
    _why.append(f"Python {GPU_GATE['python']}, but Manh's wheels need {_py} (cp313)")
GPU_GATE["fail"] = _why
with open(f"{WORK}/b32m_gpu_gate.json", "w") as _f:
    json.dump(GPU_GATE, _f, indent=1)
if _why:
    print("=" * 100, f"B32-M GPU GATE FAILED: {_why}", json.dumps(GPU_GATE)[:3000], "=" * 100, sep="\n", flush=True)
    if not RERUN:
        raise RuntimeError(f"B32-M GPU GATE FAILED (commit run): {_why}")
    MODE = "fallback"  # rerun: B32's fallback submission (written by cell 0) stands; our later cells do nothing
else:
    print(f"B32-M GPU GATE OK: {_n} GPUs, Python {GPU_GATE['python']}, LD_LIBRARY_PATH added:",
          GPU_GATE.get("ld_added"), flush=True)
