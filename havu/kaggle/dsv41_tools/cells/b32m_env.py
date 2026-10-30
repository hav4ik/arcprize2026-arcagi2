# Cell 2 (B32-M): Manh's environment for B32's solver. His cells above installed CUDA 13 and his wheels into the
# kernel's Python; this replaces B32's fork wheels, overlay and venv. The solver starts his vLLM itself through a
# `vllm` wrapper (the kernel's Python -m vllm.entrypoints.cli.main, as his dsv41_vllm.py launches it). Then: an
# import check (no working vLLM -> fallback submission, as in B32), the environment record (versions, his wheel and
# dataset versions, sha256 of the files his weight loading runs: b32m_env.json) and B32's tool-python check.
# Defines what cells 2b, 3 and 5 use: V, PY, BASE_ENV, TOOL_PY, sh(). Uses cell 0's find_inputs (no recursive scan of
# the input root: his wheels are looked up in the two dirs his cell 3 installs from, and at fixed depths).
import hashlib, importlib.metadata as _md, platform, shutil
V = f"{SCRATCH}/b32m"; PY = sys.executable
MANH_VERSIONS = json.loads(r"""__VERSIONS__""")  # recorded by build_b32m.py (dataset versions, wheel names)


def sh(cmd, timeout=None, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


os.makedirs(f"{V}/bin", exist_ok=True)
with open(f"{V}/bin/vllm", "w") as _f:  # cell 3 passes ARC_VLLM={V}/bin/vllm
    _f.write(f'#!/bin/sh\nexec "{sys.executable}" -m vllm.entrypoints.cli.main "$@"\n')
os.chmod(f"{V}/bin/vllm", 0o755)
BASE_ENV = dict(os.environ)
_cuda = next((d for d in ("/usr/local/cuda-13.0", "/usr/local/cuda") if os.path.exists(f"{d}/bin/nvcc")), None)
if _cuda:  # JIT (Triton, TileLang, Humming) needs his CUDA 13 toolchain first on PATH
    BASE_ENV.update(CUDA_HOME=_cuda, CUDA_PATH=_cuda, PATH=f"{_cuda}/bin:" + BASE_ENV.get("PATH", ""))
ENV_RECORD = {"python": sys.version.split()[0], "platform": platform.platform(), "cuda_home": _cuda,
              "manh_versions": MANH_VERSIONS, "install_cell_start_s": dict(globals().get("_B32M_T", {})),
              "env_cell_start_s": round(time.time() - T0), "rowchunk_patch": globals().get("ROWCHUNK")}
if MODE != "fallback":
    try:
        rc, out = sh(f'{PY} -c "import torch, vllm; print(torch.__version__, torch.version.cuda, '
                     f'torch.cuda.device_count(), vllm.__version__, vllm.__file__)"', timeout=900, env=BASE_ENV)
        ENV_RECORD["import_check"] = [rc, out.strip()[-400:]]
        print("vLLM import check:", rc, out.strip()[-400:], flush=True)
        assert rc == 0 and " 4 " in f" {out} ", "his vLLM does not import with 4 GPUs"
        _pk = {}
        for _p in ("vllm", "torch", "humming-kernels", "flashinfer-python", "flashinfer-cubin", "triton", "tilelang",
                   "transformers", "openai", "httpx", "tokenizers", "numpy", "scipy", "nvidia-nccl-cu13"):
            try:
                _pk[_p] = _md.version(_p)
            except Exception:
                _pk[_p] = None
        ENV_RECORD["packages"] = _pk
        ENV_RECORD["vllm_wheels_in_inputs"] = sorted(os.path.basename(p) for p in find_inputs(
            "vllm-*.whl", f"{INPUT}/datasets/*/vllm-wheels/wheels/wheels/vllm-*.whl",
            f"{INPUT}/vllm-wheels/wheels/wheels/vllm-*.whl"))
        # the files his weight loading of E128 runs (Kaggle Model copy): the DeepSeek-V4.1 model package (GPTQ
        # experts, Engram tables, quant config), the model loader, Humming's MoE / linear methods
        _vroot = os.path.dirname(_md.distribution("vllm").locate_file("vllm/__init__.py"))
        _files = sorted(glob.glob(f"{_vroot}/models/deepseek_v41/**/*.py", recursive=True)
                        + glob.glob(f"{_vroot}/model_executor/model_loader/*.py")
                        + glob.glob(f"{_vroot}/model_executor/layers/quantization/humming.py")
                        + glob.glob(f"{_vroot}/model_executor/layers/fused_moe/experts/fused_humming_moe.py"))
        ENV_RECORD["weight_loading_sha256"] = {os.path.relpath(f, os.path.dirname(_vroot)):
                                               hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in _files}
        rc, out = sh("nvcc --version | tail -1; nvidia-smi --query-gpu=driver_version,power.limit --format=csv,noheader "
                     "| head -1; free -g | head -2; df -h /tmp /kaggle/working /dev/shm | tail -3; nproc; "
                     "cat /sys/fs/cgroup/memory.max", timeout=120, env=BASE_ENV)
        ENV_RECORD["system"] = out.strip()
        print(out, flush=True)
    except Exception as exc:  # no working vLLM: keep the fallback submission
        ENV_RECORD["error"] = f"{type(exc).__name__}: {exc}"
        print("Manh's environment unusable, fallback only:", ENV_RECORD["error"], flush=True)
        MODE = "fallback"
with open(f"{WORK}/b32m_env.json", "w") as _f:
    json.dump(ENV_RECORD, _f, indent=1)
print("B32-M environment:", {k: v for k, v in ENV_RECORD.items() if k not in ("weight_loading_sha256", "system")},
      flush=True)
# The tool code runs on the image's Python (scipy etc.), as in B32.
TOOL_PY = os.environ.get("ARC_TOOL_PYTHON", sys.executable)
if MODE != "fallback":
    try:
        r = subprocess.run([TOOL_PY, "-I", "-c", "import numpy, scipy.ndimage; print(numpy.__version__)"],
                           env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1"}, capture_output=True,
                           text=True, timeout=300)
        print("tool python", TOOL_PY, "->", r.returncode, (r.stdout + r.stderr).strip()[-300:], flush=True)
    except Exception as exc:
        print("tool python check failed:", exc, flush=True)
BASE_ENV.pop("DSV41_SYNTH_INIT", None)
BASE_ENV["HOME"] = SUB_HOME  # /root/.cache is read-only on Kaggle with a model attached
BASE_ENV["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # as B32 and Manh's launcher
os.environ.setdefault("ARC_SMOKE_SANDBOXES", "16")  # cell 2b (commit runs only): kept small for the smoke
