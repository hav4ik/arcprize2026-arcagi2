"""Start Manh's production server exactly as a notebook's cell 2m does (his dsv41_vllm.py launcher, arc2-2026-src v22,
the cell's flags and MANH_SERVER_ENV), outside the notebook, for A/B tests. Env AB_ENV_OVERRIDE (JSON) is applied
after MANH_SERVER_ENV (e.g. decode flags off). Usage: run_srv.py <notebook.ipynb> <work dir> <model dir>
Run with the kernel venv's python (/root/kenv/bin/python -I). Exits once /v1/models serves dsv41; the server stays."""
import json, os, shutil, subprocess, sys, time, glob
nb_path, WORK, MODEL_SRC = sys.argv[1:4]
nb = json.load(open(nb_path))
src = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("# Cell 2m"))
hook = "_env.update(MANH_SERVER_ENV)"
assert src.count(hook) == 1
src = src.replace(hook, hook + '; _env.update(json.loads(os.environ.get("AB_ENV_OVERRIDE") or "{}")); '
                  'print("AB env override:", os.environ.get("AB_ENV_OVERRIDE"), flush=True)')
os.makedirs(WORK, exist_ok=False)
S = "/kaggle/input/datasets/nguyennguyen599/arc2-2026-src"
for f in os.listdir(S):  # as Manh's cell 2: cp arc2-2026-src/* (top-level files)
    if os.path.isfile(f"{S}/{f}"):
        shutil.copy(f"{S}/{f}", WORK)
for d in ("patches", "third_party", "pipeline_runtime_bootstrap"):  # cp -r, as his cell 2
    shutil.copytree(f"{S}/{d}", f"{WORK}/{d}")
shutil.copytree(f"{S}/vllm_delta_mhar_plugin/vllm_delta_mhar_plugin", f"{WORK}/vllm_delta_mhar_plugin")
SCRATCH = "/tmp/dsv41"; _C = f"{SCRATCH}/cache"
for _k, _v in (("UV_CACHE_DIR", f"{_C}/uv"), ("XDG_CACHE_HOME", _C), ("VLLM_CACHE_ROOT", f"{_C}/vllm"),
               ("TRITON_CACHE_DIR", f"{_C}/triton"), ("TORCHINDUCTOR_CACHE_DIR", f"{_C}/inductor"),
               ("FLASHINFER_WORKSPACE_BASE", _C), ("TORCH_EXTENSIONS_DIR", f"{_C}/torch_ext"),
               ("HF_HOME", f"{_C}/hf"), ("CUDA_CACHE_PATH", f"{_C}/nv")):
    os.environ[_k] = _v; os.makedirs(_v, exist_ok=True)
os.environ.setdefault("ARC_PORT", "18084"); os.environ.setdefault("MANH_VISION", "0")
BASE_ENV = dict(os.environ)
_cuda = next((d for d in ("/usr/local/cuda-13.0", "/usr/local/cuda") if os.path.exists(f"{d}/bin/nvcc")), None)
BASE_ENV.update(CUDA_HOME=_cuda, CUDA_PATH=_cuda, PATH=f"{_cuda}/bin:" + BASE_ENV.get("PATH", ""))
g = {"__name__": "__cell__", "T0": time.time(), "MODE": "submit", "MODEL_SRC": MODEL_SRC, "STAGE_DONE": "",
     "WORK": WORK, "BASE_ENV": BASE_ENV, "PY": sys.executable, "RERUN": True, "SCRATCH": SCRATCH,
     "json": json, "os": os, "time": time, "subprocess": subprocess, "sys": sys, "glob": glob}
exec(compile(src, "cell_2m", "exec"), g)
print("RUN_SRV pid", g.get("MANH_SERVER_PID"), "log", g.get("MANH_SERVER_LOG"), flush=True)
