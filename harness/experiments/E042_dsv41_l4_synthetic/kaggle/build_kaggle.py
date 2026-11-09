"""E042 on Kaggle 4xL4: build the wheel-source notebook, the kit dataset and the GPU benchmark notebook.

    .venv/bin/python experiments/E042_dsv41_l4_synthetic/kaggle/build_kaggle.py
    kaggle kernels push -p .../kaggle/wheels_nb          # CPU, internet on: fork wheels + deps at the box versions
    kaggle datasets create -p .../kaggle/kit             # private: model metadata, patches, scripts, JIT cache
    kaggle kernels push -p .../kaggle/gpu_nb             # 4xL4 (competition attached), internet off

The wheel notebook downloads exactly the package set of the Vast box venv (`requirements.box.freeze`, pip freeze
of /root/dsv41/venv): the two fork wheels from the publisher release (sha256-checked against the plan's lock) and
everything else with `pip download --no-deps` (PyPI + the PyTorch cu130 index), checks an offline resolve, and
installs it into a CPU venv to check the imports.

The GPU notebook installs that set offline into /tmp/dsv41/venv, applies the same patches as the box
(router shapes, compact Engram, synthetic init, PR #154 env-gated), restores the FlashInfer JIT modules compiled on
the box (the fork JIT-compiles two modules with nvcc; the pip CUDA 13 toolkit behind a CUDA_HOME shim is the
fallback), and runs (`GPU_CELLS`): realistic tool mode with PR #154, prefix caching on (run 9: E64 x 96 then x 80; run 8: E64 x 64 slots;
runs 6-7: E48 x 128)
(`tool_replay.py` over the E030 attempt shapes, KV and cache-hit monitor every 30 s), with
`--prefix-cache-retention-interval 128` so generated-token blocks are reusable. Kept for reference: run 5
(`GPU_CELLS_RUN5`: long-context decode 1K x 128, 32K x 64, 32K x 128) and run 4 (`GPU_CELLS_RUN4`: NCCL and indexer
benches, base and PR #154 grids at 1K-32K, profiles at C=64/128).
A watchdog kills the server at HARD_STOP_S of execution and a point starts only if its estimated time fits, so
the run cannot outlast its quota estimate. Everything lands in /kaggle/working/e042.
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
USER = "chankhavu"
WHEEL_SLUG = "dsv41-sm89-fork-py312-cu130"
KIT_SLUG = "e042-dsv41-kit"
GPU_SLUG = "e042-dsv41-l4-synthetic"
# Python 3.12 Kaggle image used by our other wheel sources and L4 notebooks (kaggle/build_notebook.SGL_DOCKER).
DOCKER = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"
REL = "https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130/"
FORK_WHEELS = [
    ("vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl",
     "e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed"),
    ("flashinfer_python-0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213-py3-none-any.whl",
     "078aa7682d699c67dce619fc8acf29ec41934ab5ba916268d1fe6f8850459de8"),
]
FORK_PINS = ["vllm==0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130",
             "flashinfer-python==0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213"]


def notebook(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": c.strip("\n").splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def write_kernel(outdir, slug, cells, **meta):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{slug}.ipynb").write_text(json.dumps(notebook(cells), indent=1))
    md = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
          "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
          "enable_internet": False, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
          "model_sources": [], "docker_image": DOCKER, "docker_image_pinning_type": "original"}
    md.update(meta)
    (outdir / "kernel-metadata.json").write_text(json.dumps(md, indent=1))


# ---------------------------------------------------------------- wheel source (CPU, internet on)
def wheel_cells():
    freeze = (HERE / "requirements.box.freeze").read_text().splitlines()
    pypi = [l for l in freeze if l.strip() and " @ " not in l]
    assert len(pypi) == len(freeze) - 2, "expected exactly the two fork wheels as URL requirements"
    lock = sorted(pypi + FORK_PINS, key=str.lower)
    return [
        """import sys, platform, subprocess
print(sys.version); print(platform.libc_ver())
print(subprocess.run('df -h /kaggle/working /tmp; nproc; free -g; uv --version', shell=True, capture_output=True, text=True).stdout)""",
        f"""import os
W = '/kaggle/working/wheels'; os.makedirs(W, exist_ok=True); os.makedirs('/kaggle/working/sdist', exist_ok=True)
# Exact package set of the Vast box venv (pip freeze), minus the two fork wheels (fetched by URL below).
open('/kaggle/working/pypi.txt', 'w').write({chr(10).join(pypi) + chr(10)!r})
open(W + '/requirements.lock', 'w').write({chr(10).join(lock) + chr(10)!r})
print(len({pypi!r}), 'pypi pins')""",
        f"""# 1. Fork wheels from the publisher release, checked against the plan's software lock
import hashlib, urllib.request, os
for name, sha in {FORK_WHEELS!r}:
    dst = os.path.join(W, name)
    urllib.request.urlretrieve({REL!r} + name.replace('+', '%2B'), dst)
    h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
    print(name, os.path.getsize(dst), 'sha256 ok' if h == sha else 'SHA256 MISMATCH ' + h)
    assert h == sha""",
        """# 2. Everything else at the exact versions (no dependency resolution: the freeze is complete)
! pip download --no-deps --dest /kaggle/working/wheels --extra-index-url https://download.pytorch.org/whl/cu130 -r /kaggle/working/pypi.txt 2>&1 | tail -n 15""",
        """# 3. Build any sdists into wheels
import glob, os, subprocess, sys, shutil
for f in glob.glob(W + '/*.tar.gz') + glob.glob(W + '/*.zip'):
    print('building', os.path.basename(f))
    r = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', W, f], capture_output=True, text=True)
    print(r.stdout[-600:], r.stderr[-600:])
    shutil.move(f, '/kaggle/working/sdist/')
print(len(glob.glob(W + '/*.whl')), 'wheels')""",
        """# 4. Offline resolve against the wheel dir only, hashes, manifest
import subprocess, sys, json, hashlib, glob, os
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed', '--no-index', '--find-links', W,
                    '-r', W + '/requirements.lock'], capture_output=True, text=True)
print('offline resolve rc', r.returncode); print(r.stdout[-1200:]); print(r.stderr[-2500:])
sums, total = [], 0
for f in sorted(glob.glob(W + '/*.whl')):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    sums.append(f'{h.hexdigest()}  {os.path.basename(f)}'); total += os.path.getsize(f)
open(W + '/SHA256SUMS', 'w').write('\\n'.join(sums) + '\\n')
json.dump({'n_wheels': len(sums), 'bytes': total, 'python': sys.version, 'resolve_rc': r.returncode},
          open(W + '/manifest.json', 'w'), indent=1)
print(len(sums), 'wheels;', round(total / 1e9, 2), 'GB')""",
        """# 5. Offline install into a CPU venv, exactly as the GPU notebook does; check imports
import subprocess, sys, time
t = time.time()
# --no-deps: the lock is the box's complete freeze; the fork's vllm metadata names flashinfer by URL, which uv
# rejects during resolution.
r = subprocess.run(f'uv venv --python {sys.executable} /tmp/venv && uv pip install --python /tmp/venv/bin/python --no-index '
                   f'--no-deps --find-links {W} -r {W}/requirements.lock', shell=True, capture_output=True, text=True)
print('install rc', r.returncode, round(time.time() - t), 's'); print(r.stdout[-800:]); print(r.stderr[-2500:])
r = subprocess.run(['/tmp/venv/bin/python', '-c', 'import torch, vllm, flashinfer, triton, transformers; '
                    'print(torch.__version__, torch.version.cuda, vllm.__version__, flashinfer.__version__, triton.__version__, transformers.__version__)'],
                   capture_output=True, text=True)
print(r.stdout, r.stderr[-2500:])
open('/kaggle/working/import_check.txt', 'w').write(r.stdout + r.stderr)
import shutil; shutil.rmtree('/tmp/venv', ignore_errors=True)""",
    ]


# ---------------------------------------------------------------- kit dataset
def build_kit():
    """Flat directory (Kaggle stores uploaded subdirectories as archives)."""
    kit = HERE / "kit"
    if kit.exists():
        shutil.rmtree(kit)
    kit.mkdir()
    for f in ("apply_patches.py", "synth_init.py", "pr154_apply.py", "pr154/pr154_kernel.diff"):
        shutil.copy(EXP / "patches" / f, kit)
    # PR #154 + E042 env gate applied to the box wheel's original file (pr154_apply.py), and that original's sha256.
    shutil.copy(HERE / "box_artifacts/dsv41_indexer.pr154.py", kit)
    import hashlib
    h = hashlib.sha256((HERE / "box_artifacts/dsv41_indexer.py.orig").read_bytes()).hexdigest()
    (kit / "dsv41_indexer.orig.sha256").write_text(h + "  dsv41_indexer.py\n")
    for f in ("decode_window.py", "indexer_bench.py", "nccl_allreduce_bench.py", "prof_summary.py", "tool_replay.py"):
        shutil.copy(EXP / "scripts" / f, kit)
    shutil.copy(EXP / "metadata/trace_shapes/tools_e030_hdrmin.json", kit)  # token counts only
    for f in ("model-e64-meta.tgz", "flashinfer_jit_cache_fork_sm89.tgz"):  # copied from the box beforehand
        shutil.copy(HERE / "box_artifacts" / f, kit)
    for f in ("config.json", "synthetic_manifest.json"):  # E48 = the E64 directory with these two replaced
        shutil.copy(EXP / "metadata/synthetic-e48" / f, kit / f"e48_{f}")
    (kit / "dataset-metadata.json").write_text(json.dumps(
        {"title": KIT_SLUG, "id": f"{USER}/{KIT_SLUG}", "licenses": [{"name": "other"}]}, indent=1))


# ---------------------------------------------------------------- GPU notebook
GPU_SETUP = r'''
import os, sys, time, json, glob, subprocess, shutil, signal, threading, urllib.request
T0 = time.time()
OUT = '/kaggle/working/e042'; os.makedirs(OUT, exist_ok=True)
# Hard cap on execution time (the watchdog in the server cell kills the server here; the finish cell still runs).
# Kaggle may hold a push in "RUNNING" for hours before this cell starts: T0 is the real start.
HARD_STOP_S = 11400                  # run 9: ≈ 3.2 h, two tool points (run 8: 5400, run 7: 5600, run 6: 6000, run 5: 7200)
NEW_PHASE_UNTIL_S = HARD_STOP_S      # run 4 cells: no new phase after this

def log(*a):
    s = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(f'{OUT}/run.log', 'a') as fh:
        fh.write(s + '\n')

def sh(cmd, timeout=None, env=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, f'TIMEOUT after {timeout}s'
    return r.returncode, (r.stdout or '') + (r.stderr or '')

def time_left():
    return NEW_PHASE_UNTIL_S - (time.time() - T0)

# Host driver libcuda first (the image also carries an older compat libcuda); nvidia-smi needs it too.
real = '/usr/local/nvidia/lib64'
if os.path.exists(real + '/libcuda.so'):
    os.environ['LD_LIBRARY_PATH'] = real + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_PRELOAD'] = real + '/libcuda.so'
for cmd in ['nvidia-smi', 'nvidia-smi topo -m', 'nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE | head -150',
            'free -g', 'df -h /tmp /kaggle/working', 'nproc', "grep -m1 'model name' /proc/cpuinfo",
            'cat /sys/fs/cgroup/memory.max 2>/dev/null', 'uv --version', 'python --version']:
    rc, out = sh(cmd, timeout=120)
    open(f'{OUT}/env.txt', 'a').write(f'$ {cmd}\n{out}\n')
    log('$', cmd, '\n' + out[-3000:])
'''

GPU_INSTALL = r'''
def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None
WHEELS = os.path.dirname(find_one('/kaggle/input/**/vllm-0.28.1rc1*.whl'))
KIT = os.path.dirname(find_one('/kaggle/input/**/model-e64-meta.tgz'))
log('wheels', WHEELS, 'kit', KIT)
V = '/tmp/dsv41/venv'; PY = f'{V}/bin/python'; SP = f'{V}/lib/python3.12/site-packages'
os.makedirs('/tmp/dsv41', exist_ok=True)
t = time.time()
if shutil.which('uv'):
    # --no-deps: the lock is the box's complete freeze (the fork's vllm names flashinfer by URL; uv rejects that).
    rc, out = sh(f'uv venv --python {sys.executable} {V} && uv pip install --python {PY} --no-index --no-deps '
                 f'--find-links {WHEELS} -r {WHEELS}/requirements.lock', timeout=3600)
else:
    rc, out = sh(f'{sys.executable} -m venv {V} && {PY} -m pip install --no-index --no-deps --find-links {WHEELS} '
                 f'-r {WHEELS}/requirements.lock', timeout=3600)
log('offline install rc', rc, f'{time.time() - t:.0f}s', out[-1500:])
assert rc == 0
for cmd in [f'{PY} -I {KIT}/apply_patches.py {SP}/vllm',
            f'mkdir -p ~/.cache/flashinfer && tar xzf {KIT}/flashinfer_jit_cache_fork_sm89.tgz -C ~/.cache/flashinfer',
            f'mkdir -p /tmp/dsv41/model-e64 && tar xzf {KIT}/model-e64-meta.tgz -C /tmp/dsv41/model-e64 && ls -la /tmp/dsv41/model-e64',
            f'cp -r /tmp/dsv41/model-e64 /tmp/dsv41/model-e48 && cp {KIT}/e48_config.json /tmp/dsv41/model-e48/config.json && '
            f'cp {KIT}/e48_synthetic_manifest.json /tmp/dsv41/model-e48/synthetic_manifest.json && '
            f'grep -o \'"n_routed_experts": [0-9]*\' /tmp/dsv41/model-e48/config.json']:
    rc, out = sh(cmd, timeout=600)
    log('$', cmd, 'rc', rc, out[-1200:])
    assert rc == 0

# PR #154 (env-gated): the file pre-patched from the box's original, installed only if the wheel's file is that
# original (no dependency on `patch` in the image).
import hashlib
IDX = f'{SP}/vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py'
orig_ok = hashlib.sha256(open(IDX, 'rb').read()).hexdigest() == open(f'{KIT}/dsv41_indexer.orig.sha256').read().split()[0]
if orig_ok:
    shutil.copy(f'{KIT}/dsv41_indexer.pr154.py', IDX)
log('PR #154 installed (env-gated):', orig_ok)

# CUDA 13 toolkit from the pip wheels behind a CUDA_HOME shim (no nvcc in the image), for any FlashInfer JIT not
# covered by the restored cache (kaggle/src/solver.cuda_toolkit_env, pointed at the venv).
cu = f'{SP}/nvidia/cu13'; shim = '/tmp/cuda13'
os.makedirs(f'{shim}/lib64', exist_ok=True)
for d in ('bin', 'include', 'nvvm', 'lib', 'share'):
    if os.path.exists(f'{cu}/{d}') and not os.path.lexists(f'{shim}/{d}'):
        os.symlink(f'{cu}/{d}', f'{shim}/{d}')
for f in os.listdir(f'{cu}/lib'):
    if not os.path.lexists(f'{shim}/lib64/{f}'):
        os.symlink(f'{cu}/lib/{f}', f'{shim}/lib64/{f}')
for name in ('cudart', 'nvrtc'):
    real = sorted(glob.glob(f'{cu}/lib/lib{name}.so.*'))
    if real and not os.path.lexists(f'{shim}/lib64/lib{name}.so'):
        os.symlink(real[0], f'{shim}/lib64/lib{name}.so')
# FlashInfer JIT links with -lcuda; the pip toolkit has no stub, so point lib64/stubs/libcuda.so at the host driver.
drv = next(iter(sorted(glob.glob('/usr/local/nvidia/lib64/libcuda.so*')) + sorted(glob.glob('/usr/lib/x86_64-linux-gnu/libcuda.so*'))), None)
os.makedirs(f'{shim}/lib64/stubs', exist_ok=True)
for dst in (f'{shim}/lib64/stubs/libcuda.so', f'{shim}/lib64/libcuda.so'):
    if drv and not os.path.lexists(dst):
        os.symlink(drv, dst)
log('libcuda for JIT linking:', drv)
BASE_ENV = dict(os.environ)
BASE_ENV.update(CUDA_HOME=shim, CUDA_PATH=shim, PATH=f'{V}/bin:{shim}/bin:' + os.environ['PATH'],
                LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH', '') + f':{shim}/lib64',
                VLLM_USE_V2_MODEL_RUNNER='1', FLASHINFER_CUDA_ARCH_LIST='8.9', TORCH_CUDA_ARCH_LIST='8.9',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DSV41_SYNTH_INIT='1',
                # FlashInfer's sampling module cannot JIT here: the pip nvcc (13.4) fails CCCL's check against the
                # pip CUDA 13.0 headers, and the box-built .so needs GLIBCXX_3.4.32 (newer than this image's libstdc++).
                # The benchmark decodes greedily (argmax), so vLLM's PyTorch top-k/top-p path is never on the hot path.
                VLLM_USE_FLASHINFER_SAMPLER='0')
for k in ('VLLM_MARLIN_INPUT_DTYPE', 'TRITON_PTXAS_BLACKWELL_PATH', 'NCCL_P2P_LEVEL'):
    BASE_ENV.pop(k, None)
mem_gb = int(open('/proc/meminfo').read().split()[1]) / 2**20
# 4M rows x 264 B = 0.98 GiB pinned per rank and layer (7.9 GiB total), as on the box; halve it on a small host.
BASE_ENV['DSV41_ENGRAM_SYNTH_ROWS'] = '4000000' if mem_gb > 60 else '2000000'
rc, out = sh(f'nvcc --version | tail -2; {PY} -c "import torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda, '
             f'torch.cuda.device_count(), vllm.__version__, flashinfer.__version__)"', timeout=900, env=BASE_ENV)
log('toolchain/imports:', out[-800:], '| host RAM GiB', round(mem_gb, 1), '| engram rows', BASE_ENV['DSV41_ENGRAM_SYNTH_ROWS'])
sh(f'uv pip freeze --python {PY} > {OUT}/pip_freeze.txt 2>&1 || {PY} -m pip freeze > {OUT}/pip_freeze.txt', env=BASE_ENV)
'''

GPU_MICRO = r'''
# NCCL bf16 all-reduce at TP message sizes (default transport, as vLLM uses it) and the PR #154 indexer bench.
if os.environ.get('E042_NCCL_BENCH', '0') == '1':  # done in v1 (nccl_default.json)
  rc, out = sh(f'{V}/bin/torchrun --nproc-per-node 4 {KIT}/nccl_allreduce_bench.py --label default '
             f'--out {OUT}/nccl_default.json 2>&1 | grep -E "T=|Error|error" | tail -20', timeout=900, env=BASE_ENV)
  log('nccl bench rc', rc, '\n' + out[-2500:])
env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES='0')
rc, out = sh(f'{PY} {KIT}/indexer_bench.py --json {OUT}/indexer_bench.json 2>&1 | tail -16', timeout=1800, env=env)
log('indexer bench rc', rc, '\n' + out[-3000:])
'''

GPU_SERVER = r'''
MODEL = '/tmp/dsv41/model-e64'
def serve_cmd(extra, model=None):
    return [f'{V}/bin/vllm', 'serve', model or MODEL, '--served-model-name', 'dsv41-e64', '--host', '127.0.0.1', '--port', '8100',
            '--trust-remote-code', '--tensor-parallel-size', '4', '--distributed-executor-backend', 'mp',
            '--kv-cache-dtype', 'fp8', '--block-size', '128', '--max-model-len', '65536', '--max-num-seqs', '128',
            '--max-num-batched-tokens', '2048', '--gpu-memory-utilization', '0.88', '--no-enable-prefix-caching',
            '--engram-config', '{"cpu_offload":true}', '--load-format', 'dummy', '--tokenizer-mode', 'deepseek_v41',
            '--language-model-only', '--compilation-config', '{"cudagraph_mode":"FULL"}'] + extra

class GpuMon(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True); self.path, self.stop = path, threading.Event()
    def run(self):
        q = 'timestamp,index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw'
        with open(self.path, 'a') as fh:
            while not self.stop.is_set():
                fh.write(sh(f'nvidia-smi --query-gpu={q} --format=csv,noheader', timeout=30)[1]); fh.flush()
                self.stop.wait(2)

def ready():
    try:
        urllib.request.urlopen('http://127.0.0.1:8100/v1/models', timeout=3).read(); return True
    except Exception:
        return False

LIVE = {}  # process groups the watchdog kills at HARD_STOP_S

def watchdog():
    while time.time() - T0 < HARD_STOP_S:
        time.sleep(10)
    log('WATCHDOG: hard stop, killing', list(LIVE))
    for p in list(LIVE.values()):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
threading.Thread(target=watchdog, daemon=True).start()

def run_bounded(cmd, timeout, env=None):
    """Shell command in its own process group; on timeout the whole group dies (subprocess.run would leave a
    pipeline's children holding the pipe)."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                         start_new_session=True)
    LIVE[cmd[:60]] = p
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); out, _ = p.communicate()
        out = (out or '') + f'\nTIMEOUT after {timeout}s'
    finally:
        LIVE.pop(cmd[:60], None)
    return p.returncode, out or ''

def start(name, extra=(), env_extra=None, timeout=2700, model=None):
    d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True)
    env = dict(BASE_ENV, DSV41_SYNTH_MANIFEST=f'{d}/synth', **(env_extra or {}))
    cmd = serve_cmd(list(extra), model)
    open(f'{d}/cmd.txt', 'w').write(' '.join(cmd) + '\n' + json.dumps(env_extra or {}) + '\n')
    proc = subprocess.Popen(cmd, env=env, stdout=open(f'{d}/server.log', 'w'), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=d)
    LIVE['server'] = proc
    mon = GpuMon(f'{d}/gpu.csv'); mon.start()
    t = time.time()
    while not ready():
        time.sleep(10)
        if proc.poll() is not None or time.time() - t > timeout:
            log(name, 'SERVER FAILED', proc.poll(), sh(f'grep -E "Error|error" {d}/server.log | tail -5')[1][-1500:])
            stop(proc, mon); return None, None
    kv = sh(f"grep -o 'GPU KV cache size: [0-9,]* tokens' {d}/server.log | tail -1")[1].strip()
    log(name, f'up after {time.time() - t:.0f}s', kv)
    return proc, mon

def stop(proc, mon):
    # Our own server only: its process group (start_new_session=True), never by name.
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(30)
    LIVE.pop('server', None)
    if mon is not None:
        mon.stop.set()
    for _ in range(60):
        if sh('nvidia-smi --query-compute-apps=pid --format=csv,noheader', timeout=30)[1].strip() == '':
            break
        time.sleep(2)

def grid(name, points, window=30):
    rows = []
    for ctx, c in points:
        if time_left() < 0:
            log(name, 'out of time, skipping', ctx, c); break
        rc, out = sh(f'{PY} {KIT}/decode_window.py --ctx {ctx} --conc {c} --window {window} --seed {c} '
                     f'--json {OUT}/{name}/dw_ctx{ctx}_c{c}.json 2>&1 | tail -1', timeout=3600, env=BASE_ENV)
        try:
            r = json.loads(out.strip().splitlines()[-1]); r.pop('first_token_s', None); rows.append(r)
            log(name, f'ctx {ctx} C {c}: decode {r["decode_tok_s"]} tok/s, ITL {r["itl_median_ms"]} ms, '
                      f'prefill {r["prefill_tok_s"]} tok/s')
        except Exception:
            log(name, 'grid point failed', ctx, c, out[-800:])
    json.dump(rows, open(f'{OUT}/{name}/grid.json', 'w'), indent=1)
    return rows

def smoke(name):
    body = json.dumps({'model': 'dsv41-e64', 'prompt': 'The quick brown fox jumps over the lazy dog. Count to ten:',
                       'max_tokens': 32, 'temperature': 0, 'ignore_eos': True, 'logprobs': 1}).encode()
    req = urllib.request.Request('http://127.0.0.1:8100/v1/completions', body, {'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    lp = d['choices'][0]['logprobs']['token_logprobs']
    import math
    log(name, 'smoke: tokens', d['usage']['completion_tokens'], 'logprobs finite', all(math.isfinite(x) for x in lp))
'''

GPU_RUNS = r'''
SUMMARY = {}
GRID_BASE = [(1024, 1), (1024, 8), (1024, 32), (1024, 64), (1024, 128), (8192, 32), (32768, 8)]
GRID_AB = [(1024, 64), (1024, 128), (8192, 32)]
for name, env_extra, points in [('base', {}, GRID_BASE), ('pred', {'DSV41_INDEXER_Q_PREDECODE': '1'}, GRID_AB)]:
    if time_left() < 0:
        log('out of time before', name); break
    proc, mon = start(name, env_extra=env_extra)
    try:
        if proc is not None:
            smoke(name)
            SUMMARY[name] = grid(name, points)
    except Exception as exc:
        log(name, 'ERROR', repr(exc))
    finally:
        stop(proc, mon)
    json.dump(SUMMARY, open(f'{OUT}/summary.json', 'w'), indent=1)
'''

GPU_PROFILE = r'''
# Decode profiles (one capture per server): C=64 (compare with the box's P3) and C=128.
for c in (64, 128):
    if time_left() < 0:
        log('out of time before profile', c); break
    name = f'prof_c{c}'; tdir = f'{OUT}/{name}/trace'; os.makedirs(tdir, exist_ok=True)
    pc = json.dumps({'profiler': 'torch', 'torch_profiler_dir': tdir, 'torch_profiler_with_stack': False,
                     'warmup_iterations': 3, 'active_iterations': 10})
    proc, mon = start(name, extra=['--profiler-config', pc])
    try:
        if proc is not None:
            dw = subprocess.Popen(f'{PY} {KIT}/decode_window.py --ctx 1024 --conc {c} --window 150 --seed {500 + c} '
                                  f'--json {OUT}/{name}/dw.json > {OUT}/{name}/dw.txt 2>&1', shell=True, env=BASE_ENV)
            time.sleep(c * 1024 / 700 + 30)
            post = lambda p: urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:8100/{p}', b'', method='POST'), timeout=900).read()
            post('start_profile'); time.sleep(20); post('stop_profile')
            dw.wait(1200)
            log(name, 'window (profiler on):', open(f'{OUT}/{name}/dw.txt').read()[-400:])
            rc, out = sh(f'{PY} {KIT}/prof_summary.py {tdir} --indexer-per-step 8 --json {OUT}/{name}/summary.json', timeout=1800)
            log(name, 'profile summary:\n' + out[-4000:])
    except Exception as exc:
        log(name, 'ERROR', repr(exc))
    finally:
        stop(proc, mon)
'''

GPU_LONG = r'''
# Run 5: long-context decode, E48 + PR #154, one server. Each point prefills C x ctx random tokens (~1100-1250 tok/s
# here), then decode_window captures a 10-step profile (--profile-s) and measures a 60 s window.
POINTS = [(1024, 128), (32768, 64), (32768, 128)]
PREFILL_TOK_S = 1000           # conservative (run 4: 1073-1256), for the fits-before-hard-stop check
name = 'long_e48'; tdir = f'{OUT}/{name}/trace'; os.makedirs(tdir, exist_ok=True)
pc = json.dumps({'profiler': 'torch', 'torch_profiler_dir': tdir, 'torch_profiler_with_stack': False,
                 'warmup_iterations': 3, 'active_iterations': 10})
SUMMARY = {name: []}
proc, mon = start(name, extra=['--profiler-config', pc], env_extra={'DSV41_INDEXER_Q_PREDECODE': '1'},
                  model='/tmp/dsv41/model-e48', timeout=1500)
try:
    if proc is not None:
        smoke(name)
        for ctx, c in POINTS:
            est = c * ctx / PREFILL_TOK_S + 240
            if time_left() < est:
                log(name, f'skip ctx {ctx} C {c}: needs ~{est:.0f}s, {time_left():.0f}s left'); continue
            log(name, f'ctx {ctx} C {c}: start (est {est:.0f}s)')
            rc, out = run_bounded(f'{PY} {KIT}/decode_window.py --ctx {ctx} --conc {c} --window 60 --seed {c} '
                                  f'--sock-read 10800 --profile-s 20 --json {OUT}/{name}/dw_ctx{ctx}_c{c}.json 2>&1 | tail -1',
                                  timeout=max(60, time_left()), env=BASE_ENV)
            try:
                r = json.loads(out.strip().splitlines()[-1]); r.pop('first_token_s', None); SUMMARY[name].append(r)
                log(name, f'ctx {ctx} C {c}: decode {r["decode_tok_s"]} tok/s, ITL {r["itl_median_ms"]} ms, '
                          f'prefill {r["prefill_tok_s"]} tok/s, profile {r.get("profile")}')
                if ctx >= 8192:  # long-prompt prefill rate as measured here, with a margin
                    PREFILL_TOK_S = 0.9 * r['prefill_tok_s']
            except Exception:
                log(name, 'point failed', ctx, c, out[-800:])
            pdir = f'{tdir}/ctx{ctx}_c{c}'; os.makedirs(pdir, exist_ok=True)
            for f in glob.glob(f'{tdir}/*'):
                if os.path.isfile(f):
                    shutil.move(f, pdir)
            rc, out = run_bounded(f'{PY} {KIT}/prof_summary.py {pdir} --indexer-per-step 8 --json {pdir}/summary.json',
                                  timeout=600)
            log(name, f'ctx {ctx} C {c} profile:', ' '.join(l for l in out.splitlines() if 'per_step_ms' in l)[:1500])
            json.dump(SUMMARY, open(f'{OUT}/summary.json', 'w'), indent=1)
except Exception as exc:
    log(name, 'ERROR', repr(exc))
finally:
    stop(proc, mon)
'''

GPU_TOOLS = r'''
# Runs 6-8: realistic tool mode. E030 attempt shapes (sub80s40 + header): slots start mid-attempt, run attempts back
# to back with a 1 s tool delay; the window fills what is left before the hard stop. PR #154, prefix caching on.
# Run 9: E64 x 96, then E64 x 80 (run 8's E64 x 64 peaked at 73% KV, so 64 under-fills it vs run 7's E48 x 128).
# Each point gets a fresh server (empty prefix cache) and run 8's replay settings; (model, C, replay wall cap or None
# for whatever is left). Run 8: E64 x 64; runs 6-7: E48 x 128.
TOOLS_POINTS = [('e64', 96, 5200), ('e64', 80, None)]
POINT_MIN_S = 4000  # server start + setup + drain + warm-up + a 20-min window at C 80
class KvMon(threading.Thread):
    KEYS = ('vllm:kv_cache_usage_perc', 'vllm:num_requests_running', 'vllm:num_requests_waiting',
            'vllm:num_preemptions_total', 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total',
            'vllm:generation_tokens_total')
    def __init__(self, path):
        super().__init__(daemon=True); self.path, self.stop = path, threading.Event()
    def run(self):
        with open(self.path, 'a') as fh:
            while not self.stop.is_set():
                try:
                    txt = urllib.request.urlopen('http://127.0.0.1:8100/metrics', timeout=10).read().decode()
                    vals = {}
                    for line in txt.splitlines():
                        for k in self.KEYS:
                            if line.startswith(k + '{') or line.startswith(k + ' '):
                                vals[k[5:]] = vals.get(k[5:], 0.0) + float(line.split()[-1])
                    fh.write(json.dumps({'t': round(time.time() - T0), **vals}) + '\n'); fh.flush()
                except Exception as exc:
                    fh.write(json.dumps({'t': round(time.time() - T0), 'error': repr(exc)[:200]}) + '\n')
                self.stop.wait(30)

SUMMARY = {}
for TOOLS_MODEL, TOOLS_C, cap in TOOLS_POINTS:
    name = f'tools_{TOOLS_MODEL}_c{TOOLS_C}'; d = f'{OUT}/{name}'
    if time_left() < POINT_MIN_S:
        log(name, f'skipped: {time_left():.0f}s left'); continue
    SUMMARY[name] = []
    # --prefix-cache-retention-interval 128: keep every sliding-window block, so blocks filled by generated tokens are
    # reusable (the default 0 keeps SWA KV only at the prompt end, and a re-sent transcript hits only the old prompt; P7).
    proc, mon = start(name, extra=['--enable-prefix-caching', '--max-model-len', '163840', '--prefix-cache-retention-interval', '128'],
                      env_extra={'DSV41_INDEXER_Q_PREDECODE': '1'}, model=f'/tmp/dsv41/model-{TOOLS_MODEL}', timeout=1500)
    kvm = None
    try:
        if proc is not None:
            smoke(name)
            kvm = KvMon(f'{d}/kvmon.log'); kvm.start()
            budget = time_left() - 240
            if cap:
                budget = min(budget, cap)
            log(name, f'tool replay C {TOOLS_C}, wall budget {budget:.0f}s')
            rc, out = run_bounded(f'{PY} {KIT}/tool_replay.py --shapes {KIT}/tools_e030_hdrmin.json --conc {TOOLS_C} --warm-s 300 '
                                  f'--window-s 2400 --tool-delay 1 --seed 1 --max-wall-s {budget:.0f} '
                                  f'--json {d}/r2_tools_c{TOOLS_C}.json 2>&1 | tail -1', timeout=budget + 180, env=BASE_ENV)
            try:
                r = json.loads(out.strip().splitlines()[-1]); SUMMARY[name].append(r)
                log(name, 'tool replay:', json.dumps(r))
            except Exception:
                log(name, 'tool replay failed', out[-1500:])
            json.dump(SUMMARY, open(f'{OUT}/summary.json', 'w'), indent=1)
    except Exception as exc:
        log(name, 'ERROR', repr(exc))
    finally:
        if kvm is not None:
            kvm.stop.set()
        stop(proc, mon)
'''

GPU_FINISH = r'''
for d in glob.glob(f'{OUT}/*/server.log'):
    sh(f'gzip -f {d}')
log('E042 KAGGLE DONE', json.dumps(SUMMARY)[:3000])
'''


GPU_CELLS_RUN4 = [GPU_SETUP, GPU_INSTALL, GPU_MICRO, GPU_SERVER, GPU_RUNS, GPU_PROFILE, GPU_FINISH]
GPU_CELLS_RUN5 = [GPU_SETUP, GPU_INSTALL, GPU_SERVER, GPU_LONG, GPU_FINISH]
GPU_CELLS = [GPU_SETUP, GPU_INSTALL, GPU_SERVER, GPU_TOOLS, GPU_FINISH]


def main():
    write_kernel(HERE / "wheels_nb", WHEEL_SLUG, wheel_cells(), enable_internet=True)
    build_kit()
    write_kernel(HERE / "gpu_nb", GPU_SLUG, GPU_CELLS,
                 enable_gpu=True, machine_shape="NvidiaL4", competition_sources=["arc-prize-2026-arc-agi-2"],
                 kernel_sources=[f"{USER}/{WHEEL_SLUG}"], dataset_sources=[f"{USER}/{KIT_SLUG}"])
    print("built", HERE / "wheels_nb", HERE / "kit", HERE / "gpu_nb")


if __name__ == "__main__":
    main()
