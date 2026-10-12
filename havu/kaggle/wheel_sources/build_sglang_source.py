"""Build the SGLang wheel-source notebook (CPU, internet on), modelled on chankhavu/vllm-0-23-0-py312-cu128.

    .venv/bin/python kaggle/wheel_sources/build_sglang_source.py
    .venv/bin/kaggle kernels push -p kaggle/wheel_sources/sglang-0-5-20-py312-cu130

The notebook downloads SGLang 0.5.20 and all its dependencies as Python 3.12 wheels into /kaggle/working/wheels
(cuda-tile 1.6.0rc5 is sdist-only and is built into a wheel there), adds FlashInfer 0.6.18 cubin + cu130 JIT-cache
wheels (so nothing is fetched or JIT-compiled from the internet at runtime), verifies an offline resolve with
`pip install --no-index --dry-run`, and writes SHA256SUMS, requirements.lock and manifest.json.
GPU notebooks attach it as a kernel source and install with --no-index --find-links.

Version 2 (E014, 2026-10-12): pip download is constrained to the v1 lock (sglang-0-5-20-v1.requirements.lock), so
the package set is unchanged; the notebook then installs the wheels offline into this CPU image and checks the JIT
toolchain: the CUDA 13 pip toolkit (nvidia-cuda-nvcc/crt/nvvm/cccl + runtime headers, already in the lock) behind
the same CUDA_HOME shim the solver uses (solver.cuda_toolkit_env), and compiles sglang-kernel's fp8
per_token_group_quant JIT module for sm_89 (the kernel whose nvcc call killed the first L4 run). The compiled
modules are kept in /kaggle/working/sglang_jit_cache/sm89 (seed for SGLANG_JIT_CACHE_DIR) and the result in
toolchain_check.json. Runs on CPU only.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK_V1 = (HERE / "sglang-0-5-20-v1.requirements.lock").read_text()


def solver_function(name):
    """Source of one top-level function of kaggle/src/solver.py (single source of truth for the CUDA shim)."""
    src = (HERE.parent / "src/solver.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


TOOLCHAIN_CHECK = """import glob, json, os, subprocess, sys, time, traceback
def log(*a):
    print(*a, flush=True)
""" + solver_function("cuda_toolkit_env") + """
R = {}
shim = cuda_toolkit_env()
R['shim'] = shim
R['ls_usr_local_cuda'] = subprocess.run('ls -la /usr/local/ | grep -i cuda; ls /usr/local/cuda/bin 2>&1 | head', shell=True,
                                        capture_output=True, text=True).stdout
for cmd in ['nvcc --version', 'c++ --version', 'ls -la ' + str(shim) + ' ' + str(shim) + '/lib64/libcudart.so',
            'ls ' + str(shim) + '/include | head -50', 'ls ' + str(shim) + '/bin']:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    R[cmd] = (r.stdout + r.stderr)[-3000:]
    log('$', cmd); log(R[cmd])
os.environ['SGLANG_JIT_CACHE_DIR'] = '/kaggle/working/sglang_jit_cache'
try:
    import torch
    R['torch'] = [torch.__version__, torch.version.cuda]
    from sglang.kernels.jit.utils.arch import override_jit_cuda_arch
    import importlib
    P = importlib.import_module('sglang.kernels.ops.quantization.per_token_group_quant')  # module, not the function
    R['modules'] = []
    with override_jit_cuda_arch(8, 9):
        for row_major in (True, False):
            t = time.time()
            try:
                P._jit_module(torch.bfloat16, torch.float8_e4m3fn, 128, False, row_major, True, False, False, False)
                R['modules'].append({'row_major': row_major, 'ok': True, 's': round(time.time() - t, 1)})
            except Exception as exc:
                R['modules'].append({'row_major': row_major, 'ok': False, 's': round(time.time() - t, 1),
                                     'error': traceback.format_exc()[-6000:]})
            log(R['modules'][-1])
except Exception:
    R['fatal'] = traceback.format_exc()[-6000:]
    log(R['fatal'])
R['cache_files'] = sorted(os.path.relpath(p, '/kaggle/working/sglang_jit_cache')
                          for p in glob.glob('/kaggle/working/sglang_jit_cache/**', recursive=True))[:200]
R['ok'] = bool(R.get('modules')) and all(m['ok'] for m in R['modules'])
json.dump(R, open('/kaggle/working/toolchain_check.json', 'w'), indent=1)
log('TOOLCHAIN CHECK', 'PASS' if R['ok'] else 'FAIL')
"""
SLUG = "sglang-0-5-20-py312-cu130"
# Same image as the user's vllm-0-23-0-py312-cu128 source notebook (Python 3.12).
DOCKER = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"

PKGS = "sglang==0.5.20"
FI = "0.6.18"

CELLS = [
    """import sys, platform, subprocess
print(sys.version); print(platform.libc_ver())
print(subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True).stdout)""",
    "import os\nos.makedirs('/kaggle/working/wheels', exist_ok=True); os.makedirs('/kaggle/working/sdist', exist_ok=True)\n"
    "# v1 lock as constraints: the same package versions as source version 1\n"
    f"open('/kaggle/working/constraints-v1.txt', 'w').write({LOCK_V1!r})",
    f"""# 1. SGLang + deps (binary wheels where they exist; sdists land in sdist/ and are built below)
! pip download --dest /kaggle/working/wheels --prefer-binary -c /kaggle/working/constraints-v1.txt {PKGS} 2>&1 | tail -n 25""",
    """# 2. Build sdists into wheels (cuda-tile 1.6.0rc5 is sdist-only)
import glob, os, subprocess, sys, shutil
for f in glob.glob('/kaggle/working/wheels/*.tar.gz') + glob.glob('/kaggle/working/wheels/*.zip'):
    print('building', os.path.basename(f))
    r = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', '/kaggle/working/wheels', f],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    shutil.move(f, '/kaggle/working/sdist/')
print(len(glob.glob('/kaggle/working/wheels/*.whl')), 'wheels')""",
    f"""# 3. FlashInfer cubins + JIT cache for CUDA 13.0 (avoids runtime downloads / nvcc JIT offline)
! pip download --dest /kaggle/working/wheels --no-deps --find-links https://flashinfer.ai/whl/flashinfer-cubin/ flashinfer-cubin=={FI} 2>&1 | tail -n 3
! pip download --dest /kaggle/working/wheels --no-deps --index-url https://flashinfer.ai/whl/cu130 flashinfer-jit-cache=={FI} 2>&1 | tail -n 3""",
    f"""# 4. Offline resolve check against the wheel dir only + lock file + hashes
import subprocess, sys, json, hashlib, glob, os
W = '/kaggle/working/wheels'
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed', '--no-index', '--find-links', W,
                    '--report', '/kaggle/working/resolve-report.json', '{PKGS}', 'flashinfer-cubin=={FI}',
                    'flashinfer-jit-cache=={FI}'], capture_output=True, text=True)
print('offline resolve rc', r.returncode); print(r.stdout[-1500:]); print(r.stderr[-1500:])
rep = json.load(open('/kaggle/working/resolve-report.json'))
lock = sorted(f"{{i['metadata']['name']}}=={{i['metadata']['version']}}" for i in rep['install'])
open(f'{{W}}/requirements.lock', 'w').write('\\n'.join(lock) + '\\n')
sums, total = [], 0
for f in sorted(glob.glob(W + '/*.whl')):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    sums.append(f'{{h.hexdigest()}}  {{os.path.basename(f)}}'); total += os.path.getsize(f)
open(f'{{W}}/SHA256SUMS', 'w').write('\\n'.join(sums) + '\\n')
json.dump({{'packages': '{PKGS}', 'flashinfer': '{FI}', 'n_wheels': len(sums), 'bytes': total,
           'python': sys.version}}, open(f'{{W}}/manifest.json', 'w'), indent=1)
print(len(lock), 'packages in lock;', len(sums), 'wheels;', round(total / 1e9, 2), 'GB')
print([l for l in lock if l.split('==')[0] in ('sglang', 'sglang-kernel', 'torch', 'flashinfer-python', 'flashinfer-cubin',
       'flashinfer-jit-cache', 'transformers', 'triton', 'cuda-tile', 'xgrammar')])
v1 = set(open('/kaggle/working/constraints-v1.txt').read().split())
print('lock == v1 lock:', set(lock) == v1, '| only in v2:', sorted(set(lock) - v1), '| only in v1:', sorted(v1 - set(lock)))""",
    """# 5. Offline install into this (CPU) image, exactly as the GPU notebooks do (uv, --no-index)
import subprocess, sys, time
W = '/kaggle/working/wheels'
t = time.time()
r = subprocess.run(f'uv pip install --system --python {sys.executable} --no-index --find-links {W} -r {W}/requirements.lock',
                   shell=True, capture_output=True, text=True)
print('uv install rc', r.returncode, round(time.time() - t), 's'); print(r.stdout[-1500:]); print(r.stderr[-3000:])
r = subprocess.run([sys.executable, '-c', 'import torch, sglang, flashinfer, tvm_ffi; print(torch.__version__, torch.version.cuda, sglang.__version__, flashinfer.__version__)'],
                   capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])""",
    "# 6. JIT toolchain check (CUDA 13 pip toolkit behind the solver's CUDA_HOME shim; sm_89 fp8 quant kernel)\n"
    f"open('/kaggle/working/toolchain_check.py', 'w').write({TOOLCHAIN_CHECK!r})\n"
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, '/kaggle/working/toolchain_check.py'], capture_output=True, text=True)\n"
    "print(r.stdout[-12000:]); print(r.stderr[-6000:])",
]


def main():
    out = HERE / SLUG
    out.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c}
                    for c in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"chankhavu/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
            "model_sources": [], "docker_image": DOCKER, "docker_image_pinning_type": "original"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)


if __name__ == "__main__":
    main()
