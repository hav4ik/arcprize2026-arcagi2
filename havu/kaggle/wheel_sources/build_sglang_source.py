"""Build the SGLang wheel-source notebook (CPU, internet on), modelled on chankhavu/vllm-0-23-0-py312-cu128.

    .venv/bin/python kaggle/wheel_sources/build_sglang_source.py
    .venv/bin/kaggle kernels push -p kaggle/wheel_sources/sglang-0-5-20-py312-cu130

The notebook downloads SGLang 0.5.20 and all its dependencies as Python 3.12 wheels into /kaggle/working/wheels
(cuda-tile 1.6.0rc5 is sdist-only and is built into a wheel there), adds FlashInfer 0.6.18 cubin + cu130 JIT-cache
wheels (so nothing is fetched or JIT-compiled from the internet at runtime), verifies an offline resolve with
`pip install --no-index --dry-run`, and writes SHA256SUMS, requirements.lock and manifest.json.
GPU notebooks attach it as a kernel source and install with --no-index --find-links.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "sglang-0-5-20-py312-cu130"
# Same image as the user's vllm-0-23-0-py312-cu128 source notebook (Python 3.12).
DOCKER = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"

PKGS = "sglang==0.5.20"
FI = "0.6.18"

CELLS = [
    """import sys, platform, subprocess
print(sys.version); print(platform.libc_ver())
print(subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True).stdout)""",
    f"""! mkdir -p /kaggle/working/wheels /kaggle/working/sdist
# 1. SGLang + deps (binary wheels where they exist; sdists land in sdist/ and are built below)
! pip download --dest /kaggle/working/wheels --prefer-binary {PKGS} 2>&1 | tail -n 25""",
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
       'flashinfer-jit-cache', 'transformers', 'triton', 'cuda-tile', 'xgrammar')])""",
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
