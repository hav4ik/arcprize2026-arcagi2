#!/bin/bash
# pod 3: the exact submission notebook (C1 copy: only cell 1b's manifest differs) in a Kaggle-like layout, rerun mode.
# /kaggle/input mapped to local copies; cu13-install and unsloth-main deliberately NOT mapped (his CUDA cell then fails
# harmlessly in a rerun and leaves the pod's CUDA 13 alone; the unsloth installs are training-only).
set -euo pipefail
L=/root/e066/nbrun/setup.log; mkdir -p /root/e066/nbrun; exec > >(tee -a $L) 2>&1
echo "setup $(date -u +%T)"
# kernel (the venv's Python 3.13, as Kaggle's kernel is its image's Python 3.13) + papermill runner
/usr/bin/uv pip install --python /root/b32m/venv/bin/python ipykernel > /root/e066/nbrun/pip_ipykernel.log 2>&1
/root/kgl/bin/pip install -q papermill > /root/e066/nbrun/pip_papermill.log 2>&1
/root/b32m/venv/bin/python -m ipykernel install --prefix /root/kgl --name py313b32m --display-name "py313 b32m venv" > /dev/null
/root/kgl/bin/jupyter kernelspec list
# Kaggle layout
rm -rf /kaggle; mkdir -p /kaggle/working /kaggle/temp /kaggle/tmp /kaggle/input/datasets/nguyennguyen599 \
  /kaggle/input/datasets/chankhavu/dsv41-tools-kit /kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-img-e128-2bit/pytorch/default \
  /kaggle/input/competitions/arc-prize-2026-arc-agi-2
ln -s /root/khsrc/datasets/nguyennguyen599/arc2-2026-src/versions/22 /kaggle/input/datasets/nguyennguyen599/arc2-2026-src
ln -s /root/b32m/dl /kaggle/input/datasets/nguyennguyen599/vllm-wheels
cp /root/e066/nbrun/dsv41_kit.tgz /kaggle/input/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz
ln -s /root/models/c1 /kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1
cp /root/data/evalx2t_s2026_challenges.json /kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json
# path discovery as cell 0 does it
/root/b32m/venv/bin/python -c '
import glob
f=lambda p: sorted(glob.glob(p, recursive=True))[:1]
print("challenges", f("/kaggle/input/**/arc-agi_test_challenges.json")); print("kit", f("/kaggle/input/**/dsv41_kit.tgz"))
print("model", f("/kaggle/input/**/e043_manifest.json")); print("dsv41_vllm", f("/kaggle/input/datasets/nguyennguyen599/arc2-2026-src/dsv41_vllm.py"))
print("wheels", len(glob.glob("/kaggle/input/datasets/nguyennguyen599/vllm-wheels/wheels/wheels/*.whl")))'
echo "SETUP_DONE $(date -u +%T)"
