#!/bin/bash
# E046 E128 on 4x RTX PRO 6000 (sm120): fork venv as E042 box_install.sh steps 1-2, then the Kaggle vLLM overlay.
set -euo pipefail
D=/root/dsv41; B=$D/bundle; LOG=$D/runs/install.log
mkdir -p $D/runs $D/wheels
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%T)] install start"
REL=https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130
while read -r sha name; do
  [ -f $D/wheels/$name ] || curl -fsSL -o $D/wheels/$name "$REL/${name//+/%2B}"
  echo "$sha  $D/wheels/$name" | sha256sum -c -
done <<'EOW'
e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl
078aa7682d699c67dce619fc8acf29ec41934ab5ba916268d1fe6f8850459de8 flashinfer_python-0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213-py3-none-any.whl
EOW
grep -v ' @ ' $B/requirements.box.freeze > $D/wheels/pypi.txt
UV=/opt/instance-tools/bin/uv
[ -x $D/venv/bin/python ] || $UV venv --python /usr/bin/python3.12 $D/venv
PY=$D/venv/bin/python; SP=$D/venv/lib/python3.12/site-packages
$UV pip install --python $PY --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 -r $D/wheels/pypi.txt $D/wheels/*.whl
# Overlay (12 files incl. Humming backends), as the Kaggle notebook: extract over site-packages, then sha256 check
tar xzf $B/vllm_overlay_rw.tgz -C $SP
(cd $SP && sha256sum -c $B/vllm_overlay_rw.sha256)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc --version | tail -1
$PY -c "import importlib.metadata as m, torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), vllm.__version__, flashinfer.__version__, m.version('humming-kernels'))"
ls $SP/triton/backends/nvidia/bin/ | head
echo "[$(date -u +%T)] INSTALL OK"
