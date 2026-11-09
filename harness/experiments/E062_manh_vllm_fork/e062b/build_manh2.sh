#!/bin/bash
# E062b (on the 48 GB box): build the SM89 wheel of the local merge arc/e128-kaggle-main5e1b (95376a1 = Manh's main
# 5e1bccd07c + arc/e128-kaggle 786468e) in its own build venv /root/manh2/venv. CPU-capped (MAX_JOBS 32, one nvcc
# thread each, nice 15) so acc_manhflags' sandboxes keep CPU; /root/manh/venv (used by acc_manhflags) is not touched.
set -euo pipefail
M=/root/manh2; LOG=$M/build.log; mkdir -p $M/dist
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%T)] build start"
cd $M && rm -rf src && tar xzf src.tgz
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:/opt/instance-tools/bin:$PATH
command -v cargo > /dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1; }
cargo --version
[ -x $M/venv/bin/python ] || uv venv -q --python 3.12 $M/venv
PY=$M/venv/bin/python
cd $M/src
TV=$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' requirements/cuda.txt | head -1); echo "torch pin: $TV"
uv pip install -q --python $PY --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==$TV" -r requirements/build/cuda.txt -r requirements/build/rust.txt
$PY -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
export CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=32 NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda \
  CMAKE_BUILD_TYPE=Release SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev2+g95376a1
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -1
echo "[$(date -u +%T)] wheel build"
nice -n 15 uv build --wheel --no-build-isolation --python $PY -o $M/dist . 2>&1 | grep -v -E '^\s*$' | tail -40 || {
  echo "[$(date -u +%T)] BUILD FAILED"; exit 1; }
W=$(ls -t $M/dist/vllm-*.whl | head -1)
echo "[$(date -u +%T)] wheel $W $(du -h "$W" | cut -f1) sha256 $(sha256sum "$W" | cut -c1-64)"
echo "[$(date -u +%T)] BUILD OK"
