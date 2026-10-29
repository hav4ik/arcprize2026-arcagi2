#!/bin/bash
# E062: build Manh's vLLM fork (github.com/nguyen599/vllm) from source for SM89 only, in its own uv venv, and keep the
# wheel for the L4 box. Runs on the box, detached; log /root/manh/build.log.
#   setsid nohup bash /root/manh/build_manh.sh [sha] > /dev/null 2>&1 < /dev/null &
set -euo pipefail
SHA=${1:-b9ad670a6713aa3de4b72fe665a89e17b2c9752c}
M=/root/manh; LOG=$M/build.log
mkdir -p $M/dist
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%T)] build start, sha $SHA"
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
# the Rust frontend (setuptools-rust) needs cargo
command -v cargo > /dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1; }
cargo --version
# the repo is private: the source (a clone with its .git) is copied from the workspace, not cloned here
[ -d $M/src/.git ] || { echo "no source at $M/src (copy it first)"; exit 1; }
cd $M/src && [ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "source is at $(git rev-parse HEAD), want $SHA"; exit 1; }
git log -1 --format='%H %cd %s'
[ -x $M/venv/bin/python ] || uv venv -q --python 3.12 $M/venv
PY=$M/venv/bin/python
# torch first (the version requirements/cuda.txt pins), then the build requirements
TV=$(sed -n 's/^torch==\([0-9.]*\).*/\1/p' requirements/cuda.txt | head -1)
echo "torch pin: $TV"
uv pip install -q --python $PY --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==$TV" -r requirements/build/cuda.txt -r requirements/build/rust.txt
$PY -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=${MAX_JOBS:-64} NVCC_THREADS=2 \
  VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -2
echo "[$(date -u +%T)] wheel build"
uv build --wheel --no-build-isolation --python $PY -o $M/dist . 2>&1 | grep -v -E '^\s*$' | tail -40 || {
  echo "[$(date -u +%T)] BUILD FAILED"; exit 1; }
W=$(ls -t $M/dist/vllm-*.whl | head -1)
echo "[$(date -u +%T)] wheel $W $(du -h "$W" | cut -f1) sha256 $(sha256sum "$W" | cut -c1-64)"
# runtime deps (cuda.txt pins torch/flashinfer/...), then the wheel itself
uv pip install -q --python $PY --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r requirements/cuda.txt "$W"
$PY -c "import importlib.metadata as m, torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, torch.version.cuda, '| triton', m.version('triton'), '| flashinfer', m.version('flashinfer-python'))"
uv pip freeze --python $PY > $M/dist/freeze.txt
echo "[$(date -u +%T)] BUILD OK"
