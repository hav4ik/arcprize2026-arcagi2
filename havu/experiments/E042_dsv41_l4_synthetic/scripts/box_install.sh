#!/bin/bash
# E042: rebuild the Vast box's /root/dsv41 on a fresh 4xL4 box with internet (Runpod replacement, 2026-10-23).
# Run on the box after setup_box.sh has unpacked the bundle into /root/dsv41/bundle. Same package set as the Vast venv
# (requirements.box.freeze, fork wheels sha256-checked), same patches as Kaggle (router whitelist, synthetic Engram
# and init, PR #154 env-gated) plus Humming, model metadata for E48/E64/E72, scripts and trace shapes.
set -euo pipefail
D=/root/dsv41; B=$D/bundle; LOG=$D/install.log
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%T)] install start"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
free -g | head -2; df -h / /workspace 2>/dev/null || df -h /; cat /sys/fs/cgroup/memory.max 2>/dev/null || true; nproc

command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH

# 1. Fork wheels from the publisher release, checked against the plan's lock
REL=https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130
mkdir -p $D/wheels $D/runs
while read -r sha name; do
  [ -f $D/wheels/$name ] || curl -fsSL -o $D/wheels/$name "$REL/${name//+/%2B}"
  echo "$sha  $D/wheels/$name" | sha256sum -c -
done <<'EOF'
e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl
078aa7682d699c67dce619fc8acf29ec41934ab5ba916268d1fe6f8850459de8 flashinfer_python-0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213-py3-none-any.whl
EOF

# 2. Everything else at the box's exact versions; --no-deps because the freeze is complete (and the fork's vllm names
# flashinfer by URL, which uv rejects during resolution)
grep -v ' @ ' $B/requirements.box.freeze > $D/wheels/pypi.txt
[ -x $D/venv/bin/python ] || uv venv --python 3.12 $D/venv
PY=$D/venv/bin/python; SP=$D/venv/lib/python3.12/site-packages
uv pip install --python $PY --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 -r $D/wheels/pypi.txt $D/wheels/*.whl

# 3. Patches: router whitelist, synthetic Engram rows and init (env-gated), PR #154 (env-gated, only onto the original
# file), Humming backends (opt-in)
$PY -I $B/apply_patches.py $SP/vllm
IDX=$SP/vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py
if [ "$(sha256sum < $IDX | cut -d' ' -f1)" = "$(cut -d' ' -f1 $B/dsv41_indexer.orig.sha256)" ]; then
  cp $B/dsv41_indexer.pr154.py $IDX; echo "PR #154 installed"
else
  grep -q DSV41_INDEXER_Q_PREDECODE $IDX && echo "PR #154 already installed" || { echo "PR #154: unexpected indexer file"; exit 1; }
fi
$PY -I $B/humming_v41.py $SP/vllm

# 4. CUDA 13 toolkit for FlashInfer's JIT: serve.sh expects /usr/local/cuda-13.0. Without one, the pip toolkit in the
# venv behind a shim, with libcuda from the driver as the link stub (as on Kaggle).
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  CU=$SP/nvidia/cu13; S=$D/cuda13; mkdir -p $S/lib64/stubs
  for d in bin include nvvm lib share; do [ -e $CU/$d ] && ln -sfn $CU/$d $S/$d; done
  for f in $CU/lib/*; do ln -sf $f $S/lib64/; done
  for n in cudart nvrtc; do r=$(ls $CU/lib/lib$n.so.* 2>/dev/null | head -1); [ -n "$r" ] && ln -sf $r $S/lib64/lib$n.so; done
  drv=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/local/nvidia/lib64/libcuda.so* 2>/dev/null | head -1)
  ln -sf $drv $S/lib64/stubs/libcuda.so; ln -sf $drv $S/lib64/libcuda.so
  [ -e /usr/local/cuda-13.0 ] || ln -s $S /usr/local/cuda-13.0
  echo "CUDA shim at $S (driver lib $drv)"
fi
mkdir -p ~/.cache/flashinfer && tar xzf $B/flashinfer_jit_cache_fork_sm89.tgz -C ~/.cache/flashinfer

# 5. Model metadata (no checkpoint shards: --load-format dummy + synthetic init), scripts, trace shapes
mkdir -p $D/model-e64 && tar xzf $B/model-e64-meta.tgz -C $D/model-e64
for e in 48 72; do
  rm -rf $D/model-e$e && cp -r $D/model-e64 $D/model-e$e
  cp $B/synthetic-e$e/config.json $B/synthetic-e$e/synthetic_manifest.json $D/model-e$e/
done
grep -o '"n_routed_experts": [0-9]*' $D/model-e*/config.json
mkdir -p $D/scripts $D/metadata && cp -r $B/scripts/. $D/scripts/ && cp -r $B/trace_shapes $D/metadata/
chmod +x $D/scripts/*.sh $D/scripts/*.py

# 6. Toolchain and imports
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc --version | tail -2
$PY -c "import importlib.metadata as m, torch, vllm, flashinfer; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), vllm.__version__, flashinfer.__version__, m.version(\"humming-kernels\"))"
echo "[$(date -u +%T)] INSTALL OK"
