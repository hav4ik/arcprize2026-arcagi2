#!/bin/bash
# E064b (pod): Manh's Kaggle install (notebook cell 4, as embedded in B32-M) into a Python 3.13 venv instead of the
# Kaggle image's Python. The same uv commands, offline, with his 294 wheels as the only source (UV_FIND_LINKS). Skipped:
# the /kaggle/working source installs (unsloth-zoo, unsloth, peft, quack-kernels: training packages from his
# arc2-2026-src / unsloth-main datasets, not used by vLLM serving). The Kaggle image's own packages are not here, so a
# step that needs one of them is retried online (PyPI + his wheels), and the log says which.
set -uo pipefail
V=/root/b32m/venv; W=/root/b32m/dl/wheels/wheels; LOG=/root/b32m/install.log
UV=/usr/bin/uv; PYB=$(/root/kgl/bin/uv python find 3.13)
export UV_CACHE_DIR=/root/b32m/uvcache UV_FIND_LINKS=$W
: > $LOG
[ -x $V/bin/python ] || $UV venv -p "$PYB" $V >> $LOG 2>&1
step() {  # step <label> <uv pip install args...>
  local label=$1; shift
  echo "=== $label: offline" >> $LOG
  if UV_OFFLINE=1 $UV pip install --python $V/bin/python "$@" >> $LOG 2>&1; then echo "STEP_OK $label offline" >> $LOG; return; fi
  echo "=== $label: retry online" >> $LOG
  if $UV pip install --python $V/bin/python "$@" >> $LOG 2>&1; then echo "STEP_OK $label online" >> $LOG; else echo "STEP_FAIL $label" >> $LOG; fi
}
step vllm --no-build-isolation vllm torch torchvision -U
step torchaudio torchaudio torchcodec -U
step mamba-ssm mamba-ssm==2.3.2.post1 --no-deps
step flash_attn flash_attn causal_conv1d bitsandbytes xformers
step transformer_engine $W/transformer_engine-*
step apex apex redis
step flashinfer --no-build-isolation flashinfer-python flashinfer-cubin flashinfer-jit-cache
step transformers transformers==5.20.2
step nvidia-pins --no-build-isolation --no-deps nvidia-cudnn-cu13==9.24.0.43 nvidia-nvshmem-cu13==3.6.5 nvidia-nccl-cu13==2.30.7
step cutlass-dsl nvidia-cutlass-dsl~=4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1
# the tool sandbox (B32's arc_helpers / tool code): numpy + scipy, which the Kaggle image provides
step tool-sandbox numpy scipy
$V/bin/python -c "import torch, vllm, numpy, scipy; print('IMPORT_OK', torch.__version__, torch.version.cuda, vllm.__version__, numpy.__version__, scipy.__version__)" >> $LOG 2>&1
$UV pip freeze --python $V/bin/python > /root/b32m/freeze.txt 2>> $LOG
echo INSTALL_DONE >> $LOG
