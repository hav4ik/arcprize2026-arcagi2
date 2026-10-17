#!/bin/bash
# 1x RTX PRO 6000 eval box: stop the template SGLang, install vLLM 0.31 (cu130), download Qwen3.8-27B-FP8,
# unpack the v4b solver bundle into /root/arc. Logs to /root/arc/logs/setup.log.
mkdir -p /root/arc/logs /workspace/models
cd /root/arc
supervisorctl stop sglang > /dev/null 2>&1
for p in $(pgrep -f 'sglang serve|sglang.launch_server'); do kill $p; done
tar xzf /root/arc_bundle.tgz -C /root/arc
rm -rf /workspace/models/models--nvidia--Qwen3.8-27B-NVFP4 /workspace/models/*NVFP4* 2>/dev/null
( /opt/sglang/bin/hf download Qwen/Qwen3.8-27B-FP8 --cache-dir /workspace/models > logs/dl_fp8.log 2>&1 && echo "fp8 download ok" ) &
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8
uv venv --python 3.12 /root/vllmnew > /dev/null 2>&1
. /root/vllmnew/bin/activate
for i in 1 2 3; do
  uv pip install vllm==0.31.0 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > logs/install.log 2>&1
  python -c "import vllm" 2>/dev/null && break
  sleep 20
done
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tail -1
rm -rf /root/.cache/uv
wait
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h / | tail -1
echo SETUP_DONE
