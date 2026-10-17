#!/bin/bash
# usage: fix_install.sh <eval label> <mtp> <kv dtype> <attn>
# Stops the stuck eval, retries the vLLM 0.31 install with long HTTP timeouts, then restarts the eval.
cd /root/arc
for p in $(pgrep -f 'eval_6000.sh'); do kill $p; done
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8
. /root/vllmnew/bin/activate
for i in 1 2 3; do
  uv pip install vllm==0.31.0 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    > logs/install_retry$i.log 2>&1
  python -c "import vllm" 2>/dev/null && break
  sleep 20
done
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" 2>&1 | tail -1
rm -rf /root/.cache/uv
(setsid nohup ./eval_6000.sh "$@" > eval_$1.log 2>&1 < /dev/null &)
echo "restarted $1"
