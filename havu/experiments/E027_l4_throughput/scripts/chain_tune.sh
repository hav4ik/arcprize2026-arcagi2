#!/bin/bash
# After the vLLM 0.31 MTP2 probe on the L4 box: tune vLLM 0.23's Triton W8A8 block-FP8 GEMM configs for NVIDIA L4 at
# Qwen3.8-27B TP4 shapes, install them, re-run the decode + prefill benches, then the vLLM 0.31 baseline probe.
cd /root/arc
trap - INT
for p in $(pgrep -f 'chain_l4_probe2_now.sh'); do kill $p; done   # stop the chain before it starts the baseline probe
until grep -qE "p31_mtp2_c20 (RESULT|SERVER_FAILED)" chain_l4_probe.log 2>/dev/null; do sleep 60; done
sleep 60
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5
B=/root/src/vllm_bench
python3 - <<'EOF'
import re
src = open("/root/src/vllm_bench/benchmark_w8a8_block_fp8.py").read()
shapes = "[(3584, 5120), (4096, 5120), (5120, 1536), (5120, 4352), (8704, 5120)]"
src = re.sub(r"def get_weight_shapes\(tp_size\):.*?return weight_shapes\n",
             f"def get_weight_shapes(tp_size):\n    # Qwen3.8-27B FP8, TP4 (shapes from vLLM's missing-config warnings on L4)\n    return {shapes}\n",
             src, count=1, flags=re.S)
src = src.replace("            3072,\n            4096,\n", "").replace("            1536,\n", "")
open("/root/src/vllm_bench/tune_qwen38.py", "w").write(src)
EOF
grep -A2 "def get_weight_shapes" $B/tune_qwen38.py
mkdir -p /root/arc/fp8_configs
t0=$(date +%s)
timeout 10800 /root/vllm-venv/bin/python $B/tune_qwen38.py --tp-size 4 --out-dtype bfloat16 --save-path /root/arc/fp8_configs/ \
  > /root/arc/logs/tune_fp8.log 2>&1
echo "tuning rc=$? in $(( $(date +%s) - t0 ))s: $(ls /root/arc/fp8_configs | wc -l) config files"
ls /root/arc/fp8_configs
CFG=$(/root/vllm-venv/bin/python -c "import vllm, os; print(os.path.dirname(vllm.__file__) + '/model_executor/layers/quantization/utils/configs')")
cp /root/arc/fp8_configs/*.json $CFG/ && echo "installed into $CFG"
N=fp8_tuned; mkdir -p runs/$N
MODEL=Qwen/Qwen3.8-27B-FP8 VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
  > runs/$N/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback runs/$N/server.log && { echo "$N FAILED"; break; }; done
echo "$N config-missing warnings: $(grep -c 'Config file not found' runs/$N/server.log)"
/opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
echo "prefill: $(/opt/sglang/bin/python -c "
import json; r = json.load(open('runs/$N/prefill.json'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']), 'tok/s')")"
/opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
cat runs/$N/bench.log
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
echo TUNE_DONE
VENV=/root/vllmnew ATTN=TRITON_ATTN MTP=0 ./probe2.sh p31_base_c24 Qwen/Qwen3.8-27B-FP8 24 --language-model-only
echo BASE_PROBE_DONE
