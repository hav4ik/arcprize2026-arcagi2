#!/bin/bash
# L4 box fill (2026-10-17 18:35 UTC): 1) v4b probe with the tuned FP8 configs (vLLM 0.23), 2) FP8 TP2 x DP2 decode
# bench, 3) same-day repeat of the untuned v4b probe (configs moved aside, then restored).
cd /root/arc
trap - INT
unset NCCL_P2P_LEVEL
rm -rf /root/.cache/vllm/torch_compile_cache /root/.cache/vllm/torch_aot_compile
df -h / | tail -1
M=Qwen/Qwen3.8-27B-FP8
CFG=$(/root/vllm-venv/bin/python -c "import vllm, os; print(os.path.dirname(vllm.__file__) + '/model_executor/layers/quantization/utils/configs')" 2>/dev/null)
echo "tuned configs installed: $(ls $CFG | grep -c 'device_name=NVIDIA_L4,dtype=fp8_w8a8')"
./probe.sh p_fp8_tuned_c24 $M 24
echo "tuned probe config warnings: $(grep -c 'Config file not found' runs/p_fp8_tuned_c24/server.log)"
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
up() { until curl -s -m 5 localhost:$1/v1/models > /dev/null; do sleep 15; if grep -qE "Traceback" $2; then echo "FAILED $2"; return 1; fi; done; return 0; }
N=fp8_tp2x2; mkdir -p runs/$N
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((2 * s)),$((2 * s + 1)) MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 \
    --no-enable-prefix-caching --language-model-only --tensor-parallel-size 2 --disable-custom-all-reduce --port $((8001 + s)) \
    > runs/$N/server$s.log 2>&1 < /dev/null &
  sleep 60
done
up 8001 runs/$N/server0.log; a=$?; up 8002 runs/$N/server1.log; b=$?
echo "$N up=$a,$b $(grep -ho 'GPU KV cache size: [0-9,]* tokens' runs/$N/server*.log | tr '\n' ' ')"
if [ $a = 0 ] && [ $b = 0 ]; then
  /opt/sglang/bin/python bench_dp.py http://127.0.0.1:8001,http://127.0.0.1:8002 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
  cat runs/$N/bench.log
fi
stop
mkdir -p /root/arc/fp8_configs_aside && mv $CFG/*device_name=NVIDIA_L4,dtype=fp8_w8a8* /root/arc/fp8_configs_aside/
./probe.sh p_fp8_untuned_c24_rep $M 24
echo "untuned probe config warnings: $(grep -c 'Config file not found' runs/p_fp8_untuned_c24_rep/server.log)"
mv /root/arc/fp8_configs_aside/* $CFG/ && echo "tuned configs restored: $(ls $CFG | grep -c 'device_name=NVIDIA_L4,dtype=fp8_w8a8')"
echo FILL1_DONE
