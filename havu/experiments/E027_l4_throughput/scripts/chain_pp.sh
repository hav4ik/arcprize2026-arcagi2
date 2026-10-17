#!/bin/bash
# Pipeline parallelism on vLLM 0.23 (Kaggle stack), FP8, NCCL default transport (SHM), prefix caching off:
#   pp4: TP1 x PP4    pp2tp2: TP2 x PP2 (custom all-reduce off)
# decode bench (8/16/32/48) + prefill bench (16 x ~12.3k). Baseline TP4: fp8_marlin (131/171/256/302, prefill 916).
# Then the vLLM 0.31 MTP probes (chain_l4_probe2.sh without its wait).
cd /root/arc
trap - INT
for p in $(pgrep -f 'chain_l4_probe2.sh|probe2.sh|arc_solver/solver.py'); do kill $p; done
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
up() { until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15
         if grep -qE "Traceback|No module named" $1 || ! pgrep -f 'vllm.entrypoints.openai.api_server' > /dev/null; then
           echo "FAILED $1: $(grep -E 'Error' $1 | grep -v Warning | tail -2 | cut -c1-300)"; return 1; fi; done; return 0; }
stop
unset NCCL_P2P_LEVEL
M=Qwen/Qwen3.8-27B-FP8
run() {  # label extra flags
  local N=$1; shift; mkdir -p runs/$N
  MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only "$@" \
    > runs/$N/server.log 2>&1 < /dev/null &
  if up runs/$N/server.log; then
    echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | sort -u | tr '\n' ' ')"
    /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
    echo "prefill: $(/opt/sglang/bin/python -c "
import json; r = json.load(open('runs/$N/prefill.json'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']), 'tok/s', r['wall_s'], 's, errors', r.get('errors'))")"
    /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
    cat runs/$N/bench.log
  fi
  stop
}
run pp4 --tensor-parallel-size 1 --pipeline-parallel-size 4
run pp2tp2 --tensor-parallel-size 2 --pipeline-parallel-size 2 --disable-custom-all-reduce
echo PP_DONE
sed -e '/^until grep -q PROF_DONE/d' chain_l4_probe2.sh > chain_l4_probe2_now.sh && chmod +x chain_l4_probe2_now.sh
./chain_l4_probe2_now.sh > chain_l4_probe.log 2>&1
