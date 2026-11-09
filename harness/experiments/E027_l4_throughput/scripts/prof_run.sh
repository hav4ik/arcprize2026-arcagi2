#!/bin/bash
# Profiles steady-state decode of vLLM 0.23 FP8 TP4 (v4b server settings, prefix caching on) under 24 concurrent
# ~24.6k-token requests, for NCCL default (SHM) and NCCL_P2P_LEVEL=SYS. torch profiler, 20 s window per mode.
cd /root/arc
trap - INT
for p in $(pgrep -f 'chain_p2p.sh|probe.sh|arc_solver/solver.py'); do kill $p; done
while pgrep -f 'bench_decode.py' > /dev/null; do sleep 10; done
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server|bench_decode2.py'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
stop
M=Qwen/Qwen3.8-27B-FP8
for mode in shm p2p; do
  N=prof_$mode; rm -rf runs/$N; mkdir -p runs/$N/trace
  if [ $mode = p2p ]; then export NCCL_P2P_LEVEL=SYS; else unset NCCL_P2P_LEVEL; fi
  MODEL=$M VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --language-model-only \
    --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"/root/arc/runs/$N/trace\", \"torch_profiler_with_stack\": false, \"torch_profiler_record_shapes\": false}" \
    > runs/$N/server.log 2>&1 < /dev/null &
  until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback runs/$N/server.log && { echo "$N FAILED"; break; }; done
  /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 $N runs/$N/load.json 8000 3000 24 > runs/$N/load.log 2>&1 &
  # wait for steady decode: 24 running, no prefill in the last 10-s window
  for i in $(seq 1 120); do
    sleep 10
    tail -1 runs/$N/server.log | grep -q "Avg prompt throughput: 0.0 tokens/s.*Running: 24 reqs" && break
  done
  echo "$N steady at $(date -u +%H:%M:%S): $(grep 'Avg generation' runs/$N/server.log | tail -1 | sed -E 's/.*Avg prompt/prompt/' | cut -c1-150)"
  curl -s -X POST localhost:8001/start_profile > /dev/null; sleep 20; curl -s -X POST localhost:8001/stop_profile > /dev/null
  sleep 30
  echo "$N during profile: $(grep 'Avg generation' runs/$N/server.log | tail -3 | sed -E 's/.*Avg generation/gen/' | cut -c1-60 | tr '\n' '|')"
  ls -la runs/$N/trace | tail -5
  stop
done
echo PROF_DONE
