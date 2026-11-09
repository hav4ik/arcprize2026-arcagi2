#!/bin/bash
# L4 box fill 2 (queued after chain_fill1): diagnostics for the vLLM 0.31 slowdown and MTP + TurboQuant.
#  A) cache_probe.py (does turn 2 reuse turn 1's generated tokens?) on vLLM 0.23, vLLM 0.31, vLLM 0.31 + MTP2.
#  B) MTP2 + turboquant_k8v4 degenerate-output check (mtp_check.py) with CUDA graphs PIECEWISE only, then eager
#     (upstream #53180: full-graph capture of spec-decode batches in the TurboQuant backend is the suspected cause).
cd /root/arc
trap - INT
until grep -q FILL1_DONE chain_fill1.log 2>/dev/null; do sleep 60; done
unset NCCL_P2P_LEVEL
M=Qwen/Qwen3.8-27B-FP8
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
up() { until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; if grep -qE "Traceback" $1; then echo "FAILED $1: $(grep -E 'Error' $1 | grep -v Warning | tail -1 | cut -c1-250)"; return 1; fi; done; return 0; }
srv() {  # label venv mtp attn extra...
  local N=$1 V=$2 T=$3 A=$4; shift 4; mkdir -p runs/$N
  MODEL=$M VENV=$V ATTN=$A setsid nohup ./start_vllm.sh $T --language-model-only "$@" > runs/$N/server.log 2>&1 < /dev/null &
  up runs/$N/server.log
}
for spec in "cp_v023 /root/vllm-venv 0" "cp_v031 /root/vllmnew 0" "cp_v031_mtp2 /root/vllmnew 2"; do
  set -- $spec
  if srv $1 $2 $3 TRITON_ATTN; then
    /opt/sglang/bin/python cache_probe.py http://127.0.0.1:8001 $1 runs/$1/cache.json 4 1500 > runs/$1/cache.log 2>&1
    cat runs/$1/cache.log | cut -c1-300
  fi
  stop
done
for mode in piecewise eager; do
  N=mtptq_$mode
  if [ $mode = piecewise ]; then X=(-cc '{"cudagraph_mode": "PIECEWISE"}'); else X=(--enforce-eager); fi
  if srv $N /root/vllmnew 2 TURBOQUANT --kv-cache-dtype turboquant_k8v4 "${X[@]}"; then
    /opt/sglang/bin/python mtp_check.py run http://127.0.0.1:8001 runs/$N/check.json 16 400 > runs/$N/check.log 2>&1
    /opt/sglang/bin/python - <<EOF
import json, re
R = json.load(open("runs/$N/check.json"))
bad = sum(1 for t1, t2 in R.values() for s in (t1, t2) if re.search(r'(" ){3,}|\n{5,}|The user says', s))
print("$N degenerate outputs:", bad, "of", 2 * len(R))
EOF
    grep -E "SpecDecoding metrics" runs/$N/server.log | tail -1 | sed -E "s/.*Mean acceptance length: ([0-9.]+).*/$N acceptance \1/"
  fi
  stop
done
echo FILL2_DONE
