#!/bin/bash
# Newest vLLM (/root/vllmnew), FP8, TP4, NCCL P2P, prefix caching on:
#   a: no MTP, fp8 KV   b: MTP2, fp8 KV   c: MTP2 + turboquant_k8v4   d: no MTP + turboquant_k8v4
# Each: greedy multi-turn consistency check (mtp_check.py, 2 passes); b and c also a decode bench (c16/24/32).
cd /root/arc
trap - INT
until grep -q PROF_DONE prof_run.log && grep -q INSTALL_DONE install_vllm_new.log; do sleep 30; done
export NCCL_P2P_LEVEL=SYS
M=Qwen/Qwen3.8-27B-FP8
stop() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
         for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
up() { until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; if grep -q Traceback $1; then echo "FAILED $1: $(grep -E 'Error' $1 | grep -v Warning | tail -2 | cut -c1-300)"; return 1; fi; done; return 0; }
run() {  # label mtp attn kv bench(0/1)
  local N=mtpchk_$1; mkdir -p runs/$N
  MODEL=$M VENV=/root/vllmnew ATTN=$3 setsid nohup ./start_vllm.sh $2 --language-model-only --kv-cache-dtype $4 \
    > runs/$N/server.log 2>&1 < /dev/null &
  if up runs/$N/server.log; then
    echo "$N $(grep -o 'GPU KV cache size: [0-9,]* tokens' runs/$N/server.log | tail -1)"
    /opt/sglang/bin/python mtp_check.py run http://127.0.0.1:8001 runs/$N/check.json 16 400 > runs/$N/check.log 2>&1
    tail -1 runs/$N/check.log
    if [ "$5" = 1 ]; then
      /opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/bench.json 16 24 32 > runs/$N/bench.log 2>&1
      cat runs/$N/bench.log
      grep -E "SpecDecoding metrics" runs/$N/server.log | tail -1 | cut -c1-220
    fi
  fi
  stop
}
stop
run a 0 TRITON_ATTN fp8 1
run b 2 TRITON_ATTN fp8 1
run c 2 TURBOQUANT turboquant_k8v4 1
run d 0 TURBOQUANT turboquant_k8v4 0
C="/opt/sglang/bin/python mtp_check.py compare"
for x in a b c d; do [ -f runs/mtpchk_$x/check.json ] && $C runs/mtpchk_$x/check.json runs/mtpchk_$x/check.json > /dev/null; done
[ -f runs/mtpchk_b/check.json ] && $C runs/mtpchk_a/check.json runs/mtpchk_b/check.json
[ -f runs/mtpchk_c/check.json ] && $C runs/mtpchk_d/check.json runs/mtpchk_c/check.json
[ -f runs/mtpchk_d/check.json ] && $C runs/mtpchk_a/check.json runs/mtpchk_d/check.json
/opt/sglang/bin/python - <<'EOF'
# pass-1 vs pass-2 within each server (cold vs cached prefixes)
import json, os
def div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))
for x in "abcd":
    f = f"runs/mtpchk_{x}/check.json"
    if not os.path.exists(f): continue
    R = json.load(open(f)); ids = sorted({k.split(":", 1)[1] for k in R})
    for turn in (0, 1):
        ds = [div(R["p1:" + t][turn], R["p2:" + t][turn]) for t in ids]
        print(f"{x} cold vs cached, turn{turn + 1}: identical {sum(d is None for d in ds)}/{len(ds)}, first divergence {sorted(d for d in ds if d is not None)[:8]}")
EOF
echo MTPCHK_DONE
