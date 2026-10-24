#!/bin/bash
# E042 DS1 follow-ups: E043's low-bit candidates with and without DSpark (dummy weights). Helpers from run_dspark.sh:: cost side of DeepSeek-V4.1's own DSpark drafter (3 blocks, 128 experts top-3, 5-token
# blocks, weights in the target checkpoint; the fork's DSparkV41DraftModel). Random weights accept nothing, so the
# fork's synthetic rejection sampler accepts drafts at a set mean length L; step cost does not depend on L, so
# tok/s at another L' scales by L'/L. Decode benches on the P12 server (E64, 128 seqs, util 0.88, MBT 2048, 64K, no
# prefix caching, PR #154 flag): no drafter vs n = 1 / 2 / 4 drafts, then whether E72 + drafter fits at all.
# Usage: run_dspark.sh <t72 runner pid> <t72 tool_replay pid>: lets T72's E72 x 32 replay finish, stops that runner
# so it does not start E72 x 40, then resumes T72's remaining points afterwards. Log: /root/dsv41/runs/ds.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/ds.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-400)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
    "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | tr '\n' ' ')" \
    "$(grep -o -E 'Available KV cache memory: [0-9.]+ GiB' $R/server.log | head -1)"
}
# ds <name> <model> <points> [vllm args]: points are ctx:conc pairs
ds() {
  local N=$1 m=$2 pts=$3; shift 3; local R=/root/dsv41/runs/$N/dw
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-$m SEQS=128 UTIL=0.88 $S/serve.sh $N "$@" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    mkdir -p $R
    local kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' /root/dsv41/runs/$N/server.log | tail -1 | tr -dc 0-9)
    for p in $pts; do
      ctx=${p%%:*}; c=${p##*:}
      # decode_window waits for every stream's first token: a point that does not fit would hang, so skip it
      [ "${kv:-0}" -lt $((c * (ctx + 1500))) ] && { log "$N ctx $ctx C $c: skipped, KV ${kv:-0} < $((c * (ctx + 1500)))"; continue; }
      log "$N ctx $ctx C $c: $(timeout 1800 $PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
        --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
    done
    grep -o -E "(Mean acceptance length|Avg Draft acceptance rate|Accepted throughput)[^,]*" /root/dsv41/runs/$N/server.log | tail -2 | tr '\n' ' ' | tee -a $L; echo | tee -a $L
  fi
  $S/stop.sh $N | tee -a $L
}
spec() { echo "{\"method\":\"dspark\",\"num_speculative_tokens\":$1,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$2}"; }

dsnt() {
  local N=$1 m=$2; shift 2; local R=/root/dsv41/runs/$N
  for i in $(seq 60); do busy || break; sleep 5; done
  busy && { log "ABORT before $N: busy"; exit 1; }
  # 48 seqs, not 128: snapshots go up to 40 streams, and spec decode's CUDA graphs at 128 x (n + 1) tokens cost KV
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-$m SEQS=48 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 "$@" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    local kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1 | tr -dc 0-9) c need
    for c in 40 36 32 28 24 20 16 12 8 4; do
      $PY $S/trace_shapes.py --snapshot notools --shapes $TS/notools_e043_e72.json --conc $c --seed 1 --out $R/snap.json > /dev/null
      need=$($PY -c "import json,sys; print(int(sum((1600 + 0.355 * x) * 2069257 / 1267921 for x in json.load(open(sys.argv[1])))))" $R/snap.json)
      [ $need -le $((kv * 85 / 100)) ] && break   # 85%: streams grow during the long prefill (E72 x 32 at 92% preempted)
    done
    if [ $need -gt $((kv * 85 / 100)) ]; then log "$N: no snapshot fits KV $kv"
    else log "$N no-tools snapshot C $c (need $need of KV $kv): $(timeout 2400 $PY $S/decode_window.py --ctx-file $R/snap.json \
      --window 120 --seed $c --sock-read 14400 --max-model-len 163840 --json $R/notools_c$c.json 2>&1 | tail -1 \
      | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-600)"
    fi
  fi
  log "$N host: $(free -g | awk '/Mem:/{print $3 " GiB used / " $7 " available"}'); $(grep -o -E 'Engram table offloaded[^;]*' $R/server.log | head -1 | cut -c1-120)"
  [ -f $R/gpu.csv ] && log "$N health: $($PY $S/run_health.py $R 2>&1 | cut -c1-500)"
  $S/stop.sh $N | tee -a $L
}
TS=/root/dsv41/metadata/trace_shapes
# E043's real-weight candidates (2026-10-24 04:45), here with dummy weights: speed, memory and whether the SM89 kernels
# run. Native Humming schema with E043's per-sublayer override (apply_e043_humming.py replaced humming_experts.py's
# hook on this pod; same DSV41_EXPERT_HUMMING env, plus DSV41_EXPERT_HUMMING_W2 merged in for the down projection).
# w13 uint2 g128 (K = 5120, not split by TP) and w2 g64 (TP4 K = 576 = 9 x 64), asymmetric, bf16 scales, Engram 30%.
# The hook covers every RoutedExperts of the quant config, so the drafter's 128 experts get the same schema.
H() { echo "{\"quant_method\":\"humming\",\"weight_dtype\":\"$1\",\"weight_scale_dtype\":\"bfloat16\",\"group_size\":$2,\"has_zero_point\":true}"; }
export ENGRAM_ROWS=28800000
export DSV41_EXPERT_HUMMING="$(H uint2 128)" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
dsnt ds_x128 e128
export DSV41_EXPERT_HUMMING_W2='{"weight_dtype": "uint3", "group_size": 64}'
dsnt ds_x96 e96
dsnt ds_x96_n2 e96 --speculative-config "$(spec 2 1.99)"
dsnt ds_x96_n4 e96 --speculative-config "$(spec 4 2.49)"
unset DSV41_EXPERT_HUMMING DSV41_EXPERT_HUMMING_W2 ENGRAM_ROWS
log "DS16_DONE"
/root/dsv41/scripts/run_shape_rest.sh
