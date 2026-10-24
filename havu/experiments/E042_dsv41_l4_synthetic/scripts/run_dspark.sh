#!/bin/bash
# E042 DS1 (Runpod 4xL4): cost side of DeepSeek-V4.1's own DSpark drafter (3 blocks, 128 experts top-3, 5-token
# blocks, weights in the target checkpoint; the fork's DSparkV41DraftModel). Random weights accept nothing, so the
# fork's synthetic rejection sampler accepts drafts at a set mean length L; step cost does not depend on L, so
# tok/s at another L' scales by L'/L. Decode benches on the P12 server (E64, 128 seqs, util 0.88, MBT 2048, 64K, no
# prefix caching, PR #154 flag): no drafter vs n = 1 / 2 / 4 drafts, then whether E72 + drafter fits at all.
# Usage: run_dspark.sh <t72 runner pid> <t72 tool_replay pid>: lets T72's E72 x 32 replay finish, stops that runner
# so it does not start E72 x 40, then resumes T72's remaining points afterwards. Log: /root/dsv41/runs/ds.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/ds.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
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

# 0. Take the box from T72 after its E72 x 32 replay (only with the two pids; run_seq.sh calls it without)
[ -n "${1:-}" ] && kill $1 2>/dev/null && log "stopped the T72 runner (pid $1); waiting for its E72 x 32 replay (pid $2)"
if [ -n "${2:-}" ]; then
  while kill -0 $2 2>/dev/null; do sleep 20; done
  J=/root/dsv41/runs/t72_e72_c32/r2_tools_c32.json
  echo "[$(date -u +%H:%M:%S)] t72_e72_c32 tools C 32 (logged by run_dspark.sh): $($PY -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('turns',None); print(json.dumps(d))" $J 2>&1 | cut -c1-900)" >> /root/dsv41/runs/t72.log
  $S/stop.sh t72_e72_c32 | tee -a $L
fi
log "ds start"

# 1. E64: no drafter, then n = 1 / 2 / 4 drafts at E043's real full-model no-tool acceptance, L = 1.69 / 2.19 / 2.83
P="8192:32 8192:64 32768:16"
ds ds_e64_base e64 "$P"
ds ds_e64_n1 e64 "$P" --speculative-config "$(spec 1 1.69)"
ds ds_e64_n2 e64 "$P" --speculative-config "$(spec 2 2.19)"
ds ds_e64_n4 e64 "$P" --speculative-config "$(spec 4 2.83)"
# 2. E72 + drafter, no-tool snapshot (the user: "let's try it anyways"). The full drafter is 1.85 GiB per GPU (E043:
# mtp.* 7.38 GiB) against E72's 1.18 GiB of KV, so also drafters cut to 32 / 16 experts (model-e72-d32/-d16:
# text_config.dspark_n_routed_experts; router shapes (32, 3), (16, 3) added by apply_patches.py). The no-tool server
# of T72's nt points (160K, prefix caching), at the largest C whose snapshot fits; compare with t72nt_e72_c24/28/32.
dsnt() {
  local N=$1 m=$2; shift 2; local R=/root/dsv41/runs/$N
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-$m SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 "$@" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    local kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1 | tr -dc 0-9) c need
    for c in 28 24 20 16 12 8 4; do
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
  $S/stop.sh $N | tee -a $L
}
TS=/root/dsv41/metadata/trace_shapes
# L: E043's real E72 no-tool acceptance with the full 128-expert drafter (1.99 at n = 2, 2.49 at n = 4); a cut drafter
# accepts less by an unmeasured amount, so the d32 / d16 points are upper bounds
dsnt ds_e72_d128_n2 e72 --speculative-config "$(spec 2 1.99)"
dsnt ds_e72_d32_n2 e72-d32 --speculative-config "$(spec 2 1.99)"
dsnt ds_e72_d16_n2 e72-d16 --speculative-config "$(spec 2 1.99)"
dsnt ds_e72_d16_n4 e72-d16 --speculative-config "$(spec 4 2.49)"
log "DS_DONE"

# 3. T72's remaining points (when it took the box from T72)
[ -n "${1:-}" ] && POINTS="e72:40 e72:24 e64:64" $S/run_e72t.sh
