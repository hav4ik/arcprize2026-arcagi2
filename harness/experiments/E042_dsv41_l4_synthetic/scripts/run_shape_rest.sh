#!/bin/bash
# E042 SH (Runpod 4xL4): the user's target shape, "30% of the original engrams (keep it at original precision), and
# having more experts at 2-3 bits". Engram 30% = 230.4M of 768.0M rows (FP8 rows, 264 B) = 56.6 GiB pinned: 28.8M
# synthetic rows per rank and layer. Mixed bits through Humming's per-layer "dynamic" overrides (3 bits on layers
# 0-19, 2 bits on 20-39). Then the rest of LB (E96 uint3, E160 uint2, E64 controls). Log: /root/dsv41/runs/lb.log
# Based on run_lowbit.sh (E042 LB):: more experts at fewer bits through Humming's batched MoE (patches/humming_experts.py,
# DSV41_EXPERT_HUMMING), random weights, so speed and memory only. User: "a virtual random pruning down to 128 experts
# and see the throughput". Per-GPU weights at MXFP4 (4.25 bits) cost ≈ 0.18 GiB per expert; GPTQ-style uint2 / uint3
# with 128-wide groups cost ≈ 2.1 / 3.1 bits, so E128 at 2 bits ≈ E64's memory and E96 at 3 bits ≈ E72's.
# Server: T72's no-tool config (160K, prefix caching, retention 128, PR #154 flag, 128 seqs, util 0.88). Points: decode
# 8K x 32 and 8K x 64, then the E043 no-tool snapshot at the largest C in 64..24 that fits 85% of KV.
# Control: E64 on Marlin MXFP4 and E64 on Humming uint4 at the same config (is Humming's MoE as fast as Marlin?).
# Log: /root/dsv41/runs/lb.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/lb.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
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
    "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | head -1);" \
    "$(grep -o -E 'Available KV cache memory: [0-9.]+ GiB' $R/server.log | head -1);" \
    "$(grep -o -E "Using [A-Za-z0-9_' ]+ (MoE|Mxfp4)[A-Za-z0-9_ ]*" $R/server.log | sort -u | head -2 | tr '\n' ' ')"
}
# lb <name> <model> <humming schema JSON or "" for Marlin> <synthetic Engram rows per rank and layer> <snapshot Cs>
lb() {
  local N=$1 m=$2 sch=$3 rows=$4 cs=$5; local R=/root/dsv41/runs/$N
  for i in $(seq 60); do busy || break; sleep 5; done
  busy && { log "ABORT before $N: busy"; exit 1; }
  local H=(); [ -n "$sch" ] && H=("DSV41_EXPERT_HUMMING=$sch"); [ -n "$W2" ] && H+=("DSV41_EXPERT_HUMMING_W2=$W2")
  env "${H[@]}" ENGRAM_ROWS=$rows DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-$m SEQS=${SQ:-128} UTIL=${UT:-0.88} MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 "${XA[@]}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    local kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1 | tr -dc 0-9) c need
    mkdir -p $R/dw
    for p in 8192:32 8192:64; do
      ctx=${p%%:*}; c=${p##*:}
      # 160K units: ≈ 1.63 reported tokens per 64K-unit token (S4 fit); skip what would not fit
      need=$(( c * (ctx + 1500) * 163 / 100 ))
      [ $need -gt $((kv * 85 / 100)) ] && { log "$N ctx $ctx C $c: skipped, need $need of KV $kv"; continue; }
      log "$N ctx $ctx C $c: $(timeout 1800 $PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c \
        --sock-read 10800 --json $R/dw/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
    done
    [ -z "$cs" ] && { log "$N health: $($PY $S/run_health.py $R 2>&1 | cut -c1-500)"; $S/stop.sh $N | tee -a $L; return; }
    for c in $cs; do
      $PY $S/trace_shapes.py --snapshot notools --shapes $TS/notools_e043_e72.json --conc $c --seed 1 --out $R/snap.json > /dev/null
      need=$($PY -c "import json,sys; print(int(sum((1600 + 0.355 * x) * 2069257 / 1267921 for x in json.load(open(sys.argv[1])))))" $R/snap.json)
      [ $need -le $((kv * ${GD:-85} / 100)) ] && break
    done
    if [ $need -gt $((kv * ${GD:-85} / 100)) ]; then log "$N: no snapshot fits KV $kv"
    else log "$N no-tools snapshot C $c (need $need of KV $kv): $(timeout 2700 $PY $S/decode_window.py --ctx-file $R/snap.json \
      --window 120 --seed $c --sock-read 14400 --max-model-len 163840 --json $R/notools_c$c.json 2>&1 | tail -1 \
      | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-600)"
    fi
    log "$N host: $(free -g | awk '/Mem:/{print $3 " GiB used / " $7 " available"}'); $(grep -o -E 'Engram table offloaded[^;]*' $R/server.log | head -1 | cut -c1-160)"
    log "$N health: $($PY $S/run_health.py $R 2>&1 | cut -c1-500)"
  fi
  $S/stop.sh $N | tee -a $L
}
g() { echo "{\"quant_method\":\"gptq\",\"bits\":$1,\"group_size\":128,\"sym\":true,\"desc_act\":false$2}"; }
MIX=', "dynamic": {"+:.*layers[.](1?[0-9])[.]ffn.*": {"bits": 3}}'
E30=28800000; E0=4000000
# 06:00 replan: E72 at util 0.93 ran 31 min without OOM and doubled KV (1.59M vs 821K tokens), and every no-tool
# point so far was KV-bound (ITL nearly flat in C), so E043's shapes and E96 uint3 get util 0.93 and larger snapshots.
# Guard 80%: KV peaks have run ~1.1x the snapshot estimate (ds_x128 0.83 -> 0.92). E160 uint2 dropped: ~19.5 GiB of
# weights per GPU does not fit. The E64 x 64 tool anchor was dropped too (no-tool is the focus).
spec() { echo "{\"method\":\"dspark\",\"num_speculative_tokens\":$1,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$2}"; }
H() { echo "{\"quant_method\":\"humming\",\"weight_dtype\":\"$1\",\"weight_scale_dtype\":\"bfloat16\",\"group_size\":$2,\"has_zero_point\":true}"; }
for i in $(seq 90); do busy || break; sleep 2; done
XA=(); UT=0.93; GD=80; W2=
log "sh rest start (util 0.93 replan)"
W2='{"group_size": 64}'; lb ds_x128_u93 e128 "$(H uint2 128)" $E30 "60 56 52 48 44 40 36 32"
W2='{"weight_dtype": "uint3", "group_size": 64}'; SQ=64; XA=(--speculative-config "$(spec 4 2.49)")
lb ds_x96_n4_u93 e96 "$(H uint2 128)" $E30 "60 56 52 48 44 40 36 32"
W2=; SQ=; XA=()
lb lb_e96_g3_u93 e96 "$(g 3)" $E30 "60 56 52 48 44 40 36 32"
UT=0.88; GD=85
lb lb_e64_g4 e64 "$(g 4)" $E0 ""
lb lb_e64_marlin e64 "" $E0 ""
log "SH_REST_DONE"
