#!/bin/bash
# E042 T72 (Runpod 4xL4, 2026-10-24): synthetic E72 tool-mode throughput. E72's weights (16.56 GiB per GPU) leave
# far less KV than E64's, so the question is how many concurrent attempts stay cache-healthy and what that yields.
# The Kaggle tool config (128 seqs, util 0.88, 160K, prefix caching, retention 128, PR #154 flag on, tool_replay v4,
# seed 1) at C 32, 40, 24, then E64 x 64 as the anchor against Kaggle run 8/9 (349 tok/s). Default NCCL (SHM), as on
# Kaggle. Log: /root/dsv41/runs/t72.log
# Points: <model>:<C> = tool replay; nt:<model>:<C> = no-tool steady-state snapshot (E043's real E72 no-tool shapes,
# notools_e043_e72.json: prompt mean 9.4K, output mean 31K) through decode_window, as R1. /root/dsv41/runs/t72.points,
# if present, overrides POINTS (E043 put E72's 90% quality in the no-tool harness, so the resumed run measures that).
S=/root/dsv41/scripts; L=/root/dsv41/runs/t72.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
    "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | head -1)"
}
for i in $(seq 90); do busy || break; sleep 2; done
[ -s /root/dsv41/runs/t72.points ] && POINTS=$(cat /root/dsv41/runs/t72.points)
POINTS=${POINTS:-e72:32 e72:40 e72:24 e64:64}   # POINTS: resume after run_dspark.sh took the box
log "t72 start: $POINTS"
for p in $POINTS; do
  nt=0; [ "${p%%:*}" = nt ] && { nt=1; p=${p#nt:}; }
  m=${p%%:*}; c=${p##*:}; N=t72_${m}_c$c; [ $nt = 1 ] && N=t72nt_${m}_c$c
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-$m SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N && [ $nt = 1 ]; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    R=/root/dsv41/runs/$N
    $PY $S/trace_shapes.py --snapshot notools --shapes $TS/notools_e043_e72.json --conc $c --seed 1 --out $R/snap.json >> $L
    # decode_window waits for every stream's first token, so a snapshot over the KV would hang: S4 fit in 160K units
    kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1 | tr -dc 0-9)
    need=$($PY -c "import json,sys; print(int(sum((1600 + 0.355 * x) * 2069257 / 1267921 for x in json.load(open(sys.argv[1])))))" $R/snap.json)
    if [ $need -gt $((kv * 95 / 100)) ]; then log "$N no-tools snapshot C $c: skipped, need $need of KV $kv"
    else log "$N no-tools snapshot C $c (need $need of KV $kv): $($PY $S/decode_window.py --ctx-file $R/snap.json --window 120 --seed $c \
      --sock-read 14400 --max-model-len 163840 --json $R/notools_c$c.json 2>&1 | tail -1 \
      | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-600)"
    fi
  elif [ -s /root/dsv41/runs/$N/server.pid ] && curl -s -m 3 localhost:8100/v1/models > /dev/null; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    log "$N tools C $c: $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc $c --warm-s 300 \
      --window-s 2400 --tool-delay 1 --seed 1 --json /root/dsv41/runs/$N/r2_tools_c$c.json 2>&1 | tail -1 | cut -c1-900)"
  fi
  $S/stop.sh $N | tee -a $L
done
log "T72_DONE (server stopped; box idle)"
