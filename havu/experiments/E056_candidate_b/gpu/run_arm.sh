#!/usr/bin/env bash
# E056: one server start for one arm and its tests, then the server is stopped by its process group (also on error).
#   ARM=candb   RUN=late_candb   PARITY=1 setsid nohup bash /root/e056/gpu/run_arm.sh &   # steps b, c, d (candB)
#   ARM=offload RUN=late_offload          setsid nohup bash /root/e056/gpu/run_arm.sh &   # step d (offload only)
# Progress lines go to /root/e056/runs/$TAG.progress (TAG=late), in E051 run_capacity.sh's format so E051's
# summarize_capacity.py reads them (arm dirs $TAG_<arm>). Steps, as E051's run_capacity.sh / chain_box1b.sh:
#   server start (serve_candb.sh, EXT=1) + metrics_logger.py (5 s, server PSS); startup facts; indexer probe;
#   [PARITY=1: E051 kv_parity.py --lengths 3000 20000 60000 120000 --max-tokens 300 (byte-level, private memory)];
#   smoke replay (4 streams, 60 s); reset GPU + host prefix caches; the late-phase "Kaggle hour 8" replay
#   (E051 late arms: --conc 36 --tool-delay 20 --scale 1.25 --tail-frac 0.3 --min-start-ctx 30000 --warm-s 600
#   --window-s 1500 --max-wall-s 5400 --seed 0); indexer probe again; stop.
set -uo pipefail
ARM=${ARM:?}; RUN=${RUN:?}; TAG=${TAG:-late}
K=/root/e056; G=$K/gpu; E=$K/e051/gpu; R=$K/runs; D=$R/$RUN
PY=/root/dsv41/venv_candb/bin/python
URL=http://127.0.0.1:8100
SHAPES=$K/shapes/tools_e043_e128t.json
PROG=$R/$TAG.progress
mkdir -p "$R"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$PROG"; }
hostmem() { awk '/^(MemTotal|MemAvailable|Shmem|Mlocked|Unevictable):/{printf "%s %.1f; ", $1, $2/1048576}' /proc/meminfo
            free -g | awk 'NR==2{printf "used %s GiB", $3}'; }
rpc() { curl -s -X POST "$URL/collective_rpc" -H 'Content-Type: application/json' -d "{\"method\": \"$1\"}"; }
log "$RUN: host before (GiB): $(hostmem)"
ARM=$ARM RUN=$RUN EXT=1 bash "$G/serve_candb.sh" >> "$PROG" 2>&1 || { log "$RUN: start failed"; exit 1; }
PG=$(cat "$D/server.pgid"); P=$(cat "$D/server.pid")
stop() {
  [ -f "$D/metrics_logger.pid" ] && kill "$(cat "$D/metrics_logger.pid")" 2>/dev/null
  kill -- "-$PG" 2>/dev/null
  for _ in $(seq 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if ! pgrep -g "$PG" > /dev/null && [ "$used" -lt 500 ]; then break; fi
    sleep 2
  done
  pgrep -g "$PG" > /dev/null && { log "$RUN: group $PG still alive, SIGKILL"; kill -9 -- "-$PG" 2>/dev/null; sleep 5; }
  log "$RUN: stopped; GPU max used ${used} MiB; tracebacks $(grep -c -E 'Traceback|AssertionError' "$D/server.log"); host after stop (GiB): $(hostmem)"
}
trap stop EXIT
setsid nohup "$PY" -I "$E/metrics_logger.py" --url "$URL" --out "$D/metrics.jsonl" --pgid "$PG" --interval 5 \
  < /dev/null > "$D/metrics_logger.out" 2>&1 &
echo $! > "$D/metrics_logger.pid"
t0=$(date +%s)
until grep -q "Application startup complete" "$D/server.log"; do
  kill -0 "$P" 2>/dev/null || { log "$RUN: server exited during startup"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 2400 ] && { log "$RUN: startup timeout"; exit 1; }
  sleep 10
done
log "$RUN: ready after $(( $(date +%s) - t0 ))s; host (GiB): $(hostmem)"
{ grep -m4 -E "KV offloading:" "$D/server.log"; grep -m2 -E "GPU KV cache size|num_gpu_blocks|Maximum concurrency" "$D/server.log"
  grep -E "E054 DSv4.1 indexer" "$D/server.log"; grep -m8 -iE "pinned|cpu_bytes|offload.*(GiB|bytes)" "$D/server.log"; } \
  | cut -c1-400 > "$D/startup.txt"
log "$RUN: $(grep -c 'KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups \[8\]' "$D/server.log") offload lines, $(grep -c 'grid-stride context skip on' "$D/server.log") skip-on / $(grep -c 'grid-stride context skip off' "$D/server.log") skip-off indexer lines, tracebacks $(grep -c -E 'Traceback|AssertionError' "$D/server.log")"
rpc e056_indexer_state > "$D/indexer_state_start.json"; rpc e056_loaded_files > "$D/loaded_files.json"
if [ "${PARITY:-0}" = 1 ]; then
  log "$RUN: kv parity start"
  (cd "$D" && "$PY" -I "$E/kv_parity.py" --lengths 3000 20000 60000 120000 --max-tokens 300 \
     --json "$D/kv_parity.json" > "$D/kv_parity.out" 2>&1)
  log "$RUN: parity rc=$? $(tail -1 "$D/kv_parity.out"); tracebacks $(grep -c -E 'Traceback|AssertionError' "$D/server.log")"
fi
"$PY" -I "$E/tool_replay_kv.py" --shapes "$SHAPES" --url "$URL" --model dsv41 \
  --conc 4 --tool-delay 1 --warm-s 30 --window-s 60 --max-wall-s 900 --seed 1 --json "$D/smoke.json" > "$D/smoke.out" 2>&1
rc=$?
log "$RUN: smoke rc=$rc $(tail -1 "$D/smoke.out" | cut -c1-300)"
[ "$rc" -eq 0 ] || exit 1
[ "${SKIP_REPLAY:-0}" = 1 ] && exit 0
C=36; DL=20; S=1.25; TF=0.3; MS=30000
curl -s -X POST "$URL/reset_prefix_cache?reset_external=true" > /dev/null
sleep 5
out=$D/replay_C${C}_D${DL}_S${S}_T${TF}_M${MS}
log "$RUN: replay C=$C D=$DL S=$S T=$TF M=$MS start"
"$PY" -I "$E/tool_replay_kv.py" --shapes "$SHAPES" --url "$URL" --model dsv41 --conc "$C" --tool-delay "$DL" \
  --warm-s 600 --window-s 1500 --max-wall-s 5400 --scale "$S" --tail-frac "$TF" --min-start-ctx "$MS" --seed 0 \
  --json "$out.json" > "$out.out" 2>&1
rc=$?
log "$RUN: replay C=$C D=$DL S=$S T=$TF M=$MS rc=$rc $(tail -1 "$out.out" | cut -c1-400)"
rpc e056_indexer_state > "$D/indexer_state_end.json"
kill -0 "$P" 2>/dev/null || log "$RUN: server died"
log "$RUN: done"
