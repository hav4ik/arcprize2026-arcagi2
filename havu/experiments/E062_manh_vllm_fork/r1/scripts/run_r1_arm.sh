#!/usr/bin/env bash
# E062 R1: one arm end to end on the box: server (serve_r1.sh), startup numbers, E054's 5-prompt cold probe, the long
# tool-transcript probe, decode windows at B = 48 / 56 / 64 (4,096-token seeded random prompts; B32's max-num-seqs is
# 64), then the server is stopped by its process group.
#
#   RUN=r1_pad6464 PAD=64,64 bash run_r1_arm.sh        # PAD="" = B32's padding (256,128)
set -uo pipefail
RUN=${RUN:?}; X=/root/e062/r1; D=/root/e062/runs/$RUN; PY=/root/dsv41/venv/bin/python; URL=http://127.0.0.1:8100
P=/root/e062/runs/$RUN.progress
log() { echo "$(date -u +%FT%TZ) $*" >> "$P"; }
mkdir -p /root/e062/runs
log "arm $RUN PAD=${PAD:-default}"
RUN=$RUN PAD=${PAD:-} UTIL=${UTIL:-0.938} bash $X/serve_r1.sh >> "$P" 2>&1 || { log "serve_r1 failed"; exit 1; }
stop() { pg=$(cat "$D/server.pgid" 2>/dev/null); [ -n "$pg" ] && [ "$pg" != 1 ] && kill -TERM -- "-$pg" 2>/dev/null
  for _ in $(seq 60); do pgrep -g "$pg" > /dev/null 2>&1 || break; sleep 2; done
  pgrep -g "$pg" > /dev/null 2>&1 && kill -KILL -- "-$pg" 2>/dev/null; log "server group $pg stopped"; }
trap stop EXIT
up=0
for i in $(seq 240); do
  if curl -sf -m 5 $URL/v1/models > /dev/null; then up=1; break; fi
  kill -0 "$(cat $D/server.pid)" 2>/dev/null || break
  sleep 5
done
if [ $up = 0 ]; then log "SERVER NOT UP: $(grep -E 'Error|error|assert' $D/server.log | tail -3 | cut -c1-300)"; exit 1; fi
log "server up after $(( i * 5 )) s"
python3 -I $X/startup_report.py "$D" > "$D/startup.json" 2>&1; log "startup: $(cut -c1-600 "$D/startup.json")"
grep -E "Model loading took|GPU KV cache size|Maximum concurrency|Available KV cache memory" "$D/server.log" | head -12 > "$D/startup_lines.txt"
$PY -I $X/logprob_probe.py --url $URL --repeats 3 --json "$D/probe.json" > "$D/probe.log" 2>&1; log "probe5 rc=$?"
$PY -I $X/probe_long.py --url $URL --messages $X/probe_long.json --repeats 3 --json "$D/probe_long.json" > "$D/probe_long.log" 2>&1
log "probe_long rc=$? $(tail -1 "$D/probe_long.log" | cut -c1-200)"
for B in 48 56 64; do
  $PY -c "import json; json.dump([4096] * $B, open('$D/ctx$B.json', 'w'))"
  $PY -I $X/decode_window.py --url $URL --model dsv41 --ctx-file "$D/ctx$B.json" --warm 16 --window 60 \
    --max-tokens 3000 --max-model-len 163840 --seed 1 --json "$D/dw$B.json" > "$D/dw$B.log" 2>&1
  log "decode B=$B rc=$?: $(tail -1 "$D/dw$B.log" | cut -c1-300)"
done
python3 -I $X/startup_report.py "$D" > "$D/startup_end.json" 2>&1
log "ARM DONE $RUN"
