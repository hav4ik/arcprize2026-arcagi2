#!/usr/bin/env bash
# Box 57929406 bring-up check: one start of the candidate-B server from venv_candb with the Kaggle smoke's offload
# size (16 GiB = 17179869184 B, private pinned memory, checkpoint, lru), KV pool pinned as E054/E056 on 4090 boxes
# (--gpu-memory-utilization 0.92 --num-gpu-blocks-override 21490), PYTORCH_CUDA_ALLOC_CONF unset (serve_candb.sh).
# Checks: ready, the offload log line, KV pool size, tracebacks, one short chat completion; then the server is stopped
# by its process group (never by name) and the GPUs are left idle.
#   RUN=verify_box57929406 setsid -f bash /root/e056/gpu/verify_candb_start.sh > /root/e056/runs/verify.driver.out 2>&1 < /dev/null
set -uo pipefail
RUN=${RUN:-verify_box57929406}
R=/root/e056/runs; D=$R/$RUN; PROG=$R/$RUN.progress
PY=/root/dsv41/venv_candb/bin/python
URL=http://127.0.0.1:8100
mkdir -p "$R"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$PROG"; }
hostmem() { awk '/^(MemTotal|MemAvailable|Shmem):/{printf "%s %.1f; ", $1, $2/1048576}' /proc/meminfo
            free -g | awk 'NR==2{printf "used %s GiB", $3}'; }
log "$RUN: host before (GiB): $(hostmem)"
ARM=candb RUN=$RUN OFFLOAD_BYTES=17179869184 EXT=0 bash /root/e056/gpu/serve_candb.sh >> "$PROG" 2>&1 \
  || { log "$RUN: start failed"; exit 1; }
PG=$(cat "$D/server.pgid"); P=$(cat "$D/server.pid")
stop() {
  kill -- "-$PG" 2>/dev/null
  for _ in $(seq 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if ! pgrep -g "$PG" > /dev/null && [ "$used" -lt 500 ]; then break; fi
    sleep 2
  done
  pgrep -g "$PG" > /dev/null && { log "$RUN: group $PG still alive, SIGKILL"; kill -9 -- "-$PG" 2>/dev/null; sleep 5; }
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  log "$RUN: stopped; GPU max used ${used} MiB; group alive: $(pgrep -g "$PG" | wc -l); tracebacks $(grep -c -E 'Traceback|AssertionError' "$D/server.log"); host after stop (GiB): $(hostmem)"
  log "$RUN: DONE"
}
trap stop EXIT
t0=$(date +%s)
until grep -q "Application startup complete" "$D/server.log"; do
  kill -0 "$P" 2>/dev/null || { log "$RUN: server exited during startup"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { log "$RUN: startup timeout"; exit 1; }
  sleep 10
done
log "$RUN: ready after $(( $(date +%s) - t0 ))s; host (GiB): $(hostmem); server PSS GiB $(pgrep -g "$PG" | xargs -r -I{} awk '/^Pss:/{s+=$2} END{print s+0}' /proc/{}/smaps_rollup 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1048576}')"
{ grep -m4 -E "KV offloading:" "$D/server.log"; grep -m3 -E "GPU KV cache size|num_gpu_blocks|Maximum concurrency" "$D/server.log"
  grep -E "E054 DSv4.1 indexer" "$D/server.log"; grep -m8 -iE "pinned|cpu_bytes|offload.*(GiB|bytes)" "$D/server.log"; } \
  | cut -c1-400 > "$D/startup.txt"
log "$RUN: $(grep -c 'KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups \[8\]' "$D/server.log") offload lines; $(grep -m1 -o 'GPU KV cache size: [0-9,]* tokens' "$D/server.log"); $(grep -m1 -o 'Overriding num_gpu_blocks=[0-9]* with num_gpu_blocks_override=[0-9]*' "$D/server.log"); skip-on lines $(grep -c 'grid-stride context skip on' "$D/server.log"); tracebacks $(grep -c -E 'Traceback|AssertionError' "$D/server.log")"
curl -s -m 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "dsv41", "messages": [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
  "max_tokens": 2048, "temperature": 0}' > "$D/chat.json"
log "$RUN: chat rc=$? $("$PY" -I -c '
import json, sys
d = json.load(open(sys.argv[1])); m = d["choices"][0]["message"]
r = m.get("reasoning") or m.get("reasoning_content") or ""
print("content=%r finish=%s reasoning_chars=%d usage=%s" % ((m.get("content") or "")[:80], d["choices"][0]["finish_reason"], len(r), d["usage"]))' "$D/chat.json" 2>&1 | tail -1)"
log "$RUN: tracebacks after chat $(grep -c -E 'Traceback|AssertionError' "$D/server.log")"
