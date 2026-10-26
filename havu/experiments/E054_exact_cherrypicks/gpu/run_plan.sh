#!/usr/bin/env bash
# E054 GPU plan on one 4x4090 box (README "GPU test plan"). Each step is a subcommand; run them in this order:
#
#   gpu/run_plan.sh prep          #  ~5 min  install the 2 files (sha-checked, backups), CPU checks against the venv
#   gpu/run_plan.sh kernel        #  ~8 min  (a) #156 kernel A/B, GPU 0: bitwise eager + graph replay, top-512, us/call
#   gpu/run_plan.sh engram        #  ~6 min  (c1) #58678 wkv/Engram output A/B on the real E128 weights, GPU 0
#   gpu/run_plan.sh arm old       # ~80 min  server: startup numbers, probe x3, decode B=48/36, replays C=36/48
#   gpu/run_plan.sh arm idx       # ~22 min  server: startup, probe, decode B=48/36 (isolates #156)
#   gpu/run_plan.sh arm both      # ~80 min  as old (isolates #58678 against idx; combined against old)
#   gpu/run_plan.sh report        #   1 min  go / no-go JSON (gpu/report_e054.py)
#
# Total ~3 h 20 min (REPLAY_WALL=1800, the 30-min A/B per arm and stream count); REPLAY_WALL=1500 ~3 h.
# Env: VENV=/root/dsv41/venv MODEL_DIR=/root/dsv41/model-e128t R=/root/e054/runs PORT=8100 REPLAY_WALL=1800
# Helper scripts come from the repo copy next to this dir (rsync -R, README): E042 decode_window.py and
# tool_replay.py, the E128 tool shapes (E043 results), E051's metrics_logger.py.
# Servers are stopped by their process group only (never by name: other agents share these boxes).
set -euo pipefail
E054=$(cd "$(dirname "$0")/.." && pwd)
X=$(dirname "$E054")
VENV=${VENV:-/root/dsv41/venv}
PY=$VENV/bin/python
MODEL_DIR=${MODEL_DIR:-/root/dsv41/model-e128t}
R=${R:-/root/e054/runs}
PORT=${PORT:-8100}
URL=http://127.0.0.1:$PORT
REPLAY_WALL=${REPLAY_WALL:-1800}
DW=${DW:-$X/E042_dsv41_l4_synthetic/scripts/decode_window.py}
TR=${TR:-$X/E042_dsv41_l4_synthetic/scripts/tool_replay.py}
SHAPES=${SHAPES:-$X/E043_dsv41_real_pruning/results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json}
ML=${ML:-$X/E051_kv_offload/gpu/metrics_logger.py}
mkdir -p "$R"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

ctx_files() {  # 48 contexts, 8K-40K tokens (mean ~24K, 1.15M total: fits Kaggle's 1.554M-token pool with room
  # to decode); B=36 uses the first 36, i.e. the same seeded prompts, so its prefill is a prefix-cache hit
  "$PY" -I - "$R" <<'EOF'
import json, random, sys
rng = random.Random(54)
ctx = [rng.randrange(8192, 40960) for _ in range(48)]
json.dump(ctx, open(f"{sys.argv[1]}/ctx48.json", "w"))
json.dump(ctx[:36], open(f"{sys.argv[1]}/ctx36.json", "w"))
print("ctx48 mean", sum(ctx) // 48, "sum", sum(ctx), "| ctx36 mean", sum(ctx[:36]) // 36)
EOF
}

stop_server() {
  local d=$1
  [ -f "$d/metrics_logger.pid" ] && kill "$(cat "$d/metrics_logger.pid")" 2>/dev/null || true
  if [ -f "$d/server.pgid" ]; then
    kill -- "-$(cat "$d/server.pgid")" 2>/dev/null || true
    for _ in $(seq 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "$used" -lt 500 ] && break
      sleep 5
    done
    log "server stopped ($d); GPU max used ${used} MiB"
  fi
}

case "${1:-}" in
  prep)
    for f in "$DW" "$TR" "$SHAPES" "$ML"; do [ -f "$f" ] || { echo "missing $f (README: rsync -R the helpers)"; exit 1; }; done
    bash "$E054/gpu/install_e054.sh" "$VENV"
    bash "$E054/tests/run_cpu_checks.sh" "$PY" "$("$PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
      2>&1 | tee "$R/cpu_checks.log" | tail -5
    ;;
  kernel)
    mkdir -p "$R/kernel"
    CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" "$E054/gpu/indexer_ab.py" --json "$R/kernel/indexer_ab.json" \
      2>&1 | tee "$R/kernel/indexer_ab.log" | tail -3
    ;;
  engram)
    mkdir -p "$R/kernel"
    CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" "$E054/gpu/engram_wkv_ab.py" --model "$MODEL_DIR" \
      --json "$R/kernel/engram_wkv_ab.json" 2>&1 | tee "$R/kernel/engram_wkv_ab.log" | tail -3 || true
    # informational: the Ada MXFP8 kernel (no --linear-backend humming), whose split-K depends on N
    CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" "$E054/gpu/engram_wkv_ab.py" --model "$MODEL_DIR" --backend auto \
      --tokens 1,36,48,64,2048 --json "$R/kernel/engram_wkv_ab_ada.json" 2>&1 | tail -2 || true
    ;;
  arm)
    ARM=${2:?arm old|idx|both|eng}
    D=$R/$ARM
    [ ! -e "$D" ] || { echo "$D exists (one run per arm; move it away to rerun)"; exit 1; }
    [ -f "$R/ctx48.json" ] || ctx_files
    log "arm $ARM: starting server"
    ARM=$ARM RUN=$ARM VENV=$VENV MODEL_DIR=$MODEL_DIR PORT=$PORT bash "$E054/gpu/serve_e054.sh"
    trap 'stop_server "$D"' EXIT
    setsid nohup "$PY" -I "$ML" --url "$URL" --out "$D/metrics.jsonl" --pgid "$(cat "$D/server.pgid")" --interval 5 \
      < /dev/null > "$D/metrics_logger.out" 2>&1 &
    echo $! > "$D/metrics_logger.pid"
    python3 "$E054/gpu/startup_report.py" "$D" > "$D/startup.json"
    log "startup: $(cat "$D/startup.json")"
    "$PY" -I "$E054/gpu/logprob_probe.py" --url "$URL" --repeats 3 --json "$D/probe.json" > "$D/probe.log" 2>&1
    log "probe done"
    for B in 48 36; do
      "$PY" -I "$DW" --url "$URL" --model dsv41 --ctx-file "$R/ctx$B.json" --warm 16 --window 60 \
        --max-tokens 8000 --max-model-len 163840 --json "$D/dw$B.json" > "$D/dw$B.log" 2>&1
      log "decode B=$B: $(tail -1 "$D/dw$B.log" | cut -c1-300)"
    done
    if [ "$ARM" = old ] || [ "$ARM" = both ] || [ -n "${REPLAY:-}" ]; then
      for C in 36 48; do
        "$PY" -I "$TR" --shapes "$SHAPES" --url "$URL" --model dsv41 --conc "$C" --tool-delay 1 --seed 1 \
          --warm-s 300 --window-s 1500 --max-wall-s "$REPLAY_WALL" --json "$D/replay$C.json" > "$D/replay$C.log" 2>&1
        log "replay C=$C: $(tail -1 "$D/replay$C.log" | cut -c1-300)"
      done
    fi
    python3 "$E054/gpu/startup_report.py" "$D" > "$D/startup.json"  # again: tracebacks during the run
    ;;
  report)
    python3 "$E054/gpu/report_e054.py" "$R" | tee "$R/report.json"
    ;;
  *)
    sed -n 2,16p "$0"; exit 2 ;;
esac
