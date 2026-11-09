#!/bin/bash
# pod 3: one {C1} x {variant} run: Manh's production cell 2m (verbatim, + EXTRA_VLLM_ARGS), contract check, then the
# 80-worker image-harness load run (pod 2's launch_ext_e72c152.sh with C1's dir). Status lines in /root/e066/<name>/status.
# Usage: run_variant.sh <name> '<EXTRA_VLLM_ARGS json list>'
N=$1; X=${2:-[]}; D=/root/e066/$N; mkdir -p $D; cd /root/e066
st() { echo "$(date -u +%T) $*" >> $D/status; }
st "start extra=$X"
export MANH_VISION=1 MANH_VISION_MAX_IMAGES=32 ARC_PORT=18084 ARC_CONTEXT_LIMIT=163840 EXTRA_VLLM_ARGS="$X"
T=$(date +%s)
/root/b32m/venv/bin/python -u run_prod_cell.py manh_prod_cell2m.py /root/e066/work /root/models/c1 $D/cell2m.json > $D/cell2m.log 2>&1 || { st "FAIL_CELL"; exit 1; }
SLOG=$(python3 -c "import json;print(json.load(open('$D/cell2m.json'))['MANH_SERVER_LOG'])")
st "SERVER_UP after $(( $(date +%s) - T ))s log $SLOG"
grep -m1 "GPU KV cache size" "$SLOG" >> $D/status
setsid nohup bash /root/e066/scrape_metrics.sh http://127.0.0.1:18084 $D/metrics_scrape.txt > /dev/null 2>&1 < /dev/null &
st "SCRAPER started"
grep -m1 "Available KV cache memory" "$SLOG" >> $D/status
/root/b32m/venv/bin/python run_contract.py http://127.0.0.1:18084 /root/kit_img $D/contract.json > $D/contract.log 2>&1
st "CONTRACT fail=$(python3 -c "import json;print(json.load(open('$D/contract.json')).get('fail'))")"
bash /root/e066/launch_ext_e72c152.sh ${N}_x80 "$SLOG" /root/models/c1 /root/kit_img 80 > $D/launch.log 2>&1
st "LAUNCHED run /root/runs/${N}_x80"
