#!/bin/bash
# Run dash_sync.sh every 300 s until <stop file> exists or status.json shows the run is over (> 12.2 Kaggle h).
D=$(dirname $0); STOP=$D/dash_loop.stop; S=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data/status.json
while [ ! -e $STOP ]; do
  t=$(date +%s); out=$($D/dash_sync.sh 2>&1 | tail -1); echo "$(date -u +%T) ${out:0:160}"
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$S')).get('kaggle_h', 0) < 12.2 else 1)" 2>/dev/null || { echo "run over"; break; }
  sleep $(( 300 - ($(date +%s) - t) > 10 ? 300 - ($(date +%s) - t) : 10 ))
done
