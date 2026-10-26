#!/bin/bash
D=$(dirname $0); cd /home/vu/Workspace/ARC-AGI-2
args=()
while read -r id port host run; do
  [ -z "$id" ] && continue
  timeout 90 scp -q -o ConnectTimeout=15 -P $port root@$host:/root/runs/$run/out/dsv41_logs/attempts.jsonl $D/acc/$run.jsonl 2>/dev/null && args+=("$run=$D/acc/$run.jsonl")
done < $D/pods.txt
.venv/bin/python experiments/E042_dsv41_l4_synthetic/scripts/bench_accuracy.py data/arc-agi_evaluation_solutions.json "${args[@]}"
echo ---
.venv/bin/python $D/answer_rate.py data/arc-agi_evaluation_solutions.json "${args[@]}"
