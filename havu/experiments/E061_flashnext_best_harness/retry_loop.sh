#!/usr/bin/env bash
# E061: re-run Flash-Next attempts that failed on Alibaba/OpenRouter errors (429 rate limits, "Provider returned
# error"), gently: 4 workers, 20 min apart, up to 6 passes; each resume runs only the errored attempts.
TAG=${1:-e061q_best_img}
E=/home/vu/Workspace/ARC-AGI-2; F=$E/evals/results/tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_$TAG.jsonl
left() { python3 -c "
import json
latest = {}
for l in open('$F'):
    r = json.loads(l); latest[r['task_id'], r['pair'], r['attempt']] = r
print(sum(1 for r in latest.values() if r.get('error')))"; }
cd $E/evals
for i in 1 2 3 4 5 6; do
  sleep 1200
  n=$(left); echo "[$(date -u +%H:%M)] pass $i: $n errored attempts"; [ "$n" = 0 ] && break
  bash $E/experiments/E061_flashnext_best_harness/run.sh "$TAG" 4
done
echo "[$(date -u +%H:%M)] QWEN_RETRY_DONE: $(left) errored attempts left"
