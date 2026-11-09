#!/usr/bin/env bash
# E058: re-run Qwen image-arm attempts that failed on Alibaba/OpenRouter errors (429 rate limits, "Provider returned
# error"), gently: 4 workers, 20 min apart, up to 6 passes; each resume runs only the errored attempts.
E=/home/vu/Workspace/ARC-AGI-2; F=$E/evals/results/tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e058q_pilot30.jsonl
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
  CFG="$(cat ../experiments/E048_qwen38_flash_openrouter/cfg_tools.txt)"
  ../.venv/bin/python run_deepseek_tools.py $CFG --grid-format text+image --image-cell 0 --image-align qwen \
    --attempts 2 --workers 4 --sandbox-url http://172.18.0.2:6000 --tag e058q_pilot30 >> results/log_e058q_pilot30.txt 2>&1
done
echo "[$(date -u +%H:%M)] QWEN_RETRY_DONE: $(left) errored attempts left"
