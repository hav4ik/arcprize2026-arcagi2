#!/usr/bin/env bash
# E069: wait for the 120-task run, re-run API/server-errored attempts once, then STOP the Vast box (user: the box is
# only for this benchmark; stop, not destroy). Results live locally in box/results/ (the harness runs here).
E=/home/vu/Workspace/ARC-AGI-2; X=$E/experiments/E069_qat_sft_eval/box; F=$X/results/e069_qat025_full.jsonl
P=$(cat $X/run_full.pid)
while kill -0 $P 2>/dev/null; do sleep 120; done
echo "[$(date -u +%H:%M)] main run ended: $(wc -l < $F) rows"
n=$(python3 -c "
import json
latest = {}
for l in open('$F'):
    r = json.loads(l); latest[r['task_id'], r['pair'], r['attempt']] = r
print(sum(1 for r in latest.values() if r.get('error')))")
echo "errored attempts: $n"
if [ "$n" -gt 0 ]; then
  CHALLENGES=$E/data/arc-agi_evaluation_challenges.json $X/run_b32_harness.sh e069_qat025_full --attempts 2 --workers 64 >> $X/logs/e069_qat025_full.log 2>&1
  echo "[$(date -u +%H:%M)] resume pass done: $(wc -l < $F) rows"
fi
K="$(sed -n 's/^vast_ai_api_key:[[:space:]]*//p' $E/SECRETS.yaml | tr -d "\"' ")"
$E/.venv/bin/vastai stop instance 58398827 --api-key "$K" && echo "[$(date -u +%H:%M)] VAST_STOPPED 58398827"
