#!/bin/bash
# E060: one harness run on the box against the local Flash-Next server, with a vLLM metrics monitor beside it.
#   bash run_arm.sh ARM TAG WORKERS [more harness args...]     ARM in A|B|C (cfg_ARM.txt + extra_ARM.json)
# Results: /root/e060/runs/results/TAG.jsonl (+ trace dir TAG/), log runs/TAG.log, monitor runs/monitor_TAG.jsonl.
set -u
ARM=$1; TAG=$2; WORKERS=$3; shift 3
E=/root/e060; PORT=${PORT:-18321}
mkdir -p $E/runs/results
CFG="$(cat $E/repo/exp/cfg_$ARM.txt)"
EXTRA="$(cat $E/repo/exp/extra_$ARM.json)"
$E/hvenv/bin/python $E/repo/scripts/vllm_monitor.py --url http://127.0.0.1:$PORT --out $E/runs/monitor_$TAG.jsonl --every 15 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
cd $E
echo "[$(date -u +%FT%TZ)] arm $ARM tag $TAG workers $WORKERS cfg: $CFG extra: $EXTRA args: $*"
E060_TOOL_PYTHON=$E/hvenv/bin/python $E/hvenv/bin/python $E/repo/scripts/run_local_tools.py $CFG \
  --model flashnext --base-url http://127.0.0.1:$PORT/v1 --extra-body "$EXTRA" --sandbox-url local \
  --challenges $E/repo/data/arc-agi-2-github/data/evaluation --workers "$WORKERS" \
  --out $E/runs/results/$TAG.jsonl "$@"
echo "[$(date -u +%FT%TZ)] RUN END rc=$?"
