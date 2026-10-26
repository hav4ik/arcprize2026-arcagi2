#!/usr/bin/env bash
# E048: re-run attempts that ended in an API/network error (not model failures), into a separate retry file, while
# the main run may still be going. Only (task, pair, attempt) whose latest record across main + retry files is an
# error are run; scoring takes the latest successful record per attempt across both files. Safe to re-run.
set -euo pipefail
E=/home/vu/Workspace/ARC-AGI-2
D=$E/evals/results/tools_persistent_grid_lean2sub_qwen
MAIN=$D/qwen3.8-flash_low_e048_qwenflash_sub80s40.jsonl
RETRY=$D/qwen3.8-flash_low_e048_qwenflash_sub80s40_retry.jsonl
CFG="$(cat $E/experiments/E048_qwen38_flash_openrouter/cfg_tools.txt)"
cd $E/evals
for a in 0 1; do
  PAIRS=$(python3 - "$MAIN" "$RETRY" "$a" <<'EOF'
import json, os, sys
main, retry, a = sys.argv[1], sys.argv[2], int(sys.argv[3])
latest = {}
for f in (main, retry):
    if os.path.exists(f):
        for line in open(f):
            r = json.loads(line)
            k = (r["task_id"], r["pair"], r["attempt"])
            if "error" not in r or k not in latest or "error" in latest[k]:
                latest[k] = r  # a success is never overwritten by a later error
print(",".join(f"{t}:{p}" for (t, p, att), r in sorted(latest.items()) if att == a and "error" in r))
EOF
)
  [ -z "$PAIRS" ] && { echo "attempt $a: no API errors to retry"; continue; }
  echo "attempt $a: retrying $PAIRS"
  ../.venv/bin/python run_deepseek_tools.py $CFG --attempts 2 --only-attempts $a --pairs "$PAIRS" --workers 8 \
    --out "$RETRY.a$a.jsonl" >> $E/evals/results/log_e048_qwenflash_sub80s40_retry.txt 2>&1 &
done
wait
for a in 0 1; do [ -f "$RETRY.a$a.jsonl" ] && cat "$RETRY.a$a.jsonl" >> "$RETRY" && rm "$RETRY.a$a.jsonl"; done
echo "retry pass done: $(wc -l < "$RETRY") rows in $RETRY"
