#!/usr/bin/env bash
# E048: keep retrying API/network errors. Tool run: every 15 min while it runs, then until clean (max 4 passes).
# No-tool run: after it exits, resume it (the script re-runs only errored calls), up to 3 times.
E=/home/vu/Workspace/ARC-AGI-2; X=$E/experiments/E048_qwen38_flash_openrouter
TP=$(cat $X/run_tools.pid); NP=$(cat $X/run_notool.pid)
count_tool_errors() { python3 - <<'PY'
import json, os
d = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e048_qwenflash_sub80s40"
latest = {}
for f in (d + ".jsonl", d + "_retry.jsonl"):
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l); k = (r["task_id"], r["pair"], r["attempt"])
            if "error" not in r or k not in latest or "error" in latest[k]:
                latest[k] = r
print(sum("error" in r for r in latest.values()))
PY
}
while kill -0 $TP 2>/dev/null; do sleep 900; kill -0 $TP 2>/dev/null || break; echo "[$(date -u +%H:%M)] periodic pass, errors=$(count_tool_errors)"; $X/retry_api_errors.sh; done
for i in 1 2 3 4; do n=$(count_tool_errors); echo "[$(date -u +%H:%M)] tool run done; errors left=$n"; [ "$n" = 0 ] && break; $X/retry_api_errors.sh; done
while kill -0 $NP 2>/dev/null; do sleep 120; done
cd $E/evals
for i in 1 2 3; do
  n=$(python3 -c "import json; print(sum(1 for l in open('results/arcprize_qwen38-flash_e048_notool.jsonl') if 'error' in json.loads(l)))")
  ok=$(python3 -c "import json; print(len({(d['task_id'],d['pair'],d['attempt']) for d in map(json.loads, open('results/arcprize_qwen38-flash_e048_notool.jsonl')) if 'error' not in d}))")
  echo "[$(date -u +%H:%M)] no-tool done; ok calls=$ok/334, error rows=$n"; [ "$ok" -ge 334 ] && break
  ../.venv/bin/python run_deepseek_arcprize.py $(cat $X/cfg_notool.txt | sed "s/'//g") >> results/log_e048_qwenflash_notool.txt 2>&1
done
echo "[$(date -u +%H:%M)] AUTO_RETRY_DONE"
