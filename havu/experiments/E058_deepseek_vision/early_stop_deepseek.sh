#!/usr/bin/env bash
# E058: stop the DeepSeek image pilot early once >= 20 tasks are complete and the 95% CI upper bound of the pass@2
# gain (images - text) is below +5 pp, i.e. images clearly won't help enough to matter. Kills by PID (process tree
# of the off-peak launcher), never by name.
E=/home/vu/Workspace/ARC-AGI-2; X=$E/experiments/E058_deepseek_vision
R=$E/evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_
L=$(cat $X/pilot_launcher.pid)
while kill -0 $L 2>/dev/null; do
  sleep 600
  [ -f ${R}e058_pilot30.jsonl ] || continue
  out=$($E/.venv/bin/python $X/compare.py ${R}e058_pilot30.jsonl ${R}e046p_note_all.jsonl --json)
  echo "[$(date -u +%H:%M)] $out"
  stop=$(echo "$out" | python3 -c "import json,sys; r=json.load(sys.stdin); print(int(r['tasks'] >= 20 and r['hi'] < 0.05))")
  if [ "$stop" = 1 ]; then
    for p in $(ps --ppid $L -o pid=); do for c in $(ps --ppid $p -o pid=); do kill $c; done; kill $p; done
    echo "[$(date -u +%H:%M)] EARLY STOP: images won't help (CI upper bound < +5 pp at >= 20 tasks)"
    break
  fi
done
echo "[$(date -u +%H:%M)] watcher done"
