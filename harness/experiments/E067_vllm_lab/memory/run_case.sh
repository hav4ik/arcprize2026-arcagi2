#!/usr/bin/env bash
# One lab case: start server (env as for run_server.sh), run the short standard
# workload (ladder 2k..61k, one burst of 6 x 30k prefills, 150 s of 32-way mixed
# traffic 512..32k prompts), stop, print the summary.
#   NAME=... UTIL=... MBT=... [MEMLAB_CAP_MB=512 MEMLAB_EXPANDABLE_AFTER=1 ...] ./run_case.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
L="${LOGDIR:-$SCR/memory/logs}"
P=/home/vu/miniconda3/envs/vllm/bin/python
cd "$HERE"
rm -f "$L/$NAME.req.jsonl"
if ./run_server.sh; then
  $P drive.py ladder --log "$L/$NAME.req.jsonl" 2>&1 | tail -3
  $P drive.py burst --conc 6 --len 30000 --out-tokens 64 --rounds 1 --log "$L/$NAME.req.jsonl" 2>&1 | tail -1
  $P drive.py mix --conc 32 --seconds 150 --lo 512 --hi 32768 --olo 32 --ohi 384 --seed 7 \
     --log "$L/$NAME.req.jsonl" 2>&1 | tail -1
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > "$L/$NAME.nvsmi_end.txt"
grep -c '"status": 200' "$L/$NAME.req.jsonl" 2>/dev/null | sed 's/^/ok requests: /'
grep -E "OutOfMemory|CUDA out of memory|EngineDeadError|EngineCore.*died" "$L/$NAME.server.log" | head -3 | cut -c1-300
./stop_server.sh "$NAME"
python3 analyze.py $(ls "$L/$NAME".*.jsonl | grep -v req)
