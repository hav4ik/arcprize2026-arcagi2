#!/bin/bash
# Stop run 2 (notebook in rerun mode) by PID / process group only. PIDs are discovered first, each checked by its
# cmdline, then killed: papermill + kernel (so the notebook cannot move on), the solver + its sandbox workers, then
# Manh's server (SIGTERM, 30 s, SIGKILL on the group), the scraper and the memory logger. Ends with the GPU process list.
# Usage: stop_run2.sh [--dry]  (--dry: print what it found, kill nothing)
set -u
L=/root/e066/r3/stop_run2.log; exec > >(tee -a $L) 2>&1
echo "stop_run2 $(date -u +%T)"
one() { local p; p=$(pgrep -f -- "$1" | grep -vx "$$" | head -2); [ "$(echo "$p" | grep -c .)" = 1 ] && echo "$p"; }
pg() { ps -o pgid= -p "$1" | tr -d ' '; }
PM=$(one "papermill --kernel py313b32m"); KN=$(one "ipykernel_launcher -f"); SV=$(one "/tmp/dsv41/kit/solver_dsv41.py")
SRV=$(python3 -c 'import glob,json; f=sorted(glob.glob("/kaggle/working/dsv41_server_*/server/launch.json")+glob.glob("/kaggle/working/dsv41_logs/server/launch.json")); print(json.load(open(f[-1]))["pid"] if f else "")')
SB=$( [ -n "$SV" ] && ps --ppid "$SV" -o pid= | tr -d ' ' | tr '\n' ' ')
echo "papermill=$PM (pgid $(pg ${PM:-0})) kernel=$KN (pgid $(pg ${KN:-0})) solver=$SV sandboxes=$(echo $SB | wc -w) server=$SRV"
[ "${1:-}" = "--dry" ] && { echo "dry run: nothing killed"; exit 0; }
[ -n "$PM" ] && [ -n "$KN" ] && [ -n "$SV" ] && [ -n "$SRV" ] || { echo "a process was not found uniquely; nothing killed"; exit 1; }
for x in "$PM" "$KN"; do [ -n "$x" ] && kill -KILL -- -"$(pg $x)" && echo "killed pgid $(pg $x 2>/dev/null || echo gone) of $x"; done
[ -n "$SV" ] && kill -KILL -- -"$SV" 2>/dev/null; for p in $SB; do ps -o args= -p $p | grep -q "import ast, contextlib" && kill -KILL $p; done
echo "solver + sandboxes killed $(date -u +%T)"
if [ -n "$SRV" ] && ps -o args= -p "$SRV" | grep -q "vllm.entrypoints.cli.main serve"; then
  kill -TERM -- -"$SRV"; for i in $(seq 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$SRV" 2>/dev/null; echo "server group $SRV stopped $(date -u +%T)"
fi
for p in $(pgrep -f "scrape_metrics.sh http://127.0.0.1:18084"); do kill $p; done
M=$(one "/root/memlog.sh /root/e066/nbrun/mem.log"); [ -n "$M" ] && kill -- -"$(pg $M)"
sleep 5; echo "GPU processes:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "left: $(pgrep -f 'VLLM::|solver_dsv41|ipykernel_launcher|papermill' | wc -l) processes"; free -g | head -2
echo "STOP_DONE $(date -u +%T)"
