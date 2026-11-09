#!/bin/bash
# Stop a notebook run (rerun or commit) by PID / process group only. Discovers papermill, the kernel (its cell-2s
# supervisor thread dies with it, so it cannot relaunch the server), the solver + its sandbox workers, and the live
# server (pid from dsv41_logs/e066_supervisor.json, else the newest launch.json), each checked by cmdline. Kills the
# notebook first, then the solver, then the server group (SIGTERM, 30 s, SIGKILL), then the given helper pids.
# Usage: stop_nb.sh <log> [--dry] [helper pid ...]
set -u
L=$1; shift; DRY=""; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
exec > >(tee -a $L) 2>&1
echo "stop_nb $(date -u +%T)"
one() { local p; p=$(pgrep -f -- "$1" | head -2); [ "$(echo "$p" | grep -c .)" = 1 ] && echo "$p"; }
pg() { ps -o pgid= -p "$1" | tr -d ' '; }
PM=$(one "papermill --kernel py313b32m"); KN=$(one "ipykernel_launcher -f"); SV=$(one "/tmp/dsv41/kit/solver_dsv41.py")
SRV=$(python3 -c '
import glob, json, os
try:
    print(json.load(open("/kaggle/working/dsv41_logs/e066_supervisor.json"))["pid"])
except Exception:
    f = sorted(glob.glob("/kaggle/working/dsv41_logs/**/launch.json", recursive=True), key=os.path.getmtime)
    print(json.load(open(f[-1]))["pid"] if f else "")')
SB=$( [ -n "$SV" ] && ps --ppid "$SV" -o pid= | tr -d ' ' | tr '\n' ' ')
echo "papermill=$PM (pgid $(pg ${PM:-0})) kernel=$KN (pgid $(pg ${KN:-0})) solver=$SV sandboxes=$(echo $SB | wc -w) server=$SRV helpers=$*"
[ -n "$DRY" ] && { echo "dry run: nothing killed"; exit 0; }
for x in "$PM" "$KN"; do [ -n "$x" ] && { g=$(pg $x); kill -KILL -- -"$g" && echo "killed pgid $g ($x)"; }; done
[ -n "$SV" ] && kill -KILL -- -"$SV" 2>/dev/null; for p in $SB; do ps -o args= -p $p | grep -q "import ast, contextlib" && kill -KILL $p; done
echo "solver + sandboxes killed $(date -u +%T)"
if [ -n "$SRV" ] && ps -o args= -p "$SRV" | grep -q "vllm.entrypoints.cli.main serve"; then
  kill -TERM -- -"$SRV"; for i in $(seq 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$SRV" 2>/dev/null; echo "server group $SRV stopped $(date -u +%T)"
fi
for p in "$@"; do kill -- -"$(pg $p)" 2>/dev/null || kill "$p" 2>/dev/null; done
for i in $(seq 20); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 3; done
echo "GPU processes: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')] at $(date -u +%T)"
echo "left: $(pgrep -f 'VLLM::|solver_dsv41|ipykernel_launcher|papermill' | wc -l) processes"; free -g | sed -n 2p
echo "STOP_DONE $(date -u +%T)"
