#!/bin/bash
# E066 local eval orchestrator for the two final candidates (image harness first, user 2026-10-29):
#   e72_img (per-layer 72-152, runtime-mask server) -> c1x_img (candidate 1 from its export; c1x_txt starts once
#   c1x_img has 200 finished attempts) -> e72_txt. Waits for chain_box1_main.sh's E72Q_READY, switches the box 1
#   server with serve_eval.sh, checks the tunnel before each run. Sandbox: arc-sandbox-e066c.
set -u
ROOT=/home/vu/Workspace/ARC-AGI-2
B=(ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -p 51290 root@23.16.123.156)
EV=$ROOT/experiments/E066_dsv41_image_pruning/scripts/eval_arc_img.sh
RES=$ROOT/evals/results
export SANDBOX=http://172.18.0.7:6000 BASE=http://127.0.0.1:18101/v1 MAXLEN=262144
log() { echo "$(date -u +%H:%M:%S) $*"; }
finished() {  # attempts (task, attempt) with a record and no error
  python3 - "$RES/tools_persistent_grid_lean2sub_dsv41_low_e066_$1.jsonl" <<'PY'
import json, sys, os
p = sys.argv[1]; s = set()
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: s.add((r["task_id"], r["attempt"]))
print(len(s))
PY
}
tunnel() {
  curl -s -m 10 http://127.0.0.1:18101/v1/models | grep -q dsv41 && return 0
  for p in $(ps -eo pid=,args= | awk '$2 == "ssh" && $3 == "-f" && /-L 18101:127.0.0.1:18001 -p 51290 root@23.16.123.156$/ {print $1}'); do
    kill -TERM $p  # this script's own tunnel only, by PID
  done
  ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 18101:127.0.0.1:18001 -p 51290 root@23.16.123.156
  sleep 5; curl -s -m 10 http://127.0.0.1:18101/v1/models | grep -q dsv41
}
run() {  # tag harness workers: one eval, resumed once if attempts are missing
  for pass in 1 2; do
    tunnel || { log "TUNNEL_FAILED before $1"; return 1; }
    log "start $1 (pass $pass)"
    (cd $ROOT && HARNESS=$2 WORKERS=$3 SALT=$1 bash $EV $1)
    n=$(finished $1); log "$1 finished attempts: $n/240"
    [ "$n" -ge 240 ] && return 0
  done
}
switch() {
  out=$("${B[@]}" "bash /workspace/dsv4/repo/experiments/E066_dsv41_image_pruning/scripts/serve_eval.sh $1" 2>&1 | grep -E "EVAL_SERVER_READY|SERVE_FAILED|CTRL_WRONG" | tail -1)
  log "$out"; echo "$out" | grep -q EVAL_SERVER_READY
}
log "waiting for E72Q_READY"
while :; do
  s=$("${B[@]}" "grep -hE 'E72Q_READY|DUMP_FAILED|QUANT_FAILED|SERVE_FAILED|CTRL_WRONG' /workspace/dsv4/results_chain_box1_main.log" 2>/dev/null | tail -1)
  case "$s" in *E72Q_READY*) break ;; *FAIL*|*WRONG*) log "box chain: $s"; exit 1 ;; esac
  sleep 60
done
log "E72Q_READY"
run e72_img img 192
switch c1x || exit 1
run c1x_img img 192 &
P=$!
while kill -0 $P 2>/dev/null && [ "$(finished c1x_img)" -lt 200 ]; do sleep 60; done
run c1x_txt txt 96 &
Q=$!
wait $P $Q
switch e72q || exit 1
run e72_txt txt 192
log ALL_EVALS_DONE
