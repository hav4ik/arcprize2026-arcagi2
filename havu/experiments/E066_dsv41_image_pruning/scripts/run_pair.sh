#!/bin/bash
# E066 local evals of one candidate on one box (image harness first; the text twin starts once the image run has 200
# finished attempts, or when it ends). Waits for a ready line in a box file, keeps an SSH tunnel to the box's port
# 18001, resumes a run once if attempts are missing.
# Usage: run_pair.sh <ssh_port> <ssh_host> <local_port> <sandbox_url> <ready_file> <ready_pat> <fail_pat> <img_tag> <txt_tag>
#   box 1 (E72-152): run_pair.sh 51290 23.16.123.156 18101 http://172.18.0.7:6000 \
#     /workspace/dsv4/results_chain_box1_main.log E72Q_READY 'FAILED|CTRL_WRONG' e72_img e72_txt
set -u
SP=$1; SH=$2; LP=$3; SANDBOX=$4; RF=$5; RP=$6; FP=$7; IMG=$8; TXT=$9
ROOT=/home/vu/Workspace/ARC-AGI-2
B=(ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $SP root@$SH)
EV=$ROOT/experiments/E066_dsv41_image_pruning/scripts/eval_arc_img.sh
RES=$ROOT/evals/results
export SANDBOX BASE=http://127.0.0.1:$LP/v1 MAXLEN=262144
log() { echo "$(date -u +%H:%M:%S) $*"; }
finished() {
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
  curl -s -m 10 http://127.0.0.1:$LP/v1/models | grep -q dsv41 && return 0
  for p in $(ps -eo pid=,args= | awk -v pat="-L $LP:127.0.0.1:18001 -p $SP root@$SH" '$2 == "ssh" && $3 == "-f" && index($0, pat) {print $1}'); do
    kill -TERM $p  # this script's own tunnel only, by PID
  done
  ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L $LP:127.0.0.1:18001 -p $SP root@$SH
  sleep 5; curl -s -m 10 http://127.0.0.1:$LP/v1/models | grep -q dsv41
}
run() {  # tag harness workers
  for pass in 1 2; do
    tunnel || { log "TUNNEL_FAILED before $1"; return 1; }
    log "start $1 (pass $pass)"
    (cd $ROOT && HARNESS=$2 WORKERS=$3 SALT=$1 bash $EV $1)
    n=$(finished $1); log "$1 finished attempts: $n/240"
    [ "$n" -ge 240 ] && return 0
  done
}
log "waiting for $RP in $SH:$RF"
while :; do
  s=$("${B[@]}" "grep -hE '$RP|$FP' $RF" 2>/dev/null | tail -1)
  case "$s" in *$RP*) break ;; esac
  echo "$s" | grep -qE "$FP" && { log "box: $s"; exit 1; }
  sleep 60
done
log "ready: $RP"
run $IMG img 192 &
P=$!
while kill -0 $P 2>/dev/null && [ "$(finished $IMG)" -lt 200 ]; do sleep 60; done
run $TXT txt 96 &
Q=$!
wait $P $Q
log PAIR_DONE $IMG $TXT
