#!/bin/bash
# E066 follow-up to E068: candidate 1 on the image harness with the fixed (causal) image-span attention, run exactly
# as c1x_img was (run_evals_local.sh: eval_arc_img.sh, HARNESS=img, 192 workers, salt = tag), against node 5's c1xc
# server (fix_c1_chain.sh prints C1XC_READY). Sandbox: arc-sandbox-e066c (local, sandboxed tool code).
set -u
ROOT=/home/vu/Workspace/ARC-AGI-2
B=(ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -p 42304 root@23.16.123.156)
EV=$ROOT/experiments/E066_dsv41_image_pruning/scripts/eval_arc_img.sh
OUT=$ROOT/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e066_c1xc_img.jsonl
log() { echo "$(date -u +%H:%M:%S) $*"; }
until "${B[@]}" 'grep -hE "C1XC_READY|DECISION_NEEDED|FAILED|DIFFERS|NOT_APPLIED" /workspace/dsv4/logs/fix_c1_chain.log' 2>/dev/null \
    | grep -q .; do sleep 60; done
st=$("${B[@]}" 'grep -hE "C1XC_READY|DECISION_NEEDED|FAILED|DIFFERS|NOT_APPLIED" /workspace/dsv4/logs/fix_c1_chain.log' 2>/dev/null | tail -1)
echo "$st" | grep -q C1XC_READY || { log "box chain stopped: $st"; exit 1; }
tunnel() {
  curl -s -m 10 http://127.0.0.1:18105/v1/models | grep -q dsv41 && return 0
  for p in $(ps -eo pid=,args= | awk '$2 == "ssh" && $3 == "-f" && /-L 18105:127.0.0.1:18001 -p 42304 root@23.16.123.156$/ {print $1}'); do
    kill -TERM $p  # this script's own tunnel only, by PID
  done
  ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 18105:127.0.0.1:18001 -p 42304 root@23.16.123.156
  sleep 5; curl -s -m 10 http://127.0.0.1:18105/v1/models | grep -q dsv41
}
SB=http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' arc-sandbox-e066c):6000
curl -s -m 5 $SB/health | grep -q healthy || { log "SANDBOX_DOWN $SB"; exit 1; }
finished() { python3 -c "
import json, os
p = '$OUT'; s = set()
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l)
        if 'error' not in r: s.add((r['task_id'], r['attempt']))
print(len(s))"; }
for pass in 1 2 3; do  # the harness resumes; a dropped tunnel only fails the in-flight calls
  tunnel || { log "TUNNEL_FAILED"; sleep 60; continue; }
  log "pass $pass: $(finished)/240 attempts so far"
  (cd $ROOT && HARNESS=img WORKERS=192 SALT=c1xc_img SANDBOX=$SB BASE=http://127.0.0.1:18105/v1 bash $EV c1xc_img)
  [ "$(finished)" -ge 240 ] && break
done
log "C1XC_IMG_DONE $(finished)/240"
