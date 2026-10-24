#!/bin/bash
# usage: run_eval.sh <A|B> <tag> <expert keep path|none> [wait-for-local-pid]
# waits for an earlier local eval process to exit, sets the runtime mask (+ Engram 48 GiB) on that server, runs the no-tool eval
SRV=$1; TAG=$2; KEEP=$3; WAITPID=${4:-}
if [ "$SRV" = A ]; then PORT=18000; CTRL=/dev/shm/e043/ctrlA.json; RUN=tp4A; else PORT=18001; CTRL=/dev/shm/e043/ctrlB.json; RUN=tp4B; fi
S="ssh -o BatchMode=yes -p 50293 root@147.185.60.21"
[ -n "$WAITPID" ] && while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
X=null; [ "$KEEP" != none ] && X="\\\"$KEEP\\\""
timeout 120 $S "cd /workspace/dsv4; echo \"{\\\"engram_keep\\\": \\\"/workspace/dsv4/data/engram/masks_nt2_calib/engram_top48gib.npz\\\", \\\"expert_keep\\\": $X, \\\"observe\\\": null}\" > $CTRL.tmp && mv $CTRL.tmp $CTRL; curl -s -m 120 localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"dsv41\",\"prompt\":[0,5,6,7],\"max_tokens\":1}' >/dev/null; sleep 5; grep 'E043\] ctrl' runs/$RUN/server.log | tail -1 | cut -c60-260" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
date -u
BASE=http://127.0.0.1:$PORT/v1 SALT=nt_$TAG bash /home/vu/Workspace/ARC-AGI-2/experiments/E043_dsv41_real_pruning/scripts/eval_notool.sh $TAG
date -u; echo EVAL-DONE $TAG
