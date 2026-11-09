#!/bin/bash
# usage: box_pipeline.sh <instance id> <host> <port> <current eval log> <current label> [<next label> <mtp> <kv> <attn> <extra env> <bundle>]
# Waits for the box's current eval to finish, copies its run dir locally, optionally runs one more eval (with an
# optional new solver bundle), copies that too, then destroys the instance.
ID=$1; H=$2; P=$3; LOG=$4; CUR=$5; NEXT=$6; MTP=$7; KV=$8; ATTN=$9; XENV=${10}; BUNDLE=${11}
cd /home/vu/Workspace/ARC-AGI-2
J=/home/vu/.claude/jobs/821d4fe1/tmp
OUT=experiments/E025_l4_quant_formats/e6; mkdir -p $OUT
S="ssh -o BatchMode=yes -o ConnectTimeout=20 -p $P root@$H"
wait_done() {  # label log
  until timeout 40 $S "grep -qE '$1 (EVAL_DONE|SERVER_FAILED)' /root/arc/$2" 2>/dev/null; do sleep 120; done
  timeout 40 $S "grep -E '$1 (RESULT|SERVER_FAILED|SERVER_READY)' /root/arc/$2" 2>/dev/null | grep -v "Welcome\|Have fun\|AI agents"
}
fetch() {  # label
  timeout 600 $S "cd /root/arc/runs && tar czf /root/run_$1.tgz $1" 2>/dev/null
  scp -q -P $P -o BatchMode=yes root@$H:/root/run_$1.tgz $OUT/ && tar xzf $OUT/run_$1.tgz -C $OUT && rm $OUT/run_$1.tgz && \
    echo "$(date -u +%H:%M) fetched $1 -> $OUT/$1 ($(du -sh $OUT/$1 | cut -f1))"
}
echo "$(date -u +%H:%M) waiting for $CUR on $ID"
wait_done $CUR $LOG
fetch $CUR
if [ -n "$NEXT" ]; then
  scp -q -P $P -o BatchMode=yes $J/box/eval_6000.sh root@$H:/root/arc/eval_6000.sh
  if [ -n "$BUNDLE" ]; then
    scp -q -P $P -o BatchMode=yes $BUNDLE root@$H:/root/arc_bundle_next.tgz
    timeout 60 $S "cd /root/arc && rm -rf work/arc_solver && tar xzf /root/arc_bundle_next.tgz -C /root/arc work/arc_solver && grep -c ARC_COMPACT_AT work/arc_solver/solver.py" 2>/dev/null | tail -1
  fi
  timeout 40 $S "cd /root/arc && (EXTRA_ENV='$XENV' setsid nohup ./eval_6000.sh $NEXT $MTP $KV $ATTN > eval_$NEXT.log 2>&1 < /dev/null &); echo started $NEXT" 2>/dev/null | grep -v "Welcome\|Have fun\|AI agents"
  wait_done $NEXT eval_$NEXT.log
  fetch $NEXT
fi
K="$(sed -n 's/^vast_ai_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
.venv/bin/vastai destroy instance $ID -y --api-key "$K" 2>&1 | tail -1
echo "$(date -u +%H:%M) $ID destroyed"
