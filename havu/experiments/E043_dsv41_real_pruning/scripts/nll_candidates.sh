#!/bin/bash
# E043: paired teacher-forced NLL of candidate masks on the held-out eval traces (server B, runtime masks).
# Usage: nll_candidates.sh <out.jsonl> <tag>:<engram_keep|none>:<expert_keep|none> [...]
#   env: SERVER (http://127.0.0.1:18001), CTRL (/dev/shm/e043/ctrlB.json), NSEQ (32), MAXLEN (40000), SEED (0)
set -eu
OUT=$1; shift
SERVER=${SERVER:-http://127.0.0.1:18001}
CTRL=${CTRL:-/dev/shm/e043/ctrlB.json}
cd /workspace/dsv4
for spec in "$@"; do
  IFS=: read -r tag eng exp <<< "$spec"
  e=null; [ "$eng" != none ] && e="\"$(realpath $eng)\""
  x=null; [ "$exp" != none ] && x="\"$(realpath $exp)\""
  echo "{\"engram_keep\": $e, \"expert_keep\": $x, \"observe\": null}" > $CTRL.tmp && mv $CTRL.tmp $CTRL
  curl -s -m 120 $SERVER/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv41","prompt":[0,5,6,7],"max_tokens":1}' > /dev/null
  tvenv/bin/python code/scripts/nll_eval.py --tok ${TOK:-data/tok_eval} ${PERTASK:-} --n-seq ${NSEQ:-32} --seed ${SEED:-0} \
    --max-len ${MAXLEN:-40000} --server $SERVER --tag "$tag" --out "$OUT"
done
echo "{\"engram_keep\": null, \"expert_keep\": null, \"observe\": null}" > $CTRL.tmp && mv $CTRL.tmp $CTRL
