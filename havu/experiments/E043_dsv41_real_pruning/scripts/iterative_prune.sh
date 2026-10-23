#!/bin/bash
# E043: iterative expert pruning with re-observation (suggested by E042): 384 -> 192 -> 128 -> 96 -> 72 -> 64 -> 48.
# Stage s: replay TOKENS fresh calibration tokens under the stage's keep mask (+ frozen Engram mask) with the
# observer on, aggregate, and keep the top experts of the next size by RULE among the currently kept ones.
# Usage: iterative_prune.sh [RULE=contrib] [DEC=1]   env: TOKENS (1500000), SERVER, CTRL
set -eu
cd /workspace/dsv4
RULE=${1:-contrib}; DEC=${2:-1}
DECF=""; [ "$DEC" = 1 ] && DECF="--decoder-gen-only"
TAG=${RULE}$([ "$DEC" = 1 ] && echo _dec)
SERVER=${SERVER:-http://127.0.0.1:18001}; CTRL=${CTRL:-/dev/shm/e043/ctrlB.json}
ENG=/workspace/dsv4/data/engram/masks/engram_top64gib.npz
OUT=data/reap/iter_$TAG; mkdir -p $OUT
TOK=$(ls -d data/tok/shard*)
# Stage 0: one-shot stats on 384 experts -> keep 192.
tvenv/bin/python code/scripts/reap_aggregate.py select --stats data/reap/stats_all.npz --score $RULE $DECF \
  --experts 192 --out-dir $OUT > /dev/null
mv $OUT/keep_${TAG}_e192.npy $OUT/keep_e192.npy
prev=192; seed=10
for next in 128 96 72 64 48; do
  obs=/dev/shm/e043/obs/iter_${TAG}_e$prev
  tvenv/bin/python code/scripts/replay_reap.py --tok $TOK --tokens ${TOKENS:-1500000} --seed $seed --server $SERVER \
    --ctrl $CTRL --obs-dir $obs --engram-keep $ENG --expert-keep $(realpath $OUT/keep_e$prev.npy) > $OUT/replay_e$prev.log 2>&1
  tvenv/bin/python code/scripts/reap_aggregate.py aggregate --obs $obs --out $OUT/stats_e$prev.npz > $OUT/agg_e$prev.log
  tvenv/bin/python code/scripts/reap_aggregate.py select --stats $OUT/stats_e$prev.npz --score $RULE $DECF \
    --experts $next --within $OUT/keep_e$prev.npy --out-dir $OUT > $OUT/sel_e$next.log
  mv $OUT/keep_${TAG}_e$next.npy $OUT/keep_e$next.npy
  python3 - <<EOF
import numpy as np
a = np.load("$OUT/keep_e$prev.npy"); b = np.load("$OUT/keep_e$next.npy")
assert not (b & ~a).any(), "next keep set must be a subset of the current one"
print("stage e$prev -> e$next ok", b.sum(1).min(), b.sum(1).max())
EOF
  prev=$next; seed=$((seed + 1))
done
echo ITER_DONE
