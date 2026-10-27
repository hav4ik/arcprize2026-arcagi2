#!/bin/bash
# E055: exact V5 replica on a hidden-set-sized workload (eval x2: 240 tasks / 344 pairs), V5's own Kaggle kit
# (solver 62144e4a, harness 1ad5ee24), V5 settings (per pair, x36, no gate, Humming, cudagraph cap 64), Kaggle deadlines
N=v5rep_x2
/root/podrun.sh $N ARC_KIT=/root/kit_v5 ARC_CHALLENGES=/root/data/eval_x2_challenges.json ARC_SOLUTIONS=/root/data/eval_x2_solutions.json \
  ARC_WORKERS=36 ARC_GPU_UTIL=0.938 ARC_MAX_NUM_SEQS=64 ARC_CACHE_SALT=${N}_$(date -u +%H%M%S)
sleep 2
setsid nohup /root/memlog.sh /root/runs/$N/mem.log > /dev/null 2>&1 < /dev/null &
echo "memlog started"
