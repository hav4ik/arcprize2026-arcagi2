#!/bin/bash
# E055: the next candidate on the same hidden-set-sized workload (eval x2), Kaggle deadlines: multi-test + budget 0.5,
# context limit, x44 + KV gate 0.7, retention interval 512, new code + auto-repair, util 0.938 (= Kaggle's KV pool)
N=cand_x2
/root/podrun.sh $N ARC_CHALLENGES=/root/data/eval_x2_challenges.json ARC_SOLUTIONS=/root/data/eval_x2_solutions.json \
  ARC_WORKERS=44 ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 ARC_CONTEXT_LIMIT=163840 ARC_GPU_UTIL=0.938 \
  ARC_MAX_NUM_SEQS=64 ARC_KV_ADMIT_MAX=0.7 ARC_KV_ADMIT_MIN_WORKERS=16 ARC_AUTO_REPAIR=1 ARC_CACHE_SALT=${N}_$(date -u +%H%M%S) \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512"]'
sleep 2
setsid nohup /root/memlog.sh /root/runs/$N/mem.log > /dev/null 2>&1 < /dev/null &
echo "memlog started"
