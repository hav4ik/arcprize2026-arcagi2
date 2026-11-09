#!/bin/bash
# Run the exact candidate notebook end to end with the reliability kit's runner (nbclient, clean env, kernel = /root/kenv),
# plus the kit's sampler (60 s / 5 s), a following watcher, and a /metrics scraper (port 18084, 60 s, across restarts).
# Before the run: fresh /tmp/dsv41, /tmp/arc_vision, /dev/shm/wheels and a pristine kernel venv (/root/kenv_fresh), as
# Kaggle starts from its image. Inputs mapped by symlinks (--prune-input: the model glob finds one dir, as on Kaggle).
#   commit: competition dir = the local data/ files (Kaggle's public test file); rerun: test file = evalx2t_s2026.
# Usage: launch_run.sh <name> <commit|rerun> [--dry-run]
set -euo pipefail
N=$1; MODE=$2; DRY=${3:-}
R=/root/runs/$N; W=/root/kwork_$N; K=/root/kit; NB=/root/p4/nb/dsv41-tools-submit-b32m-fix.ipynb
SRC=$(ls -d /root/kin/kh/datasets/nguyennguyen599/arc2-2026-src/versions/* | sort -V | tail -1)
KIT=/root/kin/kh/datasets/chankhavu/dsv41-tools-kit/versions/5
MAP=(--map /root/models/e128=models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1
     --map /root/kin/vllm-wheels-v11=datasets/nguyennguyen599/vllm-wheels
     --map /root/kin/cu13=datasets/nguyennguyen599/cu13-install
     --map $SRC=datasets/nguyennguyen599/arc2-2026-src
     --map $KIT=datasets/chankhavu/dsv41-tools-kit
     --map /root/comp_$MODE=competitions/arc-prize-2026-arc-agi-2)
# pod-only accommodation (as pod 2): the ~20 GB uv cache is deleted when cell 12 starts (nothing after cell 11 uses uv)
ARGS=($NB --out $R/nb --mode $MODE --clean-env --venv /root/kenv "${MAP[@]}" --prune-input --working $W
      --on-cell-start "12:rm -rf /tmp/dsv41/cache/uv; df -h / | tail -1")
if [ "$DRY" = --dry-run ]; then /root/kenv/bin/python -I $K/run_notebook.py "${ARGS[@]}" --dry-run; exit 0; fi
[ -e $R/nb.timeline ] && { echo "$R/nb.timeline exists"; exit 1; }
mkdir -p $R
rm -rf /tmp/dsv41 /tmp/arc_vision /dev/shm/wheels
rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv
sha256sum $NB > $R/notebook.sha256; echo "src $SRC kit $KIT" > $R/inputs.txt
echo "test file: $(sha256sum /root/comp_$MODE/arc-agi_test_challenges.json | cut -c1-16) ($MODE)" >> $R/inputs.txt
setsid nohup /usr/bin/python3 $K/sampler.py --out $R/mon.jsonl --working $W --timeline $R/nb.timeline \
  --fault-log $R/fault.jsonl --stop-on-notebook-end > $R/sampler.out 2>&1 < /dev/null & echo $! > $R/sampler.pid
setsid nohup bash /root/p4/scripts/scrape_loop.sh $R/metrics_scrape.txt $R/nb.timeline > /dev/null 2>&1 < /dev/null & echo $! > $R/scrape.pid
setsid nohup /root/kenv/bin/python -I $K/run_notebook.py "${ARGS[@]}" > $R/runner.out 2>&1 < /dev/null & echo $! > $R/runner.pid
sleep 3
setsid nohup /usr/bin/python3 $K/watcher.py --samples $R/mon.jsonl --follow > $R/watcher.out 2>&1 < /dev/null & echo $! > $R/watcher.pid
echo "started $N ($MODE) at $(date -u +%T): runner $(cat $R/runner.pid), sampler $(cat $R/sampler.pid), watcher $(cat $R/watcher.pid), scrape $(cat $R/scrape.pid)"
cat $R/inputs.txt; head -3 $R/nb.timeline 2>/dev/null
