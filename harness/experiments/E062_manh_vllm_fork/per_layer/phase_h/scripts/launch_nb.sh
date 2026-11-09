#!/bin/bash
# Run the exact built notebook end to end on the pod, as Kaggle's "Run All" would (phase commit: no
# KAGGLE_IS_COMPETITION_RERUN; phase rerun: KAGGLE_IS_COMPETITION_RERUN=1 on the 240-task x2t workload).
# Fresh /tmp/dsv41, /tmp/arc_vision, /dev/shm/wheels, /root/kenv and /kaggle/working; a clean environment (env -i) with the
# Kaggle-like kernel venv /root/kenv first on PATH, so Manh's `uv pip install` cells install into it.
# Usage: launch_nb.sh <commit|rerun> <name> [notebook]
set -euo pipefail
PH=$1; N=$2; R=/root/nbruns/$N; NB=${3:-/root/e062pl/dsv41-tools-submit-img-manh.ipynb}
mkdir -p $R; [ -e $R/nb.timeline ] && { echo "$R/nb.timeline exists"; exit 1; }
rm -rf /tmp/dsv41 /tmp/arc_vision /dev/shm/wheels
# the kernel venv starts as a fresh copy (the install cells add ~20 GB to it): /root/kenv_fresh = kenv_fresh_freeze.txt
rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv
bash /root/e062pl/make_kaggle_tree.sh $PH /root/kwork_$N > $R/tree.txt 2>&1
sha256sum $NB > $R/notebook.sha256
EXTRA=(); [ "$PH" = rerun ] && EXTRA=(KAGGLE_IS_COMPETITION_RERUN=1)
setsid nohup bash /root/memlog.sh $R/mem.log > /dev/null 2>&1 < /dev/null &
echo $! > $R/memlog.pid
cd /root/kwork_$N
setsid nohup env -i HOME=/root LANG=C.UTF-8 TERM=dumb VIRTUAL_ENV=/root/kenv \
  PATH=/root/kenv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "${EXTRA[@]}" \
  /root/kenv/bin/python -I /root/e062pl/run_nb.py $NB $R/nb > $R/run_nb.log 2>&1 < /dev/null &
echo $! > $R/run_nb.pid
echo "started $N ($PH) pid $(cat $R/run_nb.pid) at $(date -u +%T)"; cat $R/tree.txt
