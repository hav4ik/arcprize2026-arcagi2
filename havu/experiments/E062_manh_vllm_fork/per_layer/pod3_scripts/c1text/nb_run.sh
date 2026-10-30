#!/bin/bash
# c1text: the notebook end to end with papermill (cwd /kaggle/working). Usage: nb_run.sh <commit|rerun>
M=$1; D=/root/e066/c1text; cd /kaggle/working
export VIRTUAL_ENV=/root/kenv PATH=/root/kenv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
[ "$M" = rerun ] && export KAGGLE_IS_COMPETITION_RERUN=1
echo "nb start $M $(date -u +%T)" >> $D/run_$M.log
/root/kgl/bin/papermill --kernel py313b32m --log-output --request-save-on-cell-execute --cwd /kaggle/working \
  $D/nb_$M.ipynb $D/nb_${M}_out.ipynb >> $D/run_$M.log 2>&1
echo "nb end rc=$? $(date -u +%T)" >> $D/run_$M.log
