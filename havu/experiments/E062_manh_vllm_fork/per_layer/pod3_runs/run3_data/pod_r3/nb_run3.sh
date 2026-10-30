#!/bin/bash
# run 3: the notebook end to end in rerun mode (papermill, cwd /kaggle/working), as run 2's nb_run.sh
cd /kaggle/working
export KAGGLE_IS_COMPETITION_RERUN=1 VIRTUAL_ENV=/root/b32m/venv PATH=/root/b32m/venv/bin:/usr/local/bin:/usr/bin:/bin
echo "nb start $(date -u +%T)" >> /root/e066/r3/run.log
/root/kgl/bin/papermill --kernel py313b32m --log-output --request-save-on-cell-execute --cwd /kaggle/working \
  /root/e066/r3/nb_run3.ipynb /root/e066/r3/nb_run3_out.ipynb >> /root/e066/r3/run.log 2>&1
echo "nb end rc=$? $(date -u +%T)" >> /root/e066/r3/run.log
