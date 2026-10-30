#!/bin/bash
# run the notebook end to end in rerun mode with papermill (cell outputs streamed to run.log), cwd /kaggle/working
cd /kaggle/working
export KAGGLE_IS_COMPETITION_RERUN=1 VIRTUAL_ENV=/root/b32m/venv PATH=/root/b32m/venv/bin:/usr/local/bin:/usr/bin:/bin
echo "nb start $(date -u +%T)" >> /root/e066/nbrun/run.log
/root/kgl/bin/papermill --kernel py313b32m --log-output --request-save-on-cell-execute --cwd /kaggle/working \
  /root/e066/nbrun/nb_c1.ipynb /root/e066/nbrun/nb_c1_out.ipynb >> /root/e066/nbrun/run.log 2>&1
echo "nb end rc=$? $(date -u +%T)" >> /root/e066/nbrun/run.log
