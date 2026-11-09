#!/bin/bash
# E066 C1 image model, Kaggle arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pyTorch/default/1 (137 files, 132.06 GB):
# the 121 non-Engram files, per file, version-pinned, in two sequential streams over size-balanced halves (as pod 4's
# dl_model.sh), each with its own kagglehub cache outside the model dir. Token on stdin, only in these processes' env.
set -uo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
S=/root/sftv01/scripts; L=/root/sftv01/logs; D=/root/models/c1_img; mkdir -p $D $L
H=arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pyTorch/default/1
for x in a b; do
  KAGGLEHUB_CACHE=/root/models/.khcache_c1_$x setsid nohup /root/kgl/bin/python -I $S/dl_kaggle_files.py model $H $S/c1_files_$x.tsv $D \
    > $L/dl_c1_$x.log 2>&1 < /dev/null &
  echo "stream $x pid $!"
  sleep 5
done
unset KAGGLE_API_TOKEN
