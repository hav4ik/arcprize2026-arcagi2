#!/bin/bash
# The E128 Kaggle model chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/PyTorch/default/1 (136 files, 131.09 GB), per file,
# version-pinned, in two sequential streams over size-balanced halves (e128_files_a/b.tsv), each with its own kagglehub
# cache; then sha256 of every file against the manifest embedded in the candidate's cell 1b (verify_e128.sh).
# Token on stdin (never on a command line). Logs: /root/p4/logs/dl_e128_{a,b}.log, verify_e128.log
set -uo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
S=/root/p4/scripts; L=/root/p4/logs; D=/root/models/e128; mkdir -p $D $L
H=chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1
for x in a b; do
  KAGGLEHUB_CACHE=$D/.khcache_$x setsid nohup /root/kgl/bin/python -I $S/dl_kaggle_files.py model $H $S/e128_files_$x.tsv $D \
    > $L/dl_e128_$x.log 2>&1 < /dev/null &
  echo "stream $x pid $!"
  sleep 5
done
unset KAGGLE_API_TOKEN
