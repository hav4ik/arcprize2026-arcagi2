#!/bin/bash
# E066 box 1: FP4 Engram rows for the C1 export (CPU, nice 19, beside the C1 eval server), then export/c1_fp4 =
# the C1 export with those Engram files (hardlinks for everything else), for the FP4-vs-FP8 Engram dev NLL check.
set -eu
cd /workspace/dsv4
nice -n 19 venv/bin/python repo/experiments/E066_dsv41_image_pruning/scripts/fp4_engram_chunked.py --src export/c1_gptq \
  --out export/engram64_fp4 --threads 64 > logs/fp4_engram.log 2>&1
tail -1 logs/fp4_engram.log
rm -rf export/c1_fp4; mkdir -p export/c1_fp4
for f in export/c1_gptq/*; do
  b=$(basename $f)
  if [[ $b == model-engram-* ]]; then ln export/engram64_fp4/$b export/c1_fp4/$b
  elif [ -d $f ]; then cp -al $f export/c1_fp4/$b
  else ln $f export/c1_fp4/$b; fi
done
echo FP4_EXPORT_READY
