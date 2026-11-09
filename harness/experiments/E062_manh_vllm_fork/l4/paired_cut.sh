#!/bin/bash
# E062 L4: paired score of the Manh replica vs B32 (candBf_x2t, old pod) and B32+R1 (pod #2's b32r1_x2t, read from
# its local mirror) cut at the same solver hour H (pod #2's cut_run.py + replica_report.py --baseline).
# Usage: paired_cut.sh H   -> l4/reports/manh_vs_*_<H>h.md
set -euo pipefail
H=$1; R=/home/vu/Workspace/ARC-AGI-2; E=$R/experiments/E062_manh_vllm_fork; T=$R/experiments/E055_kaggle_replica/toolkit
C=$E/l4/cuts; mkdir -p $C
for b in candBf_x2t:$T/runs/candBf_x2t b32r1_x2t:$E/l4pod2/runs/b32r1_x2t manh_x2t_w80:$E/l4/runs/manh_x2t_w80; do
  n=${b%%:*}; src=${b#*:}; rm -rf $C/${n}_$H; python3 -I $E/l4pod2/cut_run.py $src $H $C/${n}_$H > /dev/null
  $R/.venv/bin/python $T/replica_report.py $C/${n}_$H --no-traces --out $C/${n}_$H/report.md > /dev/null
done
for base in candBf_x2t b32r1_x2t; do
  $R/.venv/bin/python $T/replica_report.py $C/manh_x2t_w80_$H --no-traces --baseline $C/${base}_$H \
    --out $E/l4/reports/manh_vs_${base}_${H}h.md > /dev/null
  echo "== vs $base at $H h"; grep -E -i "paired|baseline|Kaggle score" $E/l4/reports/manh_vs_${base}_${H}h.md | head -6 | cut -c1-300
done
for n in candBf_x2t b32r1_x2t manh_x2t_w80; do echo "$n: $(grep -m1 'Kaggle score' $C/${n}_$H/report.md | cut -c1-120)"; done
