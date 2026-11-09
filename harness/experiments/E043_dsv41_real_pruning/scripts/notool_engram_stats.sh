#!/bin/bash
# E043 no-tool round: Engram statistics for the tech report (run on the box, CPU only).
# - report stats of the 109M-token calibration census (row-frequency histograms, mask sizes) vs dev and vs eval traces
# - data sufficiency: census on half of the calibration rollouts, masks, coverage on dev (vs the full census)
# - coverage of every no-tool mask (and the tool-round 64 GiB mask) on dev and on the no-tool eval traces
set -u
cd /workspace/dsv4/code/scripts
D=../../data; P=../../venv/bin/python; O=$D/engram/notool_stats; mkdir -p $O
$P engram_report_stats.py --counts $D/engram/counts_nt2_calib.npz --geo $D/engram_geometry.npz --masks-dir $D/engram/masks_nt2_calib \
  --eval-tok $D/tok_nt2_dev --out $O/engram_stats_notool_dev.json > $O/report_dev.log 2>&1
$P engram_report_stats.py --counts $D/engram/counts_nt2_calib.npz --geo $D/engram_geometry.npz --masks-dir $D/engram/masks_nt2_calib \
  --eval-tok $D/tok_nt_eval --out $O/engram_stats_notool_eval.json > $O/report_eval.log 2>&1
$P engram_census.py count --tok $D/tok_nt2_calib_h0 --workers 1 --out $D/engram/counts_nt2_half.npz > $O/half_count.log 2>&1
$P engram_census.py select --counts $D/engram/counts_nt2_half.npz --budget-gib 16 32 48 128 --out-dir $D/engram/masks_nt2_half > $O/half_select.log 2>&1
for T in tok_nt2_dev tok_nt_eval; do
  $P engram_census.py coverage --masks $D/engram/masks_nt2_calib/engram_top16gib.npz $D/engram/masks_nt2_calib/engram_top32gib.npz \
    $D/engram/masks_nt2_calib/engram_top48gib.npz $D/engram/masks_nt2_calib/engram_top64gib.npz $D/engram/masks_nt2_calib/engram_top128gib.npz \
    $D/engram/masks_nt2_half/engram_top16gib.npz $D/engram/masks_nt2_half/engram_top32gib.npz $D/engram/masks_nt2_half/engram_top48gib.npz \
    $D/engram/masks_nt2_half/engram_top128gib.npz $D/engram/masks/engram_top64gib.npz --tok $D/$T > $O/coverage_$T.log 2>&1
done
echo NOTOOL-ENGRAM-STATS-DONE
