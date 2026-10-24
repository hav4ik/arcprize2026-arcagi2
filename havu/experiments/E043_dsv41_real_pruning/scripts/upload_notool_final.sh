#!/bin/bash
# E043 no-tool round, final sweep before the box is released: everything not yet on the private HF artifacts
# dataset (token in $HF_TOKEN). Waits for notool_engram_stats.sh to finish first.
set -u
cd /workspace/dsv4
while ! grep -q NOTOOL-ENGRAM-STATS-DONE logs/notool_engram_stats.log 2>/dev/null; do sleep 30; done
R=chankhavu/dsv41-flash1017-e043-pruning-artifacts
U="tvenv/bin/python code/scripts/hf_upload.py --repo $R --type dataset"
S=hf_stage_final; rm -rf $S; mkdir -p $S/rollouts_snapshot $S/eval_traces $S/pilot/engram $S/pilot/reap $S/engram_stats $S/reap $S/results
cp data/rollouts_ds2/rollouts/*.parquet $S/rollouts_snapshot/
(cd data/rollouts_ds2/rollouts && sha256sum *.parquet) > $S/rollouts_snapshot/SHA256SUMS
cp data/notool_eval120.parquet $S/eval_traces/
cp -r data/engram/masks_nt data/engram/masks_nt_calib $S/pilot/engram/
cp data/reap/stats_nt.npz data/reap/answerblock_nt.npz data/reap/compare_tool_notool.json $S/pilot/reap/; cp -r data/reap/keep_nt $S/pilot/reap/
cp data/nll_nt_dev.jsonl $S/pilot/
cp -r data/engram/notool_stats/* $S/engram_stats/; cp -r data/engram/masks_nt2_half $S/engram_stats/
cp data/engram/unseen_nt_eval.json data/engram/unseen_kinds_nt_eval.txt $S/engram_stats/
cp data/reap/expert_stats_notool.json data/reap/compare_tool_notool2.json $S/reap/; cp -r data/reap/keep_nt2 $S/reap/keep
cp data/nll_nt2_dev*.jsonl data/copy_probe_nt2*.jsonl $S/results/
cat > $S/README_final.md <<'MD'
# notool/ final additions (2026-10-24)
- `rollouts_snapshot/`: the exact `notool_e044_*` parquet files used (SHA256SUMS; the source dataset keeps growing).
- `eval_traces/notool_eval120.parquet`: full-model no-tool traces on the 120 eval tasks (E044 DeepInfra + E035 API), metrics only.
- `pilot/`: first-snapshot pilot (1,778 calibration rollouts): Engram masks, REAP stats on 10M tokens, keep sets, dev NLL.
- `engram_stats/`: census report stats vs dev and vs eval traces, half-data census masks and coverage logs
  (data sufficiency), unseen-row analysis on the eval traces.
- `reap/expert_stats_notool.json`: activity, coverage curves, per-layer kept shares, overlaps, tool-vs-no-tool comparison.
- `results/`: dev NLL (incl. the physical-export parity check) and answer-writing probe records.
MD
$U --path $S --in-repo notool --message "E043 no-tool final sweep"
$U --path data/engram/counts_nt2_half.npz --in-repo notool/engram_counts
$U --path data/engram/counts_nt.npz --in-repo notool/pilot/engram_counts
$U --path data/engram/counts_nt_calib.npz --in-repo notool/pilot/engram_counts
$U --path data/server_series.json --in-repo results; $U --path data/throughput_series.json --in-repo results
$U --path logs --in-repo box_logs --message "box logs (final)"
B=hf_stage_runs; rm -rf $B; mkdir -p $B
for r in runs/*; do mkdir -p $B/$(basename $r); cp $r/server.log $r/cmd.txt $r/env.txt $B/$(basename $r)/ 2>/dev/null; done
$U --path $B --in-repo box_runs --message "server logs (final)"
$U --path /dev/shm/e043/obs/nt_main2 --in-repo notool/pilot/obs --message "no-tool pilot observer records"
echo FINAL-UPLOAD-DONE
