#!/bin/bash
# E043 no-tool round: upload its artifacts under notool/ of the private HF artifacts dataset (token in $HF_TOKEN).
set -u
cd /workspace/dsv4
R=chankhavu/dsv41-flash1017-e043-pruning-artifacts
U="tvenv/bin/python code/scripts/hf_upload.py --repo $R --type dataset"
S=hf_stage_nt; rm -rf $S; mkdir -p $S/engram $S/reap $S/results $S/split
cp -r data/engram/masks_nt2_calib $S/engram/masks
cp data/reap/stats_nt2.npz data/reap/answerblock_nt2.npz data/reap/compare_tool_notool2.json $S/reap/
cp -r data/reap/keep_nt2 $S/reap/keep
cp data/nll_nt2_dev.jsonl data/nll_nt2_dev_B.jsonl data/copy_probe_nt2.jsonl data/copy_probe_nt2_B.jsonl $S/results/
cp data/tok_nt2_calib/index.jsonl $S/split/calib_index.jsonl; cp data/tok_nt2_dev/index.jsonl $S/split/dev_index.jsonl
cat > $S/README.md <<'MD'
# E043 no-tool round artifacts

Statistics from the no-tool rollouts only (`notool_e044_*` configs of
livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036, snapshot 2026-10-23 21:40 UTC, 5,379 rollouts).
- `split/`: family-disjoint calibration (4,931) / dev (448) index files (`split_dev.py`); offsets refer to `tok/calib`, `tok/dev`.
- `engram/masks/`: top-frequency Engram row masks from the calibration census (48 GiB used for the evals).
- `engram_counts/`: raw per-row lookup counts on the 109M calibration tokens.
- `reap/`: REAP statistics on 20M replayed calibration tokens (`stats_nt2.npz`), answer-block statistics
  (`answerblock_nt2.npz`, generated `</think>`..EOS), keep sets `keep/keep_contrib_dec[_sp0_w<W>]_e<E>.npy`
  (bool [40, 384]); evaluated: E72 w10, E64 w20, E64 w10.
- `results/`: dev NLL (`nll_*`) and answer-writing probe (`copy_probe_*`) records.
- `obs/`: raw observer step files of the two replay halves.
MD
$U --path $S --in-repo notool --message "E043 no-tool round artifacts"
$U --path data/engram/counts_nt2_calib.npz --in-repo notool/engram_counts
$U --path data/tok_nt2_calib --in-repo notool/tok/calib --message "no-tool calibration token streams"
$U --path data/tok_nt2_dev --in-repo notool/tok/dev
$U --path data/tok_nt_eval --in-repo notool/tok/eval_traces
$U --path /dev/shm/e043/obs/nt2_a --in-repo notool/obs/half_a --message "no-tool observer half a"
$U --path /dev/shm/e043/obs/nt2_b --in-repo notool/obs/half_b --message "no-tool observer half b"
echo UPLOAD_DONE
