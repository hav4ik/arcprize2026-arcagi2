#!/bin/bash
# E043: upload the remaining box-only artifacts to the private HF artifacts dataset (token in $HF_TOKEN).
set -u
cd /workspace/dsv4
R=chankhavu/dsv41-flash1017-e043-pruning-artifacts
U="tvenv/bin/python code/scripts/hf_upload.py --repo $R --type dataset"
S=hf_stage2; rm -rf $S; mkdir -p $S/reap $S/results $S/code
cp -r data/reap/protect data/reap/iter_contrib_dec $S/reap/
cp data/reap/special_all.npz data/reap/toolblock_all.npz $S/reap/
cp data/*.jsonl data/engram/unseen_eval6.json $S/results/
cp -r code/scripts code/patches $S/code/
$U --path $S --message "E043 tool-block keep sets, iterative stages, NLL/probe results, final code"
for d in /dev/shm/e043/obs/iter_contrib_dec_e*; do $U --path $d --in-repo obs/$(basename $d) --message "observer $(basename $d)"; done
echo FINAL_UPLOAD_DONE
