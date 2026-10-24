#!/bin/bash
# E043: copy every small result off the 8x box into results/box/ (re-run after each stage; the box can disappear).
# Takes data/*.json(l) results, keep sets, split indexes, probe prompts, calibration manifests, GPTQ per-layer error
# stats (from /dev/shm and the exports), all logs and server run dirs (throughput source). Large tensors live on HF.
# Usage: pull_box_results.sh   (then server_log_series.py / lowbit_summary.py read from results/)
set -euo pipefail
cd "$(dirname "$0")/.."
B=results/box
mkdir -p $B
timeout 600 ssh -o BatchMode=yes -p 50293 root@147.185.60.21 'cd /workspace/dsv4; tar czf - --ignore-failed-read \
  data/*.jsonl data/*.json data/tool_dev_probe.parquet data/reap/*.json data/reap/keep_* data/reap/protect data/engram/*.json \
  data/engram/masks*/*.json data/tok_*/index.jsonl data/calx_*/manifest.json data/unsloth/README.md logs runs \
  $(ls -d /dev/shm/e043/gptq_*/ export/*/gptq/ 2>/dev/null | sed "s#\$#*.json#") 2>/dev/null' 2>/dev/null \
  | tar xzf - -C $B
cp $B/data/nll_lowbit_dev.jsonl results/lowbit_nll_dev.jsonl
cp $B/data/copy_probe_lowbit.jsonl results/lowbit_copy_probe.jsonl
for f in nll_tool_dev.jsonl gen_probe_tool_dev.jsonl; do [ -f $B/data/$f ] && cp $B/data/$f results/tool_$f || true; done
du -sh $B; find $B -type f | wc -l
