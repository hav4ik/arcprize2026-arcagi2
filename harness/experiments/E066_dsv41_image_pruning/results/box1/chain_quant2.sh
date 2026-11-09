#!/bin/bash
# E066: resume of chain_quant.sh after the --model-dir fix (capture of both candidates already done).
set -u
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
bash $S/gptq_candidate.sh quant c1 cand/c1_keep.npy || { echo QUANT_FAILED c1; exit 1; }
bash $S/gptq_candidate.sh quant c2 cand/c2_keep.npy || { echo QUANT_FAILED c2; exit 1; }
echo QUANT_DONE
bash $S/serve_gptq.sh c1q cand/c1_keep.npy gptq/c1
for i in $(seq 1 80); do curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo SERVE_FAILED; tail -20 runs/c1q/server.log; exit 1; }
echo C1Q_READY
bash $S/nll_masks_mm.sh results/q_screen.jsonl c1q:cand/c1_keep.npy
echo C1Q_NLL_DONE
