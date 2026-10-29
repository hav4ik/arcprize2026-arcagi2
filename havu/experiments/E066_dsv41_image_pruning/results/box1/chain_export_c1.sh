#!/bin/bash
# E066 box 1: tool Engram from HF, then the candidate 1 export (CPU only, beside the running C1 eval server) and its
# content check.
set -u
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
bash $S/export_candidate.sh engram || { echo ENGRAM_FAILED; exit 1; }
bash $S/export_candidate.sh export c1 cand/c1_keep.npy gptq/c1 "E128 2-bit GPTQ, tool use with images (E066 candidate 1)" \
  "tool + image keep set: E065 image and text harness rollouts, class weights prompt/gen/tool-call/image 1/1/10/1 (layers 0-19), 0/1/10/1 (20-39)" \
  || { echo EXPORT_FAILED; exit 1; }
bash $S/export_candidate.sh check c1 cand/c1_keep.npy gptq/c1 || { echo CHECK_FAILED; exit 1; }
echo C1_EXPORT_ALL_DONE
