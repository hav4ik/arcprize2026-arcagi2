#!/bin/bash
# E066 box 1: once the E72-152 HF upload is verified, its path/size/sha256 manifest (for dsv4-infer's mounted-file check).
cd /workspace/dsv4
until grep -qE "UPLOAD_(DONE|FAILED) e72c152|EXPORT_FAILED|CHECK_FAILED" results_chain_box1_main.log logs/upload_e72c152.log 2>/dev/null; do sleep 30; done
grep -q "UPLOAD_DONE e72c152" logs/upload_e72c152.log || { echo "NO_MANIFEST: upload not verified"; exit 1; }
nice -n 10 venv/bin/python repo/experiments/E066_dsv41_image_pruning/scripts/sha_manifest.py export/e72c152_gptq \
  results/e72c152_gptq_sha256_manifest.json && echo MANIFEST_DONE
