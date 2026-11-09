#!/bin/bash
# After the profile on the L4 box: real tool-traffic probes on vLLM 0.31 (NCCL default transport).
# MTP2 + TurboQuant is broken in vLLM 0.31 (degenerate text), so MTP2 runs with fp8 KV at concurrency 20.
cd /root/arc
trap - INT
until grep -q PROF_DONE prof_run.log 2>/dev/null; do sleep 60; done
unset NCCL_P2P_LEVEL
M=Qwen/Qwen3.8-27B-FP8
VENV=/root/vllmnew ATTN=TRITON_ATTN MTP=2 ./probe2.sh p31_mtp2_c20 $M 20 --language-model-only
VENV=/root/vllmnew ATTN=TRITON_ATTN MTP=0 ./probe2.sh p31_base_c24 $M 24 --language-model-only
echo L4_PROBES_DONE
