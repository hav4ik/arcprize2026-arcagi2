#!/bin/bash
# E020 launch queue: K2+K3 once K0+K1 are dispatched (>= 294 of 334 finished), then x96pal once K2+K3 are.
cd "$(dirname "$0")/../.."
n() { cat "$@" 2>/dev/null | wc -l; }
R=evals/results
until [ "$(n $R/qwen_tools_e020_K0_all.jsonl $R/qwen_tools_e020_K1_all.jsonl)" -ge 294 ]; do sleep 60; done
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh K2 all 1 20
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh K3 all 1 20
date -u
until [ "$(n $R/qwen_tools_e020_K2_all.jsonl $R/qwen_tools_e020_K3_all.jsonl)" -ge 294 ]; do sleep 60; done
SLOTS=auto:10 experiments/E020_qwen_harness_80/launch.sh x96pal all 1 20
date -u
