#!/bin/bash
# E020 queue 2: once K2 attempt 1 + K2nc are dispatched, start K3 attempt 1; once that is dispatched, give x96pal (router/slots.json) its slots back through a
# second cap controller that counts the K arms' leases (router/slots_k.json); then x96pex once x96pal is dispatched.
cd "$(dirname "$0")/../.."
R=evals/results
k2a1() { grep -c '"attempt": 1,' $R/qwen_tools_e020_K2_all.jsonl; }
n() { cat "$@" 2>/dev/null | wc -l; }
until [ $(( $(k2a1) + $(n $R/qwen_tools_e020_K2nc_all.jsonl) )) -ge 284 ]; do sleep 60; done
SLOTS=auto:0 SLOT_STATE=experiments/E019_qwen_program_answers/router/slots_k.json \
  experiments/E020_qwen_harness_80/launch.sh K3 all 2 25 --only-attempt 1
echo "K3 attempt 1 launched $(date -u)"
k3a1() { grep -c '"attempt": 1,' $R/qwen_tools_e020_K3_all.jsonl; }
until [ "$(k3a1)" -ge 142 ]; do sleep 60; done
nohup .venv/bin/python experiments/E020_qwen_harness_80/cap_controller.py --per-box 10 \
  --state experiments/E019_qwen_program_answers/router/slots.json \
  --other-state experiments/E019_qwen_program_answers/router/slots_k.json >> $R/log_e020_cap_controller_pal.txt 2>&1 &
echo "pal controller $! $(date -u)"
until [ "$(n $R/qwen_tools_e020_x96pal_all.jsonl)" -ge 147 ]; do sleep 60; done
SLOTS=auto:0 experiments/E020_qwen_harness_80/launch.sh x96pex all 1 20
date -u
