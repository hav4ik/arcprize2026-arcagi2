#!/bin/bash
# E020 queue 3: x96pex once x96pal is dispatched (slots.json, managed by the pal cap controller).
cd "$(dirname "$0")/../.."
until [ "$(cat evals/results/qwen_tools_e020_x96pal_all.jsonl | wc -l)" -ge 147 ]; do sleep 60; done
SLOTS=auto:0 experiments/E020_qwen_harness_80/launch.sh x96pex all 1 20
date -u
