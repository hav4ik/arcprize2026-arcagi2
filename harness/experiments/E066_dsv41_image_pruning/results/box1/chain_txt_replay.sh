#!/bin/bash
# E066: after the image replay exits, replay 12M tokens of text-twin calibration rollouts (token-id path, E043
# replay_reap.py) with the observer on and the same frozen tool Engram mask.
cd /workspace/dsv4
while pgrep -f "replay_mm.py --tok data/e066/tok_img_calib_s" > /dev/null; do sleep 20; done
grep -q "\"rollouts\"" results/replay_main.log || { echo "image replay did not finish cleanly"; exit 1; }
rm -rf obs/e066_txt
tvenv/bin/python repo/experiments/E043_dsv41_real_pruning/scripts/replay_reap.py --tok data/e066/tok_txt_calib_sub \
  --tokens 12000000 --seed 1 --server http://127.0.0.1:18001 --ctrl /dev/shm/e043/ctrlB.json \
  --obs-dir /workspace/dsv4/obs/e066_txt --engram-keep /workspace/dsv4/data/e066/engram/e043tool_top64gib.npz
