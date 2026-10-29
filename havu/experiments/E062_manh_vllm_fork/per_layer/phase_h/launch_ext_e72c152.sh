#!/bin/bash
# Phase H step 4b (on the pod): the image-harness solver (kit_img, solver 9a6758cf = pinned_img with S25
# ARC_EXTERNAL_SERVER) attached to Manh's server config (serve_manh.sh, port 18084) on E72-152, through podrun.sh
# directly (replica_run.sh refuses to start while a GPU holds memory, which the external server does). Same
# workload and replica deadlines (shifted 600 s) as replica_run.sh writes; 120 workers.
# Usage (on the pod): launch_ext_e72c152.sh <name> <server log> [model dir] [kit dir] [workers]
set -u
N=${1:?name}; SLOG=${2:?server log}; MODEL=${3:-/root/models/e72c152}; KIT=${4:-/root/kit_img}; NW=${5:-120}
[ -e /root/runs/$N ] && { echo "run dir exists"; exit 3; }
mkdir -p /root/runs
setsid nohup /root/podrun.sh "$N" \
  ARC_ADMIT_MAX_WAITING=1 ARC_AUTO_REPAIR=1 ARC_CACHE_SALT=${N}_$(date +%H%M%S) \
  ARC_CHALLENGES=/root/data/evalx2t_s2026_challenges.json ARC_SOLUTIONS=/root/data/evalx2t_s2026_solutions.json \
  ARC_CONTEXT_LIMIT=163840 ARC_STOP_NEW_S=38400 ARC_FORCE_AT_S=39600 ARC_HARD_STOP_S=40800 ARC_KILL_AT_S=41700 \
  ARC_GRID_FORMAT=text+image ARC_KIT=$KIT ARC_KV_ADMIT_MAX=0.85 ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_MAX_NUM_SEQS=$NW ARC_MODEL_SRC=$MODEL ARC_MODEL_DIR=$MODEL \
  ARC_MULTI_BUDGET=0.5 ARC_MULTI_TEST=1 ARC_TOOL_PYTHON=/root/b32m/venv/bin/python ARC_VENV=/root/b32m/venv \
  ARC_WORKERS=$NW ARC_EXTERNAL_SERVER=http://127.0.0.1:18084/v1 ARC_EXTERNAL_SERVER_LOG=$SLOG \
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/root/b32m/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  > /root/runs/$N.launch.out 2>&1 < /dev/null &
sleep 5; cat /root/runs/$N.launch.out
