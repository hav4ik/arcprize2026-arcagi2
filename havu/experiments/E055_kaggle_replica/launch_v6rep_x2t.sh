#!/bin/bash
# E055 V6 calibration replica: dsv41-tools-submit-mtf Version 6 exactly (LB 61.94): Kaggle V5 kit + pinned V6
# solver 4b344011 / harness a8a5b665 (kaggle/dsv41_tools/pinned_mtf_v6), kit train_check 44c4d210, kit overlay 377d83e0,
# x36, multi-test 0.5, context limit, KV gate 0.6/16, V5 server flags (retention 128); same workload as candBf_x2t.
cd "$(dirname "$0")"
exec toolkit/replica_run.sh 30092 213.173.105.28 v6rep_x2t evalx2t_s2026 \
  ARC_KIT=/root/kit_v6 \
  ARC_VENV=/root/dsv41/venv \
  ARC_WORKERS=36 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.6 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}"]'
