#!/bin/bash
# E062 L4: stage the Runpod 4xL4 pod (run from the workspace). Same parts as b4090/stage_box.sh, adapted:
#   download: E128 from HF over plain HTTP (E057b dl_e128_http.sh, revision 9bce98c5) with the full-access HF token
#             (the fine-grained key 404s on this repo; E056 box notes), sent over SSH stdin, never on disk
#   b32:      E042 setup_box.sh -> box_install.sh (yhfgyyf fork wheel venv), podprep (kit, 12-file overlay, tool venv,
#             data), B32's 16-file overlay with shas checked, /root/kit_b = kit + pinned_candb (b4090/b32_after_install.sh),
#             plus the replica workload evalx2t_s2026 in /root/data
# Usage: stage_pod.sh <port> <host> {download|b32|all}
set -euo pipefail
P=$1; H=$2; WHAT=${3:-all}
ROOT=/home/vu/Workspace/ARC-AGI-2
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$P" "root@$H")

if [[ $WHAT == download || $WHAT == all ]]; then
  "${SSH[@]}" 'mkdir -p /root/dsv41/runs 2>/dev/null || { mkdir -p /workspace/dsv41 && ln -sfn /workspace/dsv41 /root/dsv41 && mkdir -p /root/dsv41/runs; }
    [ -x /root/hfenv/bin/hf ] || { uv venv -q --python 3.12 /root/hfenv && uv pip install -q --python /root/hfenv/bin/python "huggingface_hub[cli]"; }
    /root/hfenv/bin/python -c "import huggingface_hub; print(\"hf_hub\", huggingface_hub.__version__)"'
  scp -q -P "$P" "$ROOT/experiments/E057b_workers64_stress/results/setup/dl_e128_http.sh" "root@$H:/root/dsv41/dl_e128_http.sh"
  sed -n 's/^huggingface_full_access_api_key:[[:space:]]*//p' "$ROOT/SECRETS.yaml" | tr -d "\"' " | \
    "${SSH[@]}" 'read -r T; HF_TOKEN="$T" setsid nohup bash /root/dsv41/dl_e128_http.sh > /dev/null 2>&1 < /dev/null & echo "download started"'
fi

if [[ $WHAT == b32 || $WHAT == all ]]; then
  bash "$ROOT/experiments/E042_dsv41_l4_synthetic/scripts/setup_box.sh" "root@$H" "$P"
  "${SSH[@]}" 'mkdir -p /root/podkit /root/e062'
  scp -q -P "$P" "$ROOT/kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz" "$ROOT/kaggle/dsv41_tools/out/local_run.py" \
    "$ROOT/experiments/E046_multi_test_rollout/pods/podrun.sh" "$ROOT/experiments/E046_multi_test_rollout/pods/memlog.sh" \
    "$ROOT/experiments/E046_multi_test_rollout/pods/podprep.sh" \
    "$ROOT/data/arc-agi_evaluation_challenges.json" "$ROOT/data/arc-agi_evaluation_solutions.json" \
    "$ROOT/data/arc-agi_training_challenges.json" "$ROOT/data/arc-agi_training_solutions.json" "root@$H:/root/podkit/"
  scp -q -P "$P" "$ROOT/experiments/E056_candidate_b/kit/vllm_overlay_rw.tgz" "root@$H:/root/e062/b32_overlay.tgz"
  scp -q -P "$P" "$ROOT/experiments/E056_candidate_b/kit/vllm_overlay_rw.sha256" "root@$H:/root/e062/b32_overlay.sha256"
  tar czf - -C "$ROOT/kaggle/dsv41_tools/pinned_candb" solver_dsv41.py run_deepseek_tools.py train_check.py \
    auto_repair.py auto_repair_names.json | "${SSH[@]}" 'mkdir -p /root/e062/pinned_candb && tar xzf - -C /root/e062/pinned_candb'
  W=$ROOT/experiments/E055_kaggle_replica/toolkit/workloads
  scp -q -P "$P" $W/evalx2t_s2026_challenges.json $W/evalx2t_s2026_solutions.json "root@$H:/root/e062/"
  scp -q -P "$P" "$ROOT/experiments/E062_manh_vllm_fork/b4090/b32_after_install.sh" "root@$H:/root/e062/b32_after_install.sh"
  "${SSH[@]}" 'setsid nohup bash /root/e062/b32_after_install.sh > /root/e062/b32_after_install.out 2>&1 < /dev/null & echo "b32 post-install waiter started"'
fi
