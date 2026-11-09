#!/bin/bash
# E062: stage a fresh Vast 4x4090 box (run from the workspace). Three independent parts, each detached on the box:
#   1. E128 download (HF, plain HTTP, resumable; token over SSH stdin, never on disk) -> /root/dsv41/model-e128t
#   2. B32's stack: E042 setup_box.sh/box_install.sh (yhfgyyf fork wheel + freeze + patches) -> /root/dsv41/venv,
#      then podprep (kit + 12-file overlay, tool venv, data), then E056's 16-file overlay (B32), shas checked;
#      kit_b = kit + pinned_candb files at their kit paths
#   3. Manh's fork: build_manh.sh -> /root/manh/venv + wheel in /root/manh/dist
# Usage: stage_box.sh <port> <host> {download|b32|manh|all}
set -euo pipefail
P=$1; H=$2; WHAT=${3:-all}
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E062_manh_vllm_fork/b4090
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$P" "root@$H")

if [[ $WHAT == download || $WHAT == all ]]; then
  "${SSH[@]}" 'mkdir -p /root/dsv41/runs; command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } > /dev/null 2>&1
    export PATH=$HOME/.local/bin:$PATH; [ -x /root/hfenv/bin/hf ] || { uv venv -q --python 3.12 /root/hfenv && uv pip install -q --python /root/hfenv/bin/python "huggingface_hub[cli]"; }
    /root/hfenv/bin/hf version 2>/dev/null || /root/hfenv/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"'
  scp -q -P "$P" "$ROOT/experiments/E057b_workers64_stress/results/setup/dl_e128_http.sh" "root@$H:/root/dsv41/dl_e128_http.sh"
  # the token goes over stdin into the detached downloader's environment only
  sed -n 's/^huggingface_api_key:[[:space:]]*//p' "$ROOT/SECRETS.yaml" | tr -d "\"' " | \
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
  scp -q -P "$P" "$HERE/b32_after_install.sh" "root@$H:/root/e062/b32_after_install.sh"
  "${SSH[@]}" 'setsid nohup bash /root/e062/b32_after_install.sh > /root/e062/b32_after_install.out 2>&1 < /dev/null & echo "b32 post-install waiter started"'
fi

if [[ $WHAT == manh || $WHAT == all ]]; then
  "${SSH[@]}" 'mkdir -p /root/manh'
  scp -q -P "$P" "$HERE/build_manh.sh" "root@$H:/root/manh/build_manh.sh"
  "${SSH[@]}" 'setsid nohup bash /root/manh/build_manh.sh > /dev/null 2>&1 < /dev/null & echo "manh build started"'
fi
