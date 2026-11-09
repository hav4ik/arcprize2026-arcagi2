#!/bin/bash
# E062 accuracy arms on the Vast 48 GB 4090 box (58066982): the 120 Kaggle eval tasks x 1 attempt through the notebook
# pipeline (/root/podrun.sh + local_run.py) with B32's kit (kit_b: solver 77fe80a3, harness 882c817f, train_check
# 8fe36175, auto-repair) and B32's solver settings (x56, max-num-seqs 64, gate 0.85 + queue hold, multi-test 0.5,
# context limit), as E052 arm A (76.5 task pass@1) but on B32's code. Deadlines are out of the way (30 / 31 / 32 / 33 h).
# Memory is NOT Kaggle-sized: util 0.92 of a 48 GB card (accuracy only).
#   ARM=manh bash launch_acc.sh   # Manh's fork (wheel b9ad670a67 + patches 01-03 of branch arc/e128-kaggle), pod #1's flags
#   ARM=b32  bash launch_acc.sh   # B32's stack (16-file overlay 1a3a6679, B32's humming_utils), B32's flags
set -euo pipefail
ARM=${ARM:?manh|b32}; NAME=${NAME:-acc_$ARM}
read -r V EXTRA SENV < <(python3 -I - "$ARM" <<'PY'
import base64, json, sys
arm = sys.argv[1]
off = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"cpu_bytes_to_use": 34359738368, "offload_prompt_only": False}}
graphs = json.dumps({"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64})
if arm == "manh":   # pod #1's launch_manh.sh flags (its 2026-10-28 ~09:50 version), graph cap 64 for x56
    off["kv_connector_extra_config"]["eviction_policy"] = "lru"
    extra = ["--linear-backend", "humming", "--decode-context-parallel-size", "4", "--cp-kv-cache-interleave-size",
             "64", "--prefix-cache-retention-interval", "512", "--cpu-offload-gb", "1", "--cpu-offload-params",
             "embed_tokens", "--compilation-config", graphs, "--kv-transfer-config", json.dumps(off)]
    senv, venv = {"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300",
                  "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64"}, "/root/manh/venv"
elif arm == "b32":  # B32's ARC_SERVER_EXTRA_ARGS (launch_candBf_x2t.sh)
    off["kv_connector_extra_config"].update(swa_store_policy="checkpoint", shared_region=False, eviction_policy="lru")
    extra = ["--linear-backend", "humming", "--compilation-config", graphs, "--prefix-cache-retention-interval", "512",
             "--kv-transfer-config", json.dumps(off)]
    senv, venv = {}, "/root/dsv41/venv"
else:
    sys.exit("ARM must be manh|b32")
b = lambda x: base64.b64encode(json.dumps(x).encode()).decode()
print(venv, b(extra), b(senv))
PY
)
ssh -o BatchMode=yes -o ConnectTimeout=20 -p 50331 root@38.146.30.19 "set -e; mkdir -p /root/runs; test ! -e /root/runs/$NAME || { echo exists; exit 1; }
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ \$used -lt 1000 ] || { echo GPU busy \$used; exit 1; }
  for v in \$(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN\$|HUGGING_FACE_HUB_TOKEN\$|HF_API_TOKEN\$)'); do unset \$v; done
  EXTRA=\$(echo $EXTRA | base64 -d); SENV=\$(echo $SENV | base64 -d)
  setsid nohup /root/podrun.sh $NAME ARC_KIT=/root/kit_b ARC_VENV=$V PATH=$V/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
    ARC_ATTEMPTS=1 ARC_WORKERS=56 ARC_MAX_NUM_SEQS=64 ARC_GPU_UTIL=0.92 ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 \
    ARC_CONTEXT_LIMIT=163840 ARC_KV_ADMIT_MAX=0.85 ARC_KV_ADMIT_MIN_WORKERS=16 ARC_ADMIT_MAX_WAITING=1 ARC_AUTO_REPAIR=1 \
    ARC_STOP_NEW_S=108000 ARC_FORCE_AT_S=111600 ARC_HARD_STOP_S=115200 ARC_KILL_AT_S=118800 \
    \"ARC_SERVER_ENV_JSON=\$SENV\" \"ARC_SERVER_EXTRA_ARGS=\$EXTRA\" > /root/runs/$NAME.launch.out 2>&1 < /dev/null &
  sleep 5; cat /root/runs/$NAME.launch.out; grep -E 'VENV|WORKERS|ATTEMPTS|EXTRA|ENV_JSON' /root/runs/$NAME/env.txt | cut -c1-400
  setsid nohup /root/memlog.sh /root/runs/$NAME/mem.log > /dev/null 2>&1 < /dev/null &" 2>&1 | grep -v -E "^(Welcome|Have fun|AI agents)"
