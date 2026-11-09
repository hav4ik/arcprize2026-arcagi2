#!/bin/bash
# Burst 4: remaining vLLM decode configs.
cd "$(dirname "$0")/.."
scripts/e2e_one.sh tp1 1 CUDA_VISIBLE_DEVICES=0
scripts/e2e_one.sh tp2_shm_disable 2 NCCL_SHM_DISABLE=1
scripts/e2e_one.sh tp2_har_cold 2 HAR_ENABLE=1 PYTHONPATH=$PWD/scripts/har_vllm HAR_POLL_NS=50000
scripts/e2e_one.sh tp2_p2p_disable 2 NCCL_P2P_DISABLE=1
scripts/e2e_one.sh tp2_proto_ll 2 NCCL_PROTO=LL
scripts/e2e_one.sh tp2_har_small 2 HAR_ENABLE=1 PYTHONPATH=$PWD/scripts/har_vllm HAR_UNIT_KB=4 HAR_WRITERS=4 HAR_READERS=4 HAR_MAX_MB=0.125
