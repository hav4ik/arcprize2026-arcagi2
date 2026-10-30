#!/bin/bash
# Runs ar_bench.py once per NCCL env setting (separate processes: NCCL reads
# transport envs once per process). Call under both GPU locks.
P=/home/vu/miniconda3/envs/vllm/bin/python
cd "$(dirname "$0")/.."
run() { label=$1; shift; env "$@" LAB_TMP=$LAB_TMP NCCL_DEBUG=INFO timeout 300 $P scripts/ar_bench.py --label $label --out results/ar_$label.json > results/ar_$label.log 2>&1; echo "$label rc=$?"; }
run p2p_disable NCCL_P2P_DISABLE=1
run proto_ll NCCL_PROTO=LL
run proto_ll128 NCCL_PROTO=LL128
run proto_simple NCCL_PROTO=Simple
run shm_memcpy NCCL_SHM_USE_CUDA_MEMCPY=1
run nch8 NCCL_MIN_NCHANNELS=8
run nch8_ll NCCL_MIN_NCHANNELS=8 NCCL_PROTO=LL
run shm_disable NCCL_SHM_DISABLE=1
