#!/bin/bash
# E042: all-reduce micro-bench across NCCL protocol settings (GPUs must be free).
# NCCL_P2P_LEVEL=SYS is box-only (on Kaggle P2P is ~12x slower); the others are Kaggle-transferable settings.
O=/root/dsv41/runs/nccl; mkdir -p $O; cd $O
T=/root/dsv41/venv/bin/torchrun; B=/root/dsv41/scripts/nccl_allreduce_bench.py
unset NCCL_PROTO NCCL_ALGO NCCL_P2P_LEVEL
NCCL_DEBUG=INFO $T --nproc-per-node 4 $B --label default --out default.json 2>&1 | grep -E "^default|via (SHM|P2P|NET)" | sort -u | head -20 | tee default.txt
for proto in Simple LL128 LL; do
  NCCL_PROTO=$proto $T --nproc-per-node 4 $B --label $proto --out $proto.json 2>&1 | grep -E "^$proto" | tee $proto.txt
done
NCCL_P2P_LEVEL=SYS $T --nproc-per-node 4 $B --label p2p_sys --out p2p_sys.json 2>&1 | grep -E "^p2p_sys" | tee p2p_sys.txt
NCCL_P2P_LEVEL=SYS NCCL_PROTO=Simple $T --nproc-per-node 4 $B --label p2p_simple --out p2p_simple.json 2>&1 | grep -E "^p2p_simple" | tee p2p_simple.txt
echo NCCL_DONE
