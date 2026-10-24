#!/bin/bash
until grep -qE "FOLLOWUPS_DONE|ABORT" /root/dsv41/runs/followups.log 2>/dev/null; do sleep 15; done
/root/dsv41/scripts/stop.sh p2_mbt8k >> /root/dsv41/runs/followups.log 2>&1
exec /root/dsv41/scripts/nccl_sweep.sh > /root/dsv41/runs/nccl_sweep.out 2>&1
