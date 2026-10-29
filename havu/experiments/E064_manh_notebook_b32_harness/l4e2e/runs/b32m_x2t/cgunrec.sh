#!/bin/bash
# E064b: the cgroup (v1) unreclaimable host memory (total_rss + total_shmem, as Kaggle counts against its 173 GiB) every 60 s
while true; do
  awk -v t=$(date -u +%T) "/^total_rss /{r=\$2} /^total_shmem /{s=\$2} /^total_cache /{c=\$2} END {printf \"%s cg_rss_gib=%.1f cg_shmem_gib=%.1f cg_unreclaimable_gib=%.1f cg_cache_gib=%.1f\n\", t, r/2^30, s/2^30, (r+s)/2^30, c/2^30}" /sys/fs/cgroup/memory/memory.stat >> /root/runs/b32m_x2t/cgmem.log
  sleep 60
done
