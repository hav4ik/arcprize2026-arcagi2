#!/bin/bash
# memdetail <log>: every 120 s, host memory that cannot be reclaimed (AnonPages + Shmem; pinned CUDA host memory shows
# as Shmem), page cache, MemAvailable, top-process PSS split, and per-GPU memory used (flat = no allocator growth)
while true; do
  m=$(awk '/^(AnonPages|Shmem|MemAvailable|Cached|PageTables|SUnreclaim|KernelStack):/{printf "%s=%.2f ", $1, $2/2^20}' /proc/meminfo | tr -d ':')
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' )
  p=$(for q in $(pgrep -f "VLLM::|vllm serve|solver_dsv41"); do awk '/^Pss_Anon:/{a=$2} /^Pss_Shmem:/{s=$2} END{printf "%.2f/%.2f;", a/2^20, s/2^20}' /proc/$q/smaps_rollup 2>/dev/null; done)
  n=$(pgrep -c python)
  echo "$(date -u +%T) ${m}gpu_mib=${g} pss_anon/shmem_gib=${p} python_procs=$n" >> "$1"
  sleep 120
done
