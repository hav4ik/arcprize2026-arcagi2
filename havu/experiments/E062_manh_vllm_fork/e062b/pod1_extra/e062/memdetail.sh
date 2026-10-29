#!/bin/bash
# memdetail <log>: every 120 s, host memory that cannot be reclaimed (AnonPages + Shmem; pinned CUDA host memory shows
# as Shmem), page cache, MemAvailable, the server's PSS split, the tool sandboxes' total (processes with
# oom_score_adj 1000, as the solver's executor sets), the other processes' anon PSS, and per-GPU memory used
while true; do
  m=$(awk '/^(AnonPages|Shmem|MemAvailable|Cached|PageTables|SUnreclaim|KernelStack):/{printf "%s=%.2f ", $1, $2/2^20}' /proc/meminfo | tr -d ':')
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' )
  p=$(for q in $(pgrep -f "VLLM::|vllm serve|solver_dsv41"); do awk '/^Pss_Anon:/{a=$2} /^Pss_Shmem:/{s=$2} END{printf "%.2f/%.2f;", a/2^20, s/2^20}' /proc/$q/smaps_rollup 2>/dev/null; done)
  sb=$(for q in /proc/[0-9]*; do [ "$(cat $q/oom_score_adj 2>/dev/null)" = 1000 ] && awk '/^Pss_Anon:/{print $2}' $q/smaps_rollup 2>/dev/null; done | awk '{s+=$1; n++} END{printf "%d/%.2f", n, s/2^20}')
  other=$(for q in /proc/[0-9]*; do [ "$(cat $q/oom_score_adj 2>/dev/null)" = 1000 ] && continue; awk '/^Pss_Anon:/{print $2}' $q/smaps_rollup 2>/dev/null; done | awk '{s+=$1} END{printf "%.2f", s/2^20}')
  n=$(pgrep -c python)
  echo "$(date -u +%T) ${m}gpu_mib=${g} pss_anon/shmem_gib=${p} sandboxes_n/pss_anon_gib=${sb} container_nonsandbox_pss_anon_gib=${other} python_procs=$n" >> "$1"
  sleep 120
done
