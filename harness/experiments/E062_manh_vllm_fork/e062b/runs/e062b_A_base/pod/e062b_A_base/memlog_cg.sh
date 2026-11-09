#!/bin/bash
while true; do
  cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
  g=$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';')
  echo "$(date -u +%T) used_gib=$(free -g | awk 'NR==2{print $3}') rss_gib=$(ps -eo rss= | awk '{s+=$1} END {printf "%.1f", s/1048576}') mlocked_kb=$(grep Mlocked /proc/meminfo | awk '{print $2}') cg_gib=$(awk -v b=$cg 'BEGIN{printf "%.1f", b/1073741824}') gpu_w_mhz=$g" >> $1
  sleep 60
done
