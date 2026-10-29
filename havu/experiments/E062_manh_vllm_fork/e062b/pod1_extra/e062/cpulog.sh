#!/bin/bash
# E062 L4: cgroup CPU use every 60 s (quota 22.95 CPUs): CPUs used over the window, throttled periods and seconds
O=$1; D=/sys/fs/cgroup/cpu,cpuacct; [ -d $D ] || D=/sys/fs/cgroup/cpu
u0=$(cat $D/cpuacct.usage); s0=$(awk "/throttled_time/{print \$2}" $D/cpu.stat); n0=$(awk "/nr_throttled/{print \$2}" $D/cpu.stat); t0=$(date +%s.%N)
while true; do
  sleep 60
  u=$(cat $D/cpuacct.usage); s=$(awk "/throttled_time/{print \$2}" $D/cpu.stat); n=$(awk "/nr_throttled/{print \$2}" $D/cpu.stat); t=$(date +%s.%N)
  echo "$(date -u +%T) cpus_used=$(awk -v a=$u0 -v b=$u -v x=$t0 -v y=$t "BEGIN{printf \"%.2f\", (b-a)/1e9/(y-x)}") throttled_periods=$((n-n0)) throttled_s=$(awk -v a=$s0 -v b=$s "BEGIN{printf \"%.1f\", (b-a)/1e9}") load1=$(cut -d" " -f1 /proc/loadavg)" >> $O
  u0=$u; s0=$s; n0=$n; t0=$t
done
