#!/bin/bash
# E081c: read-only hardware inventory of a 4xL4 pod (no GPU load). Compare with the Kaggle probes (identity section).
echo "== host"; hostname; nproc; free -g | head -2; lscpu | grep -E "Model name|Socket|NUMA node\(s\)" ; uname -r
echo "== gpus"; nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,utilization.gpu,memory.used --format=csv
echo "== throttle"; nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active,power.draw --format=csv
echo "== topo"; nvidia-smi topo -m
echo "== p2p"; nvidia-smi topo -p2p r 2>/dev/null | head -8
echo "== pcie link status"; for b in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do d=$(echo $b | sed 's/^0000//; s/^00000000/0000/' | tr 'A-F' 'a-f'); lspci -s ${d#0000:} -vv 2>/dev/null | grep -E "LnkCap:|LnkSta:" | sed "s/^/$d /"; done
echo "== container limits"; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null
