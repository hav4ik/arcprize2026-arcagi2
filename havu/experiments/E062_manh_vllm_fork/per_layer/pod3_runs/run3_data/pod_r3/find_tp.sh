#!/bin/bash
# The live vLLM TP workers (cmdline "VLLM::Worker_TP<r>"), their parent chain up to the launcher, and the GPU processes.
for p in $(pgrep -f '^VLLM::Worker_TP[0-9]$'); do
  chain=""; q=$p; while [ "$q" -gt 1 ]; do chain="$chain $q:$(ps -o comm= -p $q)"; q=$(ps -o ppid= -p $q | tr -d ' '); done
  echo "$(tr -d '\0' < /proc/$p/cmdline) pid=$p pgid=$(ps -o pgid= -p $p | tr -d ' ') rss_gib=$(awk '/VmRSS/{printf "%.1f", $2/1048576}' /proc/$p/status) chain:$chain"
done | sort
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
