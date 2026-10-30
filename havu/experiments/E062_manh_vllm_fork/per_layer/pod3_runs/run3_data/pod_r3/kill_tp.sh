#!/bin/bash
# SIGKILL exactly ONE vLLM TP worker (rank 2 or 3), found by its exact cmdline "VLLM::Worker_TP<r>"; nothing else.
# Usage: kill_tp.sh <2|3>
set -euo pipefail
r=$1; [ "$r" = 2 ] || [ "$r" = 3 ] || { echo "rank must be 2 or 3"; exit 1; }
p=$(pgrep -f "^VLLM::Worker_TP$r\$"); [ "$(echo "$p" | grep -c .)" = 1 ] || { echo "expected one TP$r worker, got: $p"; exit 1; }
[ "$(tr -d '\0' < /proc/$p/cmdline)" = "VLLM::Worker_TP$r" ] || { echo "cmdline mismatch"; exit 1; }
echo "KILL TP$r pid=$p at $(date -u +%T.%N | cut -c1-12) (epoch $(date +%s.%N | cut -c1-14))" | tee -a /root/e066/r3/kill_tp.log
kill -KILL $p
