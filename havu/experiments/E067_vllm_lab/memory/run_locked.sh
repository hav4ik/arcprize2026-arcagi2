#!/usr/bin/env bash
# Run several cases back to back under one hold of the GPU-0 lock (waits up to $WAIT s).
#   run_locked.sh "NAME=a UTIL=0.96 MBT=8192 ..." "NAME=b ..."
SCR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
HERE="$(cd "$(dirname "$0")" && pwd)"
exec flock -w "${WAIT:-1200}" -E 75 "$SCR/gpu0.lock" bash -c '
  cd "$0"; for c in "$@"; do echo "### $c"; env NOLOCK=1 $c ./run_case.sh; done' "$HERE" "$@"
