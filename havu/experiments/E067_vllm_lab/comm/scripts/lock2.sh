#!/bin/bash
# Runs "$@" holding both lab GPU locks, without sitting on one lock while
# waiting for the other: try both non-blocking, back off, retry.
L=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
while true; do
  flock -n -E 75 "$L/gpu0.lock" flock -n -E 75 "$L/gpu1.lock" "$@"
  rc=$?
  [ $rc -ne 75 ] && exit $rc
  sleep 5
done
