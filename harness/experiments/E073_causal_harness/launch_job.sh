#!/bin/bash
# A finite job, never the E072 filler loop. The pod remains running after completion.
set -u
label=$1
bash /root/queue/bin/run_job.sh /root/e073/jobs/$label.job > /root/e073/logs/$label.log 2>&1
rc=$?
printf '%s\n' "$rc" > /root/e073/logs/$label.exit
exit "$rc"
