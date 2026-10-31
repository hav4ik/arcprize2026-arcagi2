#!/bin/bash
# idempotent: start the detached runner unless one is alive
Q=/root/queue; [ -r $Q/runner.pid ] && kill -0 "$(cat $Q/runner.pid)" 2>/dev/null && { echo "runner already running pid $(cat $Q/runner.pid)"; exit 0; }
rm -f $Q/runner.stop; setsid nohup bash $Q/bin/queue_runner.sh > $Q/logs/runner.out 2>&1 < /dev/null &
sleep 2; echo "runner pid $(cat $Q/runner.pid)"
