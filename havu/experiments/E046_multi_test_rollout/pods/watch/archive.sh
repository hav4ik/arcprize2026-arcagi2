#!/bin/bash
# usage: archive.sh <port> <host>: copy every run on the pod (run.log, env.txt, mem.log, out/ without scratch) to the workspace
PORT=$1; HOST=$2; D=/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/pods/runs
for r in $(timeout 30 ssh -n -o ConnectTimeout=15 -p $PORT root@$HOST 'ls /root/runs'); do
  mkdir -p $D/$r
  timeout 300 ssh -n -o ConnectTimeout=15 -p $PORT root@$HOST "cd /root/runs/$r && tar czf - --exclude=scratch \$(ls -d run.log env.txt mem.log out 2>/dev/null)" | tar xzf - -C $D/$r
  echo "$r: $(du -sh $D/$r | cut -f1) $(ls $D/$r/out/dsv41_logs 2>/dev/null | tr '\n' ' ')"
done
