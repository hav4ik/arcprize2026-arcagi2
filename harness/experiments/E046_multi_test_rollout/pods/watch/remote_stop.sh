# usage: bash -s <run>   (on the pod) stop the run cleanly: SIGTERM solver, wait, then its local_run, then its memlog
R=$1; ts() { date -u +%T; }
SOL=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /solver_dsv41.py/ {print $1}')
LR=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /local_run.py/ {print $1}')
ML=$(ps -eo pid,comm,args | awk -v r="/root/runs/$R/mem.log" '$2 == "memlog.sh" && index($0, r) {print $1}')
echo "$(ts) solver=$SOL local_run=$LR memlog=$ML"
[ -n "$SOL" ] && kill -TERM $SOL
for i in $(seq 1 60); do [ -z "$SOL" ] || ! kill -0 $SOL 2>/dev/null && break; sleep 5; done
kill -0 $SOL 2>/dev/null && echo "$(ts) solver still alive after 300 s" || echo "$(ts) solver exited"
for i in $(seq 1 6); do [ -z "$LR" ] || ! kill -0 $LR 2>/dev/null && break; sleep 5; done
for p in $LR; do kill -0 $p 2>/dev/null && { kill -TERM $p; sleep 3; kill -0 $p 2>/dev/null && kill -KILL $p; echo "$(ts) local_run $p killed"; }; done
for p in $ML; do kill $p 2>/dev/null && echo "$(ts) memlog $p stopped"; done
tail -5 /root/runs/$R/run.log
python3 -c "import json; d=json.load(open('/root/runs/$R/out/submission.json')); print('submission ok', len(d))"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
