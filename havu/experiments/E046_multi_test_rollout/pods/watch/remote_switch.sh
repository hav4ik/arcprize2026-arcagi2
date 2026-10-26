# usage: bash -s <oldrun> <newrun> [ENV=VALUE ...]   (runs on the pod)
OLD=$1; NEW=$2; shift 2
ts() { date -u +%T; }
SOL=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /solver_dsv41.py/ {print $1}')
LR=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /local_run.py/ {print $1}')
ML=$(ps -eo pid,comm,args | awk -v r="/root/runs/$OLD/mem.log" '$2 == "memlog.sh" && index($0, r) {print $1}')
echo "$(ts) solver=$SOL local_run=$LR memlog=$ML"
[ -n "$SOL" ] && kill -TERM $SOL
for i in $(seq 1 60); do [ -z "$SOL" ] || ! kill -0 $SOL 2>/dev/null && break; sleep 5; done
kill -0 $SOL 2>/dev/null && echo "$(ts) solver still alive after 300 s" || echo "$(ts) solver exited"
tail -4 /root/runs/$OLD/run.log
for p in $LR; do kill -0 $p 2>/dev/null && { kill -TERM $p; sleep 3; kill -0 $p 2>/dev/null && kill -KILL $p; echo "$(ts) local_run $p killed"; }; done
for p in $ML; do kill $p 2>/dev/null && echo "$(ts) memlog $p stopped"; done
for i in $(seq 1 36); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$u" -lt 1024 ] && break; sleep 5
done
echo "$(ts) GPU max used ${u} MiB; leftover python: $(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /vllm|solver|local_run/ {print $1}' | tr '\n' ' ')"
/root/podrun.sh $NEW "$@" > /dev/null 2>&1 < /dev/null
sleep 5; tail -2 /root/runs/$NEW/run.log
grep -E 'KIT|WORKERS|MULTI|KV_ADMIT|GPU_UTIL|SALT' /root/runs/$NEW/env.txt | tr '\n' ' '; echo
setsid nohup /root/memlog.sh /root/runs/$NEW/mem.log > /dev/null 2>&1 < /dev/null &
echo "$(ts) STARTED $NEW"
