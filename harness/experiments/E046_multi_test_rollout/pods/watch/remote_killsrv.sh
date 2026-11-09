# runs on pod 4: SIGKILL the vLLM server process (child of the solver), log what happened
SOL=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /solver_dsv41.py/ {print $1}')
echo "$(date -u +%T) solver pid $SOL"
ps -eo pid,ppid,etime,args | awk -v s="$SOL" '($1 == s || $2 == s) && !/toolenv/' | cut -c1-200
SRV=$(ps -eo pid,ppid,args | awk -v s="$SOL" '$2 == s && /vllm/ && /serve/ {print $1}')
echo "$(date -u +%T) server pid(s): $SRV"
[ -z "$SRV" ] && { echo "NO SERVER PID FOUND"; exit 1; }
kill -9 $SRV && echo "$(date -u +%T) KILLED -9 $SRV"
sleep 20
echo "$(date -u +%T) after 20 s:"; ps -eo pid,ppid,etime,args | awk '/vllm|VLLM|solver_dsv41/ && !/awk/' | cut -c1-160 | head -12
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -5 /root/runs/c36_restart/run.log
