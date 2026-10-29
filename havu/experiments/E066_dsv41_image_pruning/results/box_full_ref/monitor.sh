#!/bin/bash
# E066 full_ref monitor: one line per interval (default 900 s) with result-file progress, the box's vLLM metrics
# (running/waiting, KV usage, prefix-cache hit rate cumulative and since the last line, preemptions, generated
# tokens and rate), GPU memory and whether the local tunnel and eval process are alive.
# Usage: monitor.sh <ssh_port> <ip> <eval_pid> <tunnel_pid> [interval_s]
PORT=$1; IP=$2; EPID=$3; TPID=$4; IV=${5:-900}
ROOT=/home/vu/Workspace/ARC-AGI-2
RES=$ROOT/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e066_full_ref.jsonl
ph=0; pq=0; pg=0; pt=0
while true; do
  t=$(date -u +%s)
  rec=$(wc -l < "$RES" 2>/dev/null || echo 0)
  err=$(grep -c '"error": "' "$RES" 2>/dev/null || echo 0)
  m=$(timeout 40 ssh -o ConnectTimeout=15 -p "$PORT" root@"$IP" 'curl -s -m 5 http://127.0.0.1:18000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|prefix_cache_hits_total|prefix_cache_queries_total|num_preemptions_total|generation_tokens_total)\{" | sed -E "s/^vllm:([a-z_]+)\{[^}]*\} /\1=/"; echo gpumem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" ","); echo hostmem_avail_g=$(free -g | awk "/Mem:/{print \$7}")' 2>/dev/null | grep '=' | tr '\n' ' ')
  h=$(sed -n 's/.*prefix_cache_hits_total=\([0-9.e+]*\).*/\1/p' <<<"$m"); q=$(sed -n 's/.*prefix_cache_queries_total=\([0-9.e+]*\).*/\1/p' <<<"$m")
  g=$(sed -n 's/.*generation_tokens_total=\([0-9.e+]*\).*/\1/p' <<<"$m")
  extra=$(python3 -c "
h,q,g,ph,pq,pg,t,pt=[float(x or 0) for x in '${h:-0} ${q:-0} ${g:-0} $ph $pq $pg $t $pt'.split()]
win=(h-ph)/(q-pq) if q>pq else float('nan'); rate=(g-pg)/(t-pt) if pt and t>pt else float('nan')
print(f'hit_cum={h/q if q else 0:.3f} hit_win={win:.3f} gen_tok_s={rate:.0f}')")
  kill -0 "$EPID" 2>/dev/null && e=eval_up || e=EVAL_EXITED
  kill -0 "$TPID" 2>/dev/null && tu=tunnel_up || tu=TUNNEL_DOWN
  curl -s -m 10 http://127.0.0.1:18100/v1/models | grep -q dsv41 && api=api_ok || api=API_FAIL
  echo "$(date -u +%FT%T) records=$rec errors=$err $e $tu $api $extra $m"
  ph=${h:-0}; pq=${q:-0}; pg=${g:-0}; pt=$t
  [ "$e" = EVAL_EXITED ] && break
  sleep "$IV"
done
