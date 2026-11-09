#!/bin/bash
# E034: same E032 weights/server, two agents for 25 min each at c24 on the probe-48 tasks; prefix-cache counters before/after.
cd /root/arc
m() { curl -s localhost:8001/metrics | awk '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); s[a[1]]+=$2} /^vllm:num_preemptions_total/ {p+=$2} END {printf "%.0f %.0f %.0f\n", s["vllm:prefix_cache_queries_total"], s["vllm:prefix_cache_hits_total"], p}'; }
gen() { grep -E "Avg generation" runs/e034_cache/server.log | tail -n +$1 | sed -E "s/.*Avg prompt throughput: ([0-9.]+).*Avg generation throughput: ([0-9.]+).*Running: ([0-9]+).*/\1 \2 \3/" | awk '$3>=20 {p+=$1; g+=$2; r+=$3; n++} END {if(n) printf "windows %d gen %.0f per-stream %.2f prompt %.0f\n", n, g/n, g/r, p/n}'; }
echo "== B: v4b agent on E032 weights $(date -u +%H:%M)"
read q0 h0 p0 <<< "$(m)"; w0=$(grep -c "Avg generation" runs/e034_cache/server.log)
sed -e 's/ARC_LOCAL_READY_MIN=120/ARC_LOCAL_READY_MIN=25/' -e 's/ARC_LOCAL_HARD_MIN=15/ARC_LOCAL_HARD_MIN=3/' runs/vllm_v4b_c24/run.sh > runs/e034_v4b_run.sh; chmod +x runs/e034_v4b_run.sh
rm -rf /root/arc/work/arc_logs /root/arc/work/summary_local.json
timeout 2400 runs/e034_v4b_run.sh > runs/e034_cache/v4b_agent.log 2>&1
read q1 h1 p1 <<< "$(m)"
echo "v4b agent: prefix hit $(python3 -c "print(round(($h1-$h0)/max(1,$q1-$q0),3))") over $(( (q1-q0)/1000 ))k prompt tokens | preemptions $((p1-p0)) | $(gen $((w0+12)))"
mkdir -p runs/e034_cache/v4b_agent_logs && cp -r /root/arc/work/arc_logs runs/e034_cache/v4b_agent_logs/ 2>/dev/null
echo AB_B_DONE
