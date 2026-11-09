#!/bin/bash
# usage: [VENV=..] [ATTN=..] [MTP=n] probe2.sh <label> <model> <concurrency> [extra vLLM flags...]
# The v4b probe (E023 run B): 48 public-eval pairs (35 tasks), v4b agent settings, forced at server-ready + 120 min.
# Starts vLLM 0.23 TP4 (start_vllm.sh: prefix caching on, fp8 KV unless overridden, TRITON_ATTN unless ATTN is set),
# runs the solver against it, then prints the summary line and stops the server.
L=$1; MODEL=$2; C=$3; shift 3
R=/root/arc/runs/$L; mkdir -p $R
trap - INT
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 20
rm -rf /root/arc/work/arc_logs /root/arc/work/local_submission.json /root/arc/work/submission.json /root/arc/work/summary_local.json
MODEL=$MODEL VENV=${VENV:-/root/vllm-venv} ATTN=${ATTN:-TRITON_ATTN} setsid nohup /root/arc/start_vllm.sh ${MTP:-0} "$@" > $R/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do
  sleep 15
  if grep -q Traceback $R/server.log; then echo "$L SERVER_FAILED: $(grep -E 'Error' $R/server.log | tail -2 | cut -c1-300)"; return 1 2>/dev/null || exit 1; fi
done
echo "$L SERVER_READY $(date -u +%H:%M) $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
sed -e "s/ARC_CONCURRENCY=24/ARC_CONCURRENCY=$C/; s/ARC_MTP=0/ARC_MTP=${MTP:-0}/" /root/arc/runs/vllm_v4b_c24/run.sh > $R/run.sh; chmod +x $R/run.sh
$R/run.sh > $R/solver.log 2>&1 < /dev/null
cp /root/arc/work/summary_local.json /root/arc/work/submission.json $R/ 2>/dev/null
cp -r /root/arc/work/arc_logs $R/ 2>/dev/null
/opt/sglang/bin/python - $R/summary_local.json $L <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
tps = d["tokens"] / (d["gen_minutes"] * 60)
print(f"{sys.argv[2]} RESULT score {d['score']} | attempts {d['attempts']} answered {d['answered']} | tokens {d['tokens']} "
      f"in {d['gen_minutes']} min = {tps:.0f} tok/s | preemptions {d.get('server_preemption_lines')} | "
      f"kv max {d.get('kv_usage_max_pct')}% | {d.get('cache_hit_overall')}")
EOF
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 20
