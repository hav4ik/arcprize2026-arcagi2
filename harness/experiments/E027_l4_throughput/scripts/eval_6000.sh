#!/bin/bash
# usage: [EXTRA_ENV='ARC_X=1 ...'] eval_6000.sh <label> <mtp tokens> <kv dtype> <attention backend> [extra vLLM flags...]
# Full public-eval run of the v4b agent recipe (attempt 1; free attempt_2 only; no time forcing) against one vLLM 0.31
# server on this single RTX PRO 6000 (TP1, FP8 weights, prefix caching on). Concurrency = KV tokens / 55k, capped at 40.
L=$1; MTP=$2; KV=$3; ATTN=$4; shift 4
R=/root/arc/runs/$L; mkdir -p $R
cd /root/arc
trap - INT
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 15
rm -rf /root/arc/work/arc_logs /root/arc/work/local_submission.json /root/arc/work/submission.json /root/arc/work/summary_local.json
until grep -q SETUP_DONE logs_setup.txt; do sleep 20; done
MODEL=Qwen/Qwen3.8-27B-FP8 VENV=/root/vllmnew ATTN=$ATTN setsid nohup ./start_vllm.sh $MTP --tensor-parallel-size 1 \
  --kv-cache-dtype $KV --language-model-only "$@" > $R/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do
  sleep 15
  if grep -qE "Traceback|No module named" $R/server.log || ! pgrep -f 'vllm.entrypoints.openai.api_server' > /dev/null; then echo "$L SERVER_FAILED: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; exit 1; fi
done
KVT=$(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1 | grep -oE '[0-9,]+' | tr -d ,)
C=$(( KVT / 55000 )); [ $C -gt 40 ] && C=40; [ $C -lt 8 ] && C=8
echo "$L SERVER_READY $(date -u +%H:%M) KV $KVT tokens -> concurrency $C"
cat > $R/run.sh <<EOF
#!/bin/bash
export ARC_SERVER=vllm ARC_MTP=$MTP ARC_KV_PER_SEQ=48000 ARC_RATE_PRIOR=600 ARC_LOCAL_READY_MIN=600 ARC_LOCAL_HARD_MIN=15 \
 ARC_TOOLS_TIER=xhigh ARC_TOOLS_CAP=48000 ARC_TOOLS_SOFT=24000 ARC_TOOLS_ANSWER_MODE=program ARC_TOOLS_PROMPT=lean2progroles \
 ARC_TASK_SUMMARY=1 ARC_HELPERS=1 ARC_FEEDBACK=rich ARC_TURN_THINK_CAP=8192 ARC_TURN_THINK_GRACE=2048 ARC_FREE_ATTEMPT2=1 \
 ARC_ATTEMPT2=off ARC_SESSION_MEM_MB=2048 ARC_THINK_CALL_RESCUE=1 ARC_LOCAL_SOURCE=eval ARC_CONCURRENCY=$C \
 ARC_EXTERNAL_SERVER=http://127.0.0.1:8001/v1 ARC_MODEL_ID=qwen3.8-27b ARC_MODE=local ARC_INPUT=/root/arc/input ARC_WORK=/root/arc/work $EXTRA_ENV
export ARC_T0=\$(date +%s)
cd /root/arc/work && exec /opt/sglang/bin/python -c "import signal,os,sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.executable, [sys.executable, '/root/arc/work/arc_solver/solver.py'])"
EOF
chmod +x $R/run.sh
$R/run.sh > $R/solver.log 2>&1 < /dev/null
cp /root/arc/work/summary_local.json /root/arc/work/submission.json $R/ 2>/dev/null
cp -r /root/arc/work/arc_logs $R/ 2>/dev/null
/opt/sglang/bin/python - $R/summary_local.json $L <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} RESULT score {d['score']} | attempts {d['attempts']} answered {d['answered']} | tokens {d['tokens']} in "
      f"{d['gen_minutes']} min = {d['tokens'] / (d['gen_minutes'] * 60):.0f} tok/s | {d.get('cache_hit_overall')}")
PY
grep -E "SpecDecoding metrics" $R/server.log | tail -1 | cut -c1-200
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done
echo "$L EVAL_DONE"
