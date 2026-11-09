#!/bin/bash
# E037 v3 (replaces v2 after its NVFP4 benches): Kaggle-pace probes only — FP8+DFlash, FP8, INT4 AutoRound (vLLM 0.32).
# E037 v2 (2026-10-21, after the NVFP4 probe of v1): DFlash servers get --gpu-memory-utilization 0.88 (v1 OOMed at 0.95);
# the bench prompt parameter is 8000 (bench_decode2's unit gives ~3x tokens: v1's 24000 made 74k-token prompts).
# E037: NVFP4 / INT4 AutoRound / FP8 targets with and without the DFlash2 draft on 4xL4, vLLM 0.32.
# Per model: long-context decode bench (24k prompt, 1000 out, c1/c8/c24) with and without DFlash, a greedy sanity check,
# then the 48-pair Kaggle-pace probe (v4b agent, forced at server-ready + 120 min) on the faster config.
# Kaggle-style server flags only (no tool/reasoning parsers). Log: /root/arc/e037/chain.log
set -u
cd /root/arc; E=/root/arc/e037; mkdir -p $E
log() { echo "[$(date -u +%H:%M)] $*" | tee -a $E/chain.log; }
stop_server() { for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 25
                for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5; }
serve() {  # name model draft(0/1) [extra...]
  local name=$1 model=$2 draft=$3; shift 3
  stop_server; mkdir -p $E/$name
  local spec=()
  [ "$draft" = 1 ] && spec=(--speculative-config '{"method": "dflash", "model": "/workspace/models/dflash2", "num_speculative_tokens": 7}' --gpu-memory-utilization ${DFLASH_MEM:-0.88})
  (MODEL=$model VENV=/root/vllm32 setsid nohup ./start_vllm.sh 0 --language-model-only "${spec[@]}" "$@" > $E/$name/server.log 2>&1 < /dev/null &)
  local t=0
  until curl -s -m 5 localhost:8001/v1/models > /dev/null; do
    sleep 15; t=$((t+15))
    if grep -q "Traceback" $E/$name/server.log || [ $t -gt 2400 ]; then log "$name SERVE FAILED: $(grep -E 'Error' $E/$name/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1; fi
  done
  log "$name up: $(grep -oE 'Selected [A-Za-z0-9]+ for [A-Za-z0-9]+|Using [A-Za-z0-9]+ (backend|kernel)[^.]{0,60}' $E/$name/server.log | sort -u | head -4 | tr '\n' ' ') $(grep -o 'GPU KV cache size: [0-9,]* tokens' $E/$name/server.log | tail -1)"
}
bench() {  # name
  local name=$1
  timeout 3000 /opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 $name $E/$name/long.json 8000 1000 1 8 24 > $E/$name/long.log 2>&1
  log "$name long-context decode: $(grep -oE "'concurrency': [0-9]+, 'wall_s': [0-9.]+, 'gen_tokens': [0-9]+, 'tok_s': [0-9.]+, 'per_req_tok_s': [0-9.]+" $E/$name/long.log | sed -E "s/'concurrency': ([0-9]+).*'tok_s': ([0-9.]+), 'per_req_tok_s': ([0-9.]+)/c\1: \2 tok\/s (\3 per stream)/" | tr '\n' ';')"
  grep -E "accept|draft" $E/$name/server.log | grep -oE "(Mean acceptance length|Avg Draft acceptance rate|Accepted throughput)[^,]*" | tail -2 | tr '\n' ' ' | sed "s/^/  $name spec: /" >> $E/chain.log; echo >> $E/chain.log
  /root/arc/ab_kernel_greedy.sh $name $E/$name/greedy.json > $E/$name/greedy.log 2>&1; log "$name greedy: $(cut -c1-140 $E/$name/greedy.log | head -1)"
}
best() {  # name_a name_b -> the one with higher c24 tok/s
  python3 - "$E/$1/long.json" "$E/$2/long.json" "$1" "$2" <<'PY'
import json, sys
def c24(p):
    try:
        d = json.load(open(p)); rows = d if isinstance(d, list) else d.get("results", d.get("levels", []))
        return max((r.get("tok_s", 0) for r in rows if r.get("concurrency") == 24), default=0)
    except Exception: return 0
a, b = c24(sys.argv[1]), c24(sys.argv[2]); print(sys.argv[3] if a >= b else sys.argv[4])
PY
}
probe() {  # name (server already up)
  local name=$1
  rm -rf /root/arc/work/arc_logs /root/arc/work/local_submission.json /root/arc/work/submission.json /root/arc/work/summary_local.json
  /root/arc/runs/vllm_v4b_c24/run.sh > $E/$name/probe_solver.log 2>&1 < /dev/null
  cp /root/arc/work/summary_local.json /root/arc/work/submission.json $E/$name/ 2>/dev/null; cp -r /root/arc/work/arc_logs $E/$name/probe_arc_logs 2>/dev/null
  log "$name PROBE: $(/opt/sglang/bin/python -c "
import json; d = json.load(open('$E/$name/summary_local.json'))
print(d['score'], '| attempts', d['attempts'], 'answered', d['answered'], '| tokens', d['tokens'], 'in', d['gen_minutes'], 'min =', round(d['tokens'] / (d['gen_minutes'] * 60)), 'tok/s |', d.get('cache_hit_overall', '')[:60])" 2>&1)"
}
dl() {  # hf repo, local dir
  rm -rf $2; /root/vllm32/bin/hf download $1 --local-dir $2 > $E/dl_$(basename $2).log 2>&1 && log "downloaded $1 ($(du -sh $2 | cut -f1)); disk $(df -h / | tail -1 | awk '{print $4}') free" || { log "DOWNLOAD FAILED $1: $(tail -2 $E/dl_$(basename $2).log)"; return 1; }
}

log "E037 v3 start"
stop_server; rm -rf /workspace/models/nvfp4
dl Qwen/Qwen3.8-27B-FP8 /workspace/models/fp8 && {
  DFLASH_MEM=0.90 serve fp8_dflash_probe /workspace/models/fp8 1 || DFLASH_MEM=0.86 serve fp8_dflash_probe /workspace/models/fp8 1
  probe fp8_dflash_probe
  serve fp8_probe /workspace/models/fp8 0 && probe fp8_probe
  stop_server; rm -rf /workspace/models/fp8
}
dl CySpiegel/Qwen3.8-27B-Int4-AutoRound /workspace/models/int4ar && {
  serve int4ar_probe /workspace/models/int4ar 0 && probe int4ar_probe
  stop_server
}
log "E037_DONE"
