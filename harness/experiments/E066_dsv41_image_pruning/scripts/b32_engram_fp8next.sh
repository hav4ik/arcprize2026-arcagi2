#!/bin/bash
# E066 Engram A/B, follow-on driver (user 2026-10-30 04:35: postpone the 48 GiB arm; MXFP4 vs FP8 first). Waits for the
# running MXFP4 harness (RESUME_PID), resumes it once if attempts are missing, stops its server, then runs the FP8
# control with the same server, harness and smoke step. Helper functions copied from b32_engram_ab.sh.
# Original header:
# E066 Engram A/B on B32's model: the tool E128 2-bit (HF chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit = the
# Kaggle E128 B32 serves) with its 64 GiB tool Engram as shipped (FP8), rounded to MXFP4, and cut to 48 GiB rows + MXFP4.
# Harness: B32's (the B32 kit with kaggle/dsv41_tools/pinned_candb overlaid, its solver's flags), tool code in the
# local executor as on Kaggle, run ON this box; no solutions here (scored locally from `pred`). Server: B32's settings
# (text only, fp8 KV, block 128, max-model-len 163840, max-num-batched-tokens 2048, prefix caching), each arm alone at
# TP4 in the user's order: MXFP4, then 48 GiB + MXFP4, then the FP8 control last. WAIT_PID: resume behind an already
# running MXFP4 build of the shipped table.
# One cache_salt per run. Needs: export/b32_e128 (HF download), tvenv, b32kit, data/arc-agi_evaluation_challenges.json.
set -u
D=/workspace/dsv4; cd $D
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
PY=$D/tvenv/bin/python; VP=$D/venv/bin/python
SCH13='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
mkdir -p /dev/shm/e043 results logs
log() { echo "$(date -u +%H:%M:%S) $*"; }
serve() {  # run_name export_dir gpus port [tp]
  echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > /dev/shm/e043/ctrl_$1.json
  DSV41_EXPERT_HUMMING="$SCH13" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$D/export/$2/gptq \
    VISION=0 GPUS=$3 TP=${5:-2} EP=0 PORT=$4 CTRL=/dev/shm/e043/ctrl_$1.json UTIL=0.86 SEQS=192 MBT=2048 MAXLEN=163840 \
    MODEL=$D/export/$2 bash $S43/serve.sh $1
}
ready() {  # run_name port
  for i in $(seq 1 120); do
    curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q dsv41 && break
    grep -q Traceback runs/$1/server.log 2>/dev/null && break
    sleep 15
  done
  curl -s -m 5 http://127.0.0.1:$2/v1/models | grep -q dsv41 || { log "SERVE_FAILED $1"; tail -20 runs/$1/server.log; return 1; }
  curl -s -m 300 http://127.0.0.1:$2/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv41","prompt":[0,5,6,7],"max_tokens":1}' > /dev/null
  log "READY $1 | $(grep 'ctrl loaded' runs/$1/server.log | tail -1 | cut -c1-120)"
}
evalrun() {  # tag port [workers]: B32's harness flags (pinned_candb/solver_dsv41.py HARNESS + make_args), resumed once
  for pass in 1 2; do
    (cd $D/b32kit/repo/evals && $PY $S/b32_local_eval.py \
      --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse \
      --max-tokens 64000 --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 --budget-warn 0.75 \
      --effort-header minimal --turns-note 5 --train-check --rescue-code --token-exact --model dsv41 --fallback-solve \
      --multi-test --auto-repair --multi-budget 0.5 --context-limit 163840 --max-turns 40 \
      --base-url http://127.0.0.1:$2/v1 \
      --extra-body "{\"chat_template_kwargs\": {\"thinking\": true, \"reasoning_effort\": 50}, \"temperature\": 1.0, \"top_p\": 1.0, \"cache_salt\": \"$1\"}" \
      --sandbox-url local --attempts 2 --workers ${3:-96} \
      --challenges $D/data/arc-agi_evaluation_challenges.json --out $D/results/$1.jsonl >> $D/logs/eval_$1.log 2>&1)
    n=$($PY -c "import json; print(len({(r['task_id'], r['attempt']) for r in map(json.loads, open('$D/results/$1.jsonl')) if 'error' not in r}))" 2>/dev/null || echo 0)
    log "$1 pass $pass: $n/240 attempts"
    [ "$n" -ge 240 ] && return 0
    [ "$n" -eq 0 ] && { log "EVAL_FAILED $1: no attempt finished"; tail -3 $D/logs/eval_$1.log; exit 1; }
  done
}
smoke() {  # tag port: 2 tasks x 1 attempt through the local executor before the real run
  (cd $D/b32kit/repo/evals && $PY $S/b32_local_eval.py --mode grid --effort low --prompt lean2sub --answer-tool \
    --final-mode session --lenient-parse --max-tokens 64000 --max-total-output 80000 --budget-soft 40000 --budget-notes 0.01 \
    --budget-warn 0.75 --effort-header minimal --turns-note 5 --train-check --rescue-code --token-exact --model dsv41 \
    --fallback-solve --multi-test --auto-repair --multi-budget 0.5 --context-limit 163840 --max-turns 40 \
    --base-url http://127.0.0.1:$2/v1 --extra-body "{\"chat_template_kwargs\": {\"thinking\": true, \"reasoning_effort\": 50}, \"temperature\": 1.0, \"top_p\": 1.0, \"cache_salt\": \"smoke_$1\"}" \
    --sandbox-url local --attempts 1 --workers 2 --limit 2 \
    --challenges $D/data/arc-agi_evaluation_challenges.json --out $D/results/smoke_$1.jsonl > $D/logs/smoke_$1.log 2>&1)
  n=$($PY -c "import json; print(sum(1 for r in map(json.loads, open('$D/results/smoke_$1.jsonl')) if 'error' not in r and r.get('tool_calls')))" 2>/dev/null || echo 0)
  [ "$n" -ge 1 ] || { log "SMOKE_FAILED $1"; tail -5 $D/logs/smoke_$1.log; exit 1; }
  log "smoke $1: $n records with tool calls"
}
arm() {  # run_name export_dir: one arm alone at TP4
  serve $1 $2 0,1,2,3 18001 4
  ready $1 18001 || exit 1
  smoke $1 18001
  log "arm $1 (TP4)"
  evalrun $1 18001 160
  bash $S43/stop.sh $1 0,1,2,3
  log "arm $1 done"
}
RESUME_PID=${RESUME_PID:?}
log "waiting for the MXFP4 harness (pid $RESUME_PID)"
while kill -0 $RESUME_PID 2>/dev/null; do sleep 30; done
n=$($PY -c "import json; print(len({(r['task_id'], r['attempt']) for r in map(json.loads, open('$D/results/b32_fp4.jsonl')) if 'error' not in r}))" 2>/dev/null || echo 0)
log "MXFP4 harness exited: $n/240 attempts"
[ "$n" -lt 240 ] && evalrun b32_fp4 18001 160
bash $S43/stop.sh b32_fp4 0,1,2,3
log "arm b32_fp4 done"
arm b32_fp8 b32_e128
log "ALL_DONE (48 GiB arm postponed by the user; table built in export/b32_48fp4)"
