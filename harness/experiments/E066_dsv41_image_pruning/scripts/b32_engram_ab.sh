#!/bin/bash
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
until grep -q DOWNLOAD_DONE logs/download_b32.log && grep -q SETUP_DONE logs/setup_env.log; do sleep 20; done
log "download + env ready"
# MXFP4 of the shipped table, then phase 1 starts while the 48 GiB variants build
if [ -f export/b32_fp4/config.json ] && grep -q '"weight_chunks"' logs/fp4_b32.log; then
  log "MXFP4 variant already built"
elif [ -n "${WAIT_PID:-}" ]; then
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 15; done
  grep -q '"weight_chunks"' logs/fp4_b32.log || { log "FP4_FAILED (resumed build)"; exit 1; }
else
  nice -n 5 $VP $S/fp4_engram_chunked.py --src export/b32_e128 --out export/engram_b32_fp4 --threads 160 > logs/fp4_b32.log 2>&1 \
    || { log FP4_FAILED; exit 1; }
fi
$VP $S/assemble_engram_variant.py --src export/b32_e128 --engram-dir export/engram_b32_fp4 --out export/b32_fp4 || exit 1
log "FP4 variant: $(tail -1 logs/fp4_b32.log)"
# the 48 GiB rows + MXFP4 table builds on the CPU while the first arm runs (B48_PID: wait for a build already running)
if [ -n "${B48_PID:-}" ]; then
  ( while kill -0 $B48_PID 2>/dev/null; do sleep 30; done ) &
else
( nice -n 10 $VP $S/engram_subset.py --src export/b32_e128 --mask data/e066/e043tool_top64gib.npz \
    --counts data/e066/counts_calib_img.npz --geo data/e066/engram_geometry.npz --budget-gib 48 --out export/engram_b32_48 \
    > logs/subset48.log 2>&1 \
  && $VP $S/assemble_engram_variant.py --src export/b32_e128 --engram-dir export/engram_b32_48 --out export/b32_48 \
  && nice -n 10 $VP $S/fp4_engram_chunked.py --src export/b32_48 --out export/engram_b32_48fp4 --threads 64 > logs/fp4_b32_48.log 2>&1 \
  && $VP $S/assemble_engram_variant.py --src export/b32_48 --engram-dir export/engram_b32_48fp4 --out export/b32_48fp4 \
  && echo VARIANT48_READY > logs/variant48.flag ) &
fi
B48=$!
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
arm b32_fp4 b32_fp4
wait $B48; grep -q VARIANT48_READY logs/variant48.flag 2>/dev/null || { log "48 GiB VARIANT FAILED"; tail -5 logs/subset48.log logs/fp4_b32_48.log; exit 1; }
log "48 GiB variant: $(head -1 logs/subset48.log | cut -c1-300)"
arm b32_48fp4 b32_48fp4
arm b32_fp8 b32_e128
log ALL_DONE
