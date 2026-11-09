#!/bin/bash
# E066: (re)start the box 1 eval server for one candidate and check it serves the intended model.
#   c1x   candidate 1 from its own export (export/c1_gptq: GPTQ-only, compact Engram, no runtime masks needed)
#   e72q  the per-layer 72-152 candidate: runtime keep mask on the full checkpoint + its GPTQ dir (the E066 fork
#         cannot load per-layer exports); checks that the server's last "ctrl loaded" line carries the keep set
#   c1xc  c1x with E068's fix for the bidirectional image-span attention (--hf-overrides is_mm_prefix_lm false,
#         --disable-chunked-mm-input)
#   c1fx  the fixed-path re-prune (recal_chain.sh: export/c1f_gptq) with the same fix
#   c1f56x  c1fx with its Engram cut from 64 to 56 GiB (export/c1f_56: engram_subset.py, training census)
# Usage: serve_eval.sh c1x|c1xc|c1fx|c1f56x|e72q   (stops the other one first). Prints EVAL_SERVER_READY <name> on success.
set -u
D=/workspace/dsv4
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
CTRL=/dev/shm/e043/ctrlB.json
SCH13='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
cd $D
# one-shot hook: the FP4 Engram pilot runs before the next server start when fp4_pilot.todo exists
[ -f $D/fp4_pilot.todo ] && { rm -f $D/fp4_pilot.todo; bash $S/fp4_pilot.sh > logs/fp4_pilot.log 2>&1; }
for r in c1q c1x c1xc c1fx c1f56x e72q imgC imgF c1fp4; do [ -f runs/$r/server.pid ] && bash $S43/stop.sh $r 0,1,2,3 > /dev/null 2>&1; done
case $1 in
  c1x)
    echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
    DSV41_EXPERT_HUMMING="$SCH13" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$D/export/c1_gptq/gptq \
      VISION=1 GPUS=0,1,2,3 TP=4 EP=0 PORT=18001 CTRL=$CTRL UTIL=0.86 SEQS=192 MBT=8192 MAXLEN=262144 \
      MODEL=$D/export/c1_gptq bash $S43/serve.sh c1x --limit-mm-per-prompt '{"image": 64}' ;;
  c1xc|c1fx|c1f56x)
    X=$D/export/c1_gptq; [ $1 = c1fx ] && X=$D/export/c1f_gptq; [ $1 = c1f56x ] && X=$D/export/c1f_56
    echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
    DSV41_EXPERT_HUMMING="$SCH13" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$X/gptq \
      VISION=1 GPUS=0,1,2,3 TP=4 EP=0 PORT=18001 CTRL=$CTRL UTIL=0.86 SEQS=192 MBT=8192 MAXLEN=262144 \
      MODEL=$X bash $S43/serve.sh $1 --limit-mm-per-prompt '{"image": 64}' \
      --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input ;;
  e72q) bash $S/serve_gptq.sh e72q cand/e72c152_keep.npy gptq/e72c152 ;;
  *) echo "usage: serve_eval.sh c1x|c1xc|c1fx|c1f56x|e72q" >&2; exit 2 ;;
esac
for i in $(seq 1 100); do
  curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break
  grep -q Traceback runs/$1/server.log 2>/dev/null && break
  sleep 15
done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo "SERVE_FAILED $1"; tail -30 runs/$1/server.log; exit 1; }
curl -s -m 300 http://127.0.0.1:18001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":[0,5,6,7],"max_tokens":1}' > /dev/null
last=$(grep "ctrl loaded" runs/$1/server.log | tail -1)
case $1 in
  c1x|c1xc|c1fx|c1f56x) echo "$last" | grep -q "experts=None engram=None" || { echo "CTRL_WRONG $1: $last"; exit 1; } ;;
  e72q) echo "$last" | grep -q "experts=$D/cand/e72c152_keep.npy" || { echo "CTRL_WRONG $1: $last"; exit 1; } ;;
esac
echo "EVAL_SERVER_READY $1 | $last"
