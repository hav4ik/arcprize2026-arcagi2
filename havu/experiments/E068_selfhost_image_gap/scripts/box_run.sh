#!/bin/bash
# E068 on a 4x RTX PRO 6000 box with E066's stack (/workspace/dsv4/venv = E043 lock + patches, tvenv): the full
# DeepSeek-V4.1-Flash-1017 served as E066 full_ref was (TP4 + EP, fp8 KV, prefix caching, vision, max-model-len
# 262,144), then the perception probe and the real/grey NLL; then one change at a time:
#   ref       E066 full_ref settings: probe, NLL real, NLL grey
#   kvauto    + --kv-cache-dtype auto (BF16 KV): probe, NLL real
#   noprefix  + --no-enable-prefix-caching: probe
#   vitdp     + --mm-encoder-tp-mode data (vision tower replicated per GPU instead of TP-sharded): probe
#   sdpa      + --mm-encoder-attn-backend TORCH_SDPA (vision attention kernel): probe
#   causal    + --hf-overrides is_mm_prefix_lm=false --disable-chunked-mm-input: causal LM attention over image spans,
#             as DeepSeek's reference (inference/model.py get_window_topk_idxs); the fork's config sets
#             is_mm_prefix_lm = vision_n_layers > 0, i.e. bidirectional image spans. Probe, NLL real, NLL grey.
# kvauto turned out to be a duplicate of ref: the fork maps --kv-cache-dtype auto to fp8 and accepts only DeepSeek
# V4.1's native mixed FP8/FP4 KV format ("fp8_ds_mla").
# Usage: box_run.sh dl            (needs HF_TOKEN; the full checkpoint, ~12 min on E066's box)
#        box_run.sh <arm>...      (default: ref kvauto noprefix)
# Outputs: /workspace/dsv4/results/e068/{probe,nll}_<arm>.jsonl; server logs in /workspace/dsv4/runs/e068_<arm>/.
set -u
D=/workspace/dsv4
R=$D/repo
S=$R/experiments/E068_selfhost_image_gap/scripts
S43=$R/experiments/E043_dsv41_real_pruning/scripts
M=$D/DeepSeek-V4.1-Flash-1017
O=$D/results/e068
DATA=$D/data/e068_sub   # the 32 dev rollouts nll_eval_mm.py picks (--per-task --n-seq 32 --seed 0 --max-len 60000)
SERVER=http://127.0.0.1:18001
CTRL=/dev/shm/e043/ctrl_e068.json
PY=$D/tvenv/bin/python
mkdir -p $O /dev/shm/e043 $D/logs
cd $D

if [ "${1:-}" = dl ]; then
  hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir $M --max-workers 16
  echo "DL_DONE $(du -sh $M | cut -f1)"
  exit 0
fi

echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL

serve() {  # serve <arm> [extra vllm args]: stop any E068 server, start this arm, wait until it answers
  local arm=$1; shift
  for r in e068_ref e068_kvauto e068_noprefix e068_vitdp e068_sdpa e068_causal; do
    [ -f $D/runs/$r/server.pid ] && bash $S43/stop.sh $r 0,1,2,3 > /dev/null 2>&1 && rm -f $D/runs/$r/server.pid
  done
  local pf=1; [ "$arm" = noprefix ] && pf=0
  VISION=1 GPUS=0,1,2,3 TP=4 EP=1 PORT=18001 CTRL=$CTRL UTIL=0.90 SEQS=64 MBT=8192 MAXLEN=262144 PREFIX=$pf \
    MODEL=$M bash $S43/serve.sh e068_$arm --limit-mm-per-prompt '{"image": 64}' "$@"
  for i in $(seq 1 120); do
    curl -s -m 5 $SERVER/v1/models | grep -q dsv41 && break
    grep -q Traceback $D/runs/e068_$arm/server.log 2>/dev/null && break
    sleep 15
  done
  curl -s -m 5 $SERVER/v1/models | grep -q dsv41 || { echo "SERVE_FAILED $arm"; tail -30 $D/runs/e068_$arm/server.log; return 1; }
  grep "ctrl loaded" $D/runs/e068_$arm/server.log | tail -1
  echo "SERVER_READY $arm $(date -u +%H:%M)"
}

probe() { $PY $S/perception_probe.py --out $O/probe_$1.jsonl --base-url $SERVER/v1 --workers 64 > $O/probe_$1.log 2>&1; }
nll() {   # nll <arm> <real|grey>
  $PY $S/nll_image_ablation.py --images $2 --tok $DATA/tok_img_dev_s/shard0 --parquet $DATA/e065_image_dev32.parquet \
    --per-task --n-seq 32 --seed 0 --max-len 60000 --workers 4 --server $SERVER --tag $1_$2 --out $O/nll_$1.jsonl \
    > $O/nll_$1_$2.log 2>&1
}

for arm in "${@:-ref kvauto noprefix}"; do
  for a in $arm; do
    case $a in
      ref)      serve ref || continue
                probe ref & nll ref real; nll ref grey; wait ;;
      kvauto)   serve kvauto --kv-cache-dtype auto || continue
                probe kvauto & nll kvauto real; wait ;;
      noprefix) serve noprefix || continue
                probe noprefix ;;
      vitdp)    serve vitdp --mm-encoder-tp-mode data || continue
                probe vitdp ;;
      sdpa)     serve sdpa --mm-encoder-attn-backend TORCH_SDPA || continue
                probe sdpa ;;
      causal)   serve causal --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input || continue
                grep -q "multimodal-bidirectional" $D/runs/e068_causal/server.log && echo "MM_PREFIX_STILL_ON" || echo "MM_PREFIX_OFF"
                probe causal & nll causal real; nll causal grey; wait ;;
      *) echo "unknown arm $a" >&2; continue ;;
    esac
    $PY $S/perception_probe.py --score $O/probe_$a.jsonl
    echo "ARM_DONE $a $(date -u +%H:%M)"
  done
done
for r in e068_ref e068_kvauto e068_noprefix e068_vitdp e068_sdpa e068_causal; do
  [ -f $D/runs/$r/server.pid ] && bash $S43/stop.sh $r 0,1,2,3 > /dev/null 2>&1 && rm -f $D/runs/$r/server.pid
done
echo "ALL_DONE $(date -u +%H:%M)"
