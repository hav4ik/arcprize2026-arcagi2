#!/bin/bash
# E066 follow-up to E068 (the fork attends bidirectionally over image spans; DeepSeek's reference is causal; user
# 2026-10-30: "Expert check + C1 image re-eval"). On node 5 (Vast 58332344), after model-serving frees the GPUs:
#  1. full model, VISION=1, E068's fix: --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input
#  2. E066's calibration replays with the observer, same selections: image (12M tokens, seed 1, minus the 7 preflight
#     rollouts = 374 rollouts / 12,005,133 tokens; data/e066/e066_pre_manifest.jsonl rebuilt from the preflight's
#     shard0 / 200k / seed 7) and text (12M, seed 1). Relative --tok paths as in E066 (the exclude keys use them).
#  3. aggregate, select E128 with candidate 1's class weights, compare with candidate 1's keep set (keep_shift.py)
#  4. swapped >= SHIFT_MAX experts per layer on average: stop with DECISION_NEEDED, full model still up (a re-prune
#     starts with gptq_candidate.sh dump on it); else stop it and serve candidate 1 with the fix (serve_eval.sh c1xc)
#     for the local image-harness eval.
set -u
D=/workspace/dsv4; cd $D
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
CTRL=/dev/shm/e043/ctrlB.json
G=$D/data/e066/e043tool_top64gib.npz
E=$D/data/e066
log() { echo "$(date -u +%H:%M:%S) $*"; }
mkdir -p /dev/shm/e043 obs logs
echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
if [ "${SKIP_SERVE:-0}" != 1 ]; then  # SKIP_SERVE=1: imgF already up from an earlier start
VISION=1 GPUS=0,1,2,3 TP=4 PORT=18001 CTRL=$CTRL MODEL=$D/DeepSeek-V4.1-Flash-1017 bash $S43/serve.sh imgF \
  --limit-mm-per-prompt '{"image": 64}' --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break
  grep -q Traceback runs/imgF/server.log 2>/dev/null && break
  sleep 15
done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { log "SERVE_FAILED imgF"; tail -20 runs/imgF/server.log; exit 1; }
fi
grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/imgF/server.log || { log "FIX_NOT_APPLIED imgF"; exit 1; }
log "READY imgF | $(grep 'ctrl loaded' runs/imgF/server.log | tail -1 | cut -c1-120)"
rm -rf obs/e066f_img obs/e066f_txt
$D/tvenv/bin/python $S/replay_mm.py --tok data/e066/tok_img_calib_s/shard* --parquet $E/e065_image.parquet \
  --tokens 12000000 --seed 1 --exclude $E/e066_pre_manifest.jsonl --server http://127.0.0.1:18001 --ctrl $CTRL \
  --obs-dir $D/obs/e066f_img --engram-keep $G > logs/replay_img_fixed.log 2>&1
log "image replay: $(tail -1 logs/replay_img_fixed.log)"
grep -q '"rollouts": 374, "tokens": 12005133' logs/replay_img_fixed.log || { log "IMAGE_REPLAY_DIFFERS (E066: 374 / 12005133)"; exit 1; }
$D/tvenv/bin/python $S43/replay_reap.py --tok data/e066/tok_txt_calib_sub --tokens 12000000 --seed 1 \
  --server http://127.0.0.1:18001 --ctrl $CTRL --obs-dir $D/obs/e066f_txt --engram-keep $G > logs/replay_txt_fixed.log 2>&1
log "text replay: $(tail -1 logs/replay_txt_fixed.log)"
for p in "img_causal obs/e066f_img" "txt obs/e066f_txt"; do
  set -- $p
  $D/tvenv/bin/python $S/class_stats.py aggregate --obs $2 --out $E/stats_$1.npz > logs/agg_$1.log 2>&1 \
    || { log "AGG_FAILED $1"; tail -3 logs/agg_$1.log; exit 1; }
  log "aggregate $1: $(tail -1 logs/agg_$1.log)"
done
$D/tvenv/bin/python - $E <<'PY'
import sys, numpy as np
e = sys.argv[1]; a, b = np.load(f"{e}/stats_img_causal.npz"), np.load(f"{e}/stats_txt.npz")
np.savez(f"{e}/stats_both_causal.npz", **{k: a[k] + b[k] for k in ("N", "A", "W", "pos")})
PY
$D/tvenv/bin/python $S/class_stats.py select --stats $E/stats_both_causal.npz --w-enc 1,1,10,1 --w-dec 0,1,10,1 --avg 128 \
  --out-dir $E/sel_causal > logs/sel_causal.log 2>&1 || { log SELECT_FAILED; tail -3 logs/sel_causal.log; exit 1; }
$D/tvenv/bin/python $S/keep_shift.py --both-bidir $E/stats_both_bidir.npz --img-causal $E/stats_img_causal.npz \
  --txt $E/stats_txt.npz --c1-keep $E/c1_keep.npy \
  --new-keep $E/sel_causal/keep_contrib_enc1-1-10-1_dec0-1-10-1_uniform_avg128.npy --out $E/keep_shift.json \
  > logs/keep_shift.log 2>&1 || { log KEEP_SHIFT_FAILED; tail -5 logs/keep_shift.log; exit 1; }
sw=$($D/tvenv/bin/python -c "import json; print(json.load(open('$E/keep_shift.json'))['overlap_new_vs_c1']['swapped_mean'])")
log "keep set shift: $sw experts per layer swapped vs candidate 1 (B32's shipped E128 vs C1: 3.2)"
if python3 -c "import sys; sys.exit(0 if $sw >= ${SHIFT_MAX:-2} else 1)"; then
  log "DECISION_NEEDED: keep set moved $sw >= ${SHIFT_MAX:-2}; full model left up on :18001 for a re-prune"
  exit 0
fi
bash $S43/stop.sh imgF 0,1,2,3
bash $S/serve_eval.sh c1xc || { log "SERVE_FAILED c1xc"; exit 1; }
grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/c1xc/server.log || { log "FIX_NOT_APPLIED c1xc"; exit 1; }
log "C1XC_READY"
