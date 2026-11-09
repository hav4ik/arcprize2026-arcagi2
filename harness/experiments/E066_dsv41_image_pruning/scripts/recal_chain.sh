#!/bin/bash
# E066 fixed-path re-prune of the image model (user 2026-10-30: "don't re-evaluate candidate 1. go straight to
# re-calibration + new model eval"; vision tower unchanged, LLM experts pruned to 128 and 2-bit). Candidate 1's
# pipeline with E068's fix (the full model attends causally over image spans) and nothing else changed:
#  1. after fix_c1_chain.sh's text replay (TXT_PID) exits: aggregate it, add the fixed-path image stats, select E128
#     with candidate 1's class weights (keep_shift.py reports the move vs candidate 1) -> cand/c1f_keep.npy
#  2. GPTQ capture under that keep set on the fixed full model (imgF; seeds and sizes as candidate 1: 0.7M image +
#     0.7M text tokens), stop imgF, 2-bit GPTQ on 4 GPUs -> gptq/c1f
#  3. export (vision tower kept; the 64 GiB tool Engram hardlinked from export/c1_gptq, the same table) + content check
#  4. serve the export with the fix (serve_eval.sh c1fx) -> EXPORT_READY: the upload and the local ARC eval start here
#  5. dev NLL, 32 image + 32 text dev rollouts (E066's screen set) -> results/q_screen_c1f.jsonl
set -u
D=/workspace/dsv4; cd $D
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
CTRL=/dev/shm/e043/ctrlB.json
E=$D/data/e066
log() { echo "$(date -u +%H:%M:%S) $*"; }
up() { curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41; }
serve_imgF() {
  [ -f runs/imgF/server.pid ] && bash $S43/stop.sh imgF 0,1,2,3 > /dev/null 2>&1
  echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
  VISION=1 GPUS=0,1,2,3 TP=4 PORT=18001 CTRL=$CTRL MODEL=$D/DeepSeek-V4.1-Flash-1017 bash $S43/serve.sh imgF \
    --limit-mm-per-prompt '{"image": 64}' --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input
  for i in $(seq 1 120); do up && break; grep -q Traceback runs/imgF/server.log 2>/dev/null && break; sleep 15; done
  up || { log "SERVE_FAILED imgF"; tail -20 runs/imgF/server.log; exit 1; }
  grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/imgF/server.log || { log "FIX_NOT_APPLIED imgF"; exit 1; }
  log "READY imgF (restarted)"
}
mkdir -p cand results $E/engram
[ -e $E/engram/e043tool_top64gib.npz ] || ln -s ../e043tool_top64gib.npz $E/engram/e043tool_top64gib.npz
# 1. statistics and keep set
while kill -0 ${TXT_PID:?} 2>/dev/null; do sleep 20; done
log "text replay exited: $(tail -1 logs/replay_txt_fixed.log | cut -c1-160)"
$D/tvenv/bin/python $S/class_stats.py aggregate --obs obs/e066f_txt --out $E/stats_txt.npz > logs/agg_txt.log 2>&1 \
  || { log "AGG_FAILED txt"; tail -3 logs/agg_txt.log; exit 1; }
log "aggregate txt: $(tail -1 logs/agg_txt.log)"
$D/tvenv/bin/python - $E <<'PY'
import sys, numpy as np
e = sys.argv[1]; a, b = np.load(f"{e}/stats_img_causal.npz"), np.load(f"{e}/stats_txt.npz")
np.savez(f"{e}/stats_both_causal.npz", **{k: a[k] + b[k] for k in ("N", "A", "W", "pos")})
PY
$D/tvenv/bin/python $S/class_stats.py select --stats $E/stats_both_causal.npz --w-enc 1,1,10,1 --w-dec 0,1,10,1 --avg 128 \
  --out-dir $E/sel_causal > logs/sel_causal.log 2>&1 || { log SELECT_FAILED; tail -3 logs/sel_causal.log; exit 1; }
NEW=$E/sel_causal/keep_contrib_enc1-1-10-1_dec0-1-10-1_uniform_avg128.npy
$D/tvenv/bin/python $S/keep_shift.py --both-bidir $E/stats_both_bidir.npz --img-causal $E/stats_img_causal.npz \
  --txt $E/stats_txt.npz --c1-keep $E/c1_keep.npy --new-keep $NEW --out $E/keep_shift.json > logs/keep_shift.log 2>&1 \
  || { log KEEP_SHIFT_FAILED; tail -5 logs/keep_shift.log; exit 1; }
log "KEEP_SHIFT $($D/tvenv/bin/python -c "import json; r = json.load(open('$E/keep_shift.json')); print(r['overlap_new_vs_c1'], r['kept_share_causal_stats'], r['image_routing_kept_by_c1'])" | cut -c1-600)"
cp $NEW cand/c1f_keep.npy
rm -rf obs/e066f_img obs/e066f_txt  # ~45 GB; the aggregated stats are kept
# 2. GPTQ capture on the fixed full model, then GPTQ
up || serve_imgF
bash $S/gptq_candidate.sh dump c1f cand/c1f_keep.npy > logs/recal_dump.log 2>&1 || { log DUMP_FAILED; tail -5 logs/recal_dump.log; exit 1; }
log "capture: $(tail -3 logs/recal_dump.log | tr '\n' ' ' | cut -c1-300)"
bash $S43/stop.sh imgF 0,1,2,3
bash $S/gptq_candidate.sh quant c1f cand/c1f_keep.npy > logs/recal_quant.log 2>&1 || { log QUANT_FAILED; tail -5 logs/recal_quant.log; exit 1; }
log "$(tail -1 logs/recal_quant.log)"
rm -rf calx/c1f_by_layer
# 3. export + content check
ENGRAM_SRC=$D/export/c1_gptq bash $S/export_candidate.sh export c1f cand/c1f_keep.npy gptq/c1f \
  "E128 2-bit GPTQ, tool use with images, causal image attention (E066 candidate 1, fixed-path calibration)" \
  "tool + image keep set from fixed-path (causal image span) replays of E065 image and text harness rollouts, class weights prompt/gen/tool-call/image 1/1/10/1 (layers 0-19), 0/1/10/1 (20-39)" \
  > logs/recal_export.log 2>&1 || { log EXPORT_FAILED; tail -5 logs/recal_export.log; exit 1; }
log "$(tail -1 logs/recal_export.log)"
bash $S/export_candidate.sh check c1f cand/c1f_keep.npy gptq/c1f > logs/recal_check.log 2>&1 || { log CHECK_FAILED; tail -5 logs/recal_check.log; exit 1; }
log "$(tail -1 logs/recal_check.log)"
# 4. serve the export with the fix
bash $S/serve_eval.sh c1fx > logs/serve_c1fx.log 2>&1 || { log "SERVE_FAILED c1fx"; tail -5 logs/serve_c1fx.log; exit 1; }
grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/c1fx/server.log || { log "FIX_NOT_APPLIED c1fx"; exit 1; }
log "EXPORT_READY c1f | $(tail -1 logs/serve_c1fx.log | cut -c1-160)"
# 5. dev NLL on the served export (no runtime masks: the export is pruned and carries the compact Engram)
ENGRAM=none bash $S/nll_masks_mm.sh results/q_screen_c1f.jsonl c1fx:none > logs/recal_nll.log 2>&1 \
  || { log "NLL_FAILED"; tail -5 logs/recal_nll.log; exit 1; }
log "C1F_NLL_DONE"
