#!/bin/bash
# E066: the V12 + images setup with the Engram cut from 64 to 56 GiB (user 2026-10-30: "after this test is done, do
# another one where we reduce engrams from 64GiB to 56GiB"). After run_v12img.sh (64 GiB, c1fx) prints EVAL_DONE:
#  1. serve export/c1f_56 with the causal fix (serve_eval.sh c1f56x)
#  2. dev NLL on the 64 held-out training rollouts E066 screens with -> results/q_screen_c1f56.jsonl, paired with
#     results/q_screen_c1f.jsonl (the 64 GiB table on the same server type, 14:09 UTC)
#  3. the same V12 + images ARC eval (run_v12img.sh, TAG=v12img_c1f56)
set -u
D=/workspace/dsv4; cd $D
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
log() { echo "$(date -u +%H:%M:%S) $*"; }
until grep -q SUBSET56_READY logs/subset56.log; do grep -q -E "Traceback|Error" logs/subset56.log && { log SUBSET56_FAILED; exit 1; }; sleep 30; done
log "56 GiB variant: $(head -1 logs/subset56.log | cut -c1-300)"
[ "${SKIP_WAIT:-0}" = 1 ] || until grep -q "EVAL_DONE v12img_c1f$" logs/run_v12img.log; do sleep 60; done
log "switching to 56 GiB (SKIP_WAIT=${SKIP_WAIT:-0}: 1 = the 64 GiB run was stopped early by the user)"
bash $S/serve_eval.sh c1f56x > logs/serve_c1f56x.log 2>&1 || { log "SERVE_FAILED c1f56x"; tail -5 logs/serve_c1f56x.log; exit 1; }
grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/c1f56x/server.log || { log "FIX_NOT_APPLIED c1f56x"; exit 1; }
log "READY c1f56x | $(tail -1 logs/serve_c1f56x.log | cut -c1-140)"
ENGRAM=none bash $S/nll_masks_mm.sh results/q_screen_c1f56.jsonl c1f56x:none > logs/nll_c1f56.log 2>&1 || { log NLL_FAILED; tail -3 logs/nll_c1f56.log; }
log "NLL56_DONE"
TAG=v12img_c1f56 bash $S/run_v12img.sh > logs/run_v12img56.log 2>&1
log "E56_ALL_DONE"
