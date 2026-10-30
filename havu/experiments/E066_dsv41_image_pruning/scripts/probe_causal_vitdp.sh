#!/bin/bash
# E066/E068 follow-up (user 2026-10-30: "yeah run the probe"): E068's perception probe on the full model with BOTH the
# causal image-attention fix and the unsharded vision tower. E068 measured them one at a time: `causal` (fix, sharded
# tower) and `vitdp` (--mm-encoder-tp-mode data, bidirectional). Same server settings as E068's box_run.sh serve() and
# the same probe command (60 grids x 2 samples, 64 workers); only the two flag sets are combined. Stops the eval server
# (c1fx) first. Output: /workspace/dsv4/results/e066x/probe_causal_vitdp.{jsonl,log}.
set -u
D=/workspace/dsv4; cd $D
S68=$D/repo/experiments/E068_selfhost_image_gap/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
CTRL=/dev/shm/e043/ctrl_e066x.json
O=$D/results/e066x
mkdir -p $O /dev/shm/e043 logs
log() { echo "$(date -u +%H:%M:%S) $*"; }
for r in c1fx c1xc imgF e066x_causal_vitdp; do [ -f runs/$r/server.pid ] && bash $S43/stop.sh $r 0,1,2,3 > /dev/null 2>&1; done
echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
VISION=1 GPUS=0,1,2,3 TP=4 EP=1 PORT=18001 CTRL=$CTRL UTIL=0.90 SEQS=64 MBT=8192 MAXLEN=262144 PREFIX=1 \
  MODEL=$D/DeepSeek-V4.1-Flash-1017 bash $S43/serve.sh e066x_causal_vitdp --limit-mm-per-prompt '{"image": 64}' \
  --hf-overrides '{"is_mm_prefix_lm": false}' --disable-chunked-mm-input --mm-encoder-tp-mode data
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break
  grep -q Traceback runs/e066x_causal_vitdp/server.log 2>/dev/null && break
  sleep 15
done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { log "SERVE_FAILED"; tail -30 runs/e066x_causal_vitdp/server.log; exit 1; }
grep -q "'hf_overrides': {'is_mm_prefix_lm': False}" runs/e066x_causal_vitdp/server.log || { log "FIX_NOT_APPLIED"; exit 1; }
grep -q "'mm_encoder_tp_mode': 'data'" runs/e066x_causal_vitdp/server.log || { log "VITDP_NOT_APPLIED"; exit 1; }
log "SERVER_READY | $(grep 'ctrl loaded' runs/e066x_causal_vitdp/server.log | tail -1 | cut -c1-120)"
$D/tvenv/bin/python $S68/perception_probe.py --out $O/probe_causal_vitdp.jsonl --base-url http://127.0.0.1:18001/v1 \
  --workers 64 > $O/probe_causal_vitdp.log 2>&1 || { log PROBE_FAILED; tail -5 $O/probe_causal_vitdp.log; exit 1; }
$D/tvenv/bin/python $S68/perception_probe.py --score $O/probe_causal_vitdp.jsonl
bash $S43/stop.sh e066x_causal_vitdp 0,1,2,3
log "PROBE_DONE"
