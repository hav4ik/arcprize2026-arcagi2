#!/bin/bash
# E066 FP4 Engram pilot (teammate's idea, user 2026-10-29): the candidate 1 export served with MXFP4-rounded Engram
# rows (export/c1_fp4: fp4_engram_chunked.py, everything else hardlinked from c1_gptq) vs the same export with the
# shipped FP8 Engram (c1x, measured at parity): paired dev NLL on the same 64 rollouts (results/q_screen.jsonl).
# Never fails the caller: prints FP4_PILOT_* and returns 0.
D=/workspace/dsv4
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
CTRL=/dev/shm/e043/ctrlB.json
SCH13='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
cd $D
grep -q FP4_EXPORT_READY results_chain_fp4.log || { echo FP4_PILOT_SKIPPED: FP4 export not ready; exit 0; }
echo '{"engram_keep": null, "expert_keep": null, "observe": null}' > $CTRL
DSV41_EXPERT_HUMMING="$SCH13" DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$D/export/c1_fp4/gptq \
  VISION=1 GPUS=0,1,2,3 TP=4 EP=0 PORT=18001 CTRL=$CTRL UTIL=0.86 SEQS=192 MBT=8192 MAXLEN=262144 \
  MODEL=$D/export/c1_fp4 bash $S43/serve.sh c1fp4 --limit-mm-per-prompt '{"image": 64}'
for i in $(seq 1 100); do
  curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break
  grep -q Traceback runs/c1fp4/server.log 2>/dev/null && break
  sleep 15
done
if curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41; then
  ENGRAM=none bash $S/nll_masks_mm.sh results/q_screen.jsonl c1fp4:none > logs/fp4_pilot_nll.log 2>&1
  venv/bin/python - <<'PY'
import json
by = {}
for l in open("results/q_screen.jsonl"):
    r = json.loads(l); by.setdefault(r["tag"], {})[(r["batch"], r["task_id"], r["attempt"])] = r
for h, name in (("e065_", "image"), ("e065t_", "text")):
    for fld in ("gen", "blk"):
        ks = [k for k in by["c1x"] if k[0].startswith(h) and k in by.get("c1fp4", {})]
        a = sum(by["c1fp4"][k][f"nll_{fld}"] for k in ks) / sum(by["c1x"][k][f"n_{fld}"] for k in ks)
        b = sum(by["c1x"][k][f"nll_{fld}"] for k in ks) / sum(by["c1x"][k][f"n_{fld}"] for k in ks)
        print(f"FP4_PILOT {name} {'generated' if fld == 'gen' else 'tool-call'} n={len(ks)} FP8 {b:.5f} FP4 {a:.5f} diff {a - b:+.5f} ({100 * (a / b - 1):+.2f}%)")
PY
else
  echo "FP4_PILOT_SERVE_FAILED"; tail -15 runs/c1fp4/server.log
fi
bash $S43/stop.sh c1fp4 0,1,2,3 > /dev/null 2>&1
echo FP4_PILOT_DONE
