#!/bin/bash
# E066 box 1, after the unmasked C1 eval was stopped (2026-10-29 16:35 UTC):
#  1. C1 export served from its own files: dev NLL parity vs the runtime-mask 2-bit server (c1q), then its HF upload
#     in the background;
#  2. capture of the per-layer 72-152 candidate (native server), 2-bit GPTQ, its runtime-mask eval server, dev NLL;
#  3. its export (--per-layer) + content check + HF upload in the background, beside the evals (run from the local
#     machine against serve_eval.sh's server). Needs $HF_TOKEN in the environment.
set -u
D=/workspace/dsv4
cd $D
S=repo/experiments/E066_dsv41_image_pruning/scripts
S43=repo/experiments/E043_dsv41_real_pruning/scripts
R1=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit
R2=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit
bash $S/serve_eval.sh c1x || exit 1
ENGRAM=none bash $S/nll_masks_mm.sh results/q_screen.jsonl c1x:none
if venv/bin/python - <<'PY'
import json, sys
by = {}
for l in open("results/q_screen.jsonl"):
    r = json.loads(l); by.setdefault(r["tag"], {})[(r["batch"], r["task_id"], r["attempt"])] = r
for h in ("e065_", "e065t_"):
    v = {}
    for t in ("c1q", "c1x"):
        ks = [k for k in by[t] if k[0].startswith(h)]
        v[t] = sum(by[t][k]["nll_gen"] for k in ks) / sum(by[t][k]["n_gen"] for k in ks)
    print(h, {t: round(x, 5) for t, x in v.items()}, "diff", round(v["c1x"] - v["c1q"], 5))
    if abs(v["c1x"] - v["c1q"]) > 0.003:
        sys.exit(1)
PY
then
  echo PARITY_OK c1
  (bash $S/upload_candidate.sh c1 $R1 c1q c1x > logs/upload_c1.log 2>&1; tail -1 logs/upload_c1.log) &
else
  echo PARITY_FAIL c1: no upload
fi
bash $S43/stop.sh c1x 0,1,2,3
VISION=1 GPUS=0,1,2,3 TP=4 PORT=18001 CTRL=/dev/shm/e043/ctrlB.json MODEL=$D/DeepSeek-V4.1-Flash-1017 bash $S43/serve.sh imgC
for i in $(seq 1 100); do curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo SERVE_FAILED imgC; tail -20 runs/imgC/server.log; exit 1; }
echo NATIVE_READY
bash $S/gptq_candidate.sh dump e72c152 cand/e72c152_keep.npy || { echo DUMP_FAILED; exit 1; }
bash $S43/stop.sh imgC 0,1,2,3
bash $S/gptq_candidate.sh quant e72c152 cand/e72c152_keep.npy || { echo QUANT_FAILED; exit 1; }
bash $S/serve_eval.sh e72q || exit 1
bash $S/nll_masks_mm.sh results/q_screen.jsonl e72q:cand/e72c152_keep.npy
last=$(grep "ctrl loaded" runs/e72q/server.log | tail -1)
echo "$last" | grep -q "experts=$D/cand/e72c152_keep.npy" || { echo "CTRL_WRONG after NLL: $last"; exit 1; }
echo E72Q_READY
bash $S/export_candidate.sh export e72c152 cand/e72c152_keep.npy gptq/e72c152 \
  "per-layer 72-152 (avg 120) 2-bit GPTQ, tool use with images (E066)" \
  "tool + image keep set, per-layer counts (floor 72, cap 152, step 16): E065 image and text harness rollouts, class weights 1/1/10/1 (layers 0-19), 0/1/10/1 (20-39)" \
  --per-layer || { echo EXPORT_FAILED e72c152; wait; exit 1; }
bash $S/export_candidate.sh check e72c152 cand/e72c152_keep.npy gptq/e72c152 || { echo CHECK_FAILED e72c152; wait; exit 1; }
bash $S/upload_candidate.sh e72c152 $R2 e72q > logs/upload_e72c152.log 2>&1; tail -1 logs/upload_e72c152.log
wait
echo MAIN_ALL_DONE
