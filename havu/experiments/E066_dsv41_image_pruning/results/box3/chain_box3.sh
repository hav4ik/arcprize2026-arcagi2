#!/bin/bash
# E066 box 3: wait for checkpoint + env, native server with vision, candidate 3 dev-loss screen, free-generation
# probe, candidate 2 GPTQ capture. Masks in sel_box3 (computed from the same statistics as box 1; C1/C2 hashes equal).
set -u
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
S43=repo/experiments/E043_dsv41_real_pruning/scripts
M=sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1
until grep -q SETUP_DONE setup_env.log && grep -q Downloaded dl.log; do sleep 30; done
n=$(find DeepSeek-V4.1-Flash-1017 -name "*.safetensors" | wc -l); echo "safetensors files: $n"
mkdir -p /dev/shm/e043 results logs
echo "{\"engram_keep\": null, \"expert_keep\": null, \"observe\": null}" > /dev/shm/e043/ctrlB.json
VISION=1 GPUS=0,1,2,3 TP=4 EP=1 PORT=18001 CTRL=/dev/shm/e043/ctrlB.json MAXLEN=262144 \
  MODEL=/workspace/dsv4/DeepSeek-V4.1-Flash-1017 bash $S43/serve.sh nat3 --limit-mm-per-prompt "{\"image\": 64}"
for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo SERVE_FAILED; tail -30 runs/nat3/server.log; exit 1; }
echo NAT3_READY
bash $S/nll_masks_mm.sh results/c3_screen.jsonl c1:${M}_uniform_avg128.npy \
  f72_120:${M}_calblend_f72c160s8_avg120.npy f72_124:${M}_calblend_f72c160s8_avg124.npy \
  f72c224_120:${M}_calblend_f72c224s8_avg120.npy f72c224_124:${M}_calblend_f72c224s8_avg124.npy \
  f72_116:${M}_calblend_f72c160s8_avg116.npy f72_128:${M}_calblend_f72c160s8_avg128.npy
echo SCREEN_DONE
MODE=gen NSEQ=25 bash $S/nll_masks_mm.sh results/gen_probe.jsonl full:none c1:${M}_uniform_avg128.npy \
  f96_120:${M}_calblend_f96c160s8_avg120.npy f72_120:${M}_calblend_f72c160s8_avg120.npy \
  f72c224_120:${M}_calblend_f72c224s8_avg120.npy
echo GEN_DONE
mkdir -p cand && cp ${M}_calblend_f96c160s8_avg118.npy cand/c2_keep.npy
bash $S/gptq_candidate.sh dump c2 cand/c2_keep.npy && echo C2_DUMP_DONE
echo CHAIN3_DONE
