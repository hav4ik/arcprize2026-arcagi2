#!/bin/bash
# E066 box 3 (user, 08:05 PDT): floor 64 / cap 224 group first. Dev-loss screen (120, 124, 116, 128), its
# free-generation probe at 120/124 plus floor 72 at 124, then the paused probes (f96_120, f72_120, f72c224_120).
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
M=sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1
D=/workspace/dsv4/data/e066
bash $S/nll_masks_mm.sh results/c3_screen.jsonl f64c224_120:${M}_calblend_f64c224s8_avg120.npy \
  f64c224_124:${M}_calblend_f64c224s8_avg124.npy f64c224_116:${M}_calblend_f64c224s8_avg116.npy \
  f64c224_128:${M}_calblend_f64c224s8_avg128.npy
echo F64C224_SCREEN_DONE
E="\"$(realpath $D/engram/e043tool_top64gib.npz)\""
for spec in f64c224_120:${M}_calblend_f64c224s8_avg120.npy f64c224_124:${M}_calblend_f64c224s8_avg124.npy \
            f72_124:${M}_calblend_f72c160s8_avg124.npy f96_120:${M}_calblend_f96c160s8_avg120.npy \
            f72_120:${M}_calblend_f72c160s8_avg120.npy f72c224_120:${M}_calblend_f72c224s8_avg120.npy; do
  IFS=: read -r tag exp <<< "$spec"
  echo "{\"engram_keep\": $E, \"expert_keep\": \"$(realpath $exp)\", \"observe\": null}" > /dev/shm/e043/ctrlB.json.tmp && mv /dev/shm/e043/ctrlB.json.tmp /dev/shm/e043/ctrlB.json
  curl -s -m 300 http://127.0.0.1:18001/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"dsv41\",\"prompt\":[0,5,6,7],\"max_tokens\":1}" > /dev/null
  tvenv/bin/python $S/gen_probe_mm.py --tok $(ls -d $D/tok_img_dev_s/shard*) --parquet $D/e065_image.parquet --n-seq 25 --seed 0 \
    --max-tokens 16000 --workers 25 --tag $tag --out results/gen_probe.jsonl &
  tvenv/bin/python $S/gen_probe_mm.py --tok $D/tok_txt_dev --parquet $D/e065_text.parquet --n-seq 25 --seed 0 \
    --max-tokens 16000 --workers 25 --tag $tag --out results/gen_probe.jsonl &
  wait
  echo "GEN $tag done"
done
echo "{\"engram_keep\": $E, \"expert_keep\": null, \"observe\": null}" > /dev/shm/e043/ctrlB.json
echo ALL_DONE
