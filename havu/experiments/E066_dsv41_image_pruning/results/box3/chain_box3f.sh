#!/bin/bash
# E066 box 3 (user, 08:20 PDT): take over from chain_box3e once its f72_124 probe is done: probe the front-runner
# f72_120 first, then the average-112 dev-loss points of every group, then the remaining probes.
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
M=sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1
D=/workspace/dsv4/data/e066
until grep -q "GEN f72_124 done" results_chain_box3e.log; do sleep 2; done
for c in $(ps -eo pid,ppid,args | awk "\$2==15954 {print \$1}"); do kill -TERM $c; done; kill -TERM 15954; sleep 2
for c in $(ps -eo pid,args | awk "/gen_probe_mm.py/ && !/awk/ {print \$1}"); do :; done
E="\"$(realpath $D/engram/e043tool_top64gib.npz)\""
probe() {  # tag mask: image and text sets in parallel
  echo "{\"engram_keep\": $E, \"expert_keep\": \"$(realpath $2)\", \"observe\": null}" > /dev/shm/e043/ctrlB.json.tmp && mv /dev/shm/e043/ctrlB.json.tmp /dev/shm/e043/ctrlB.json
  curl -s -m 300 http://127.0.0.1:18001/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"dsv41\",\"prompt\":[0,5,6,7],\"max_tokens\":1}" > /dev/null
  tvenv/bin/python $S/gen_probe_mm.py --tok $(ls -d $D/tok_img_dev_s/shard*) --parquet $D/e065_image.parquet --n-seq 25 --seed 0 \
    --max-tokens 16000 --workers 25 --tag $1 --out results/gen_probe.jsonl &
  tvenv/bin/python $S/gen_probe_mm.py --tok $D/tok_txt_dev --parquet $D/e065_text.parquet --n-seq 25 --seed 0 \
    --max-tokens 16000 --workers 25 --tag $1 --out results/gen_probe.jsonl &
  wait; echo "GEN $1 done"
}
probe f72_120 ${M}_calblend_f72c160s8_avg120.npy
bash $S/nll_masks_mm.sh results/c3_screen.jsonl u112:${M}_uniform_avg112.npy f72_112:${M}_calblend_f72c160s8_avg112.npy \
  f72c224_112:${M}_calblend_f72c224s8_avg112.npy f64c224_112:${M}_calblend_f64c224s8_avg112.npy
echo S112_DONE
probe f96_120 ${M}_calblend_f96c160s8_avg120.npy
probe f72c224_120 ${M}_calblend_f72c224s8_avg120.npy
echo "{\"engram_keep\": $E, \"expert_keep\": null, \"observe\": null}" > /dev/shm/e043/ctrlB.json
echo ALL_DONE
