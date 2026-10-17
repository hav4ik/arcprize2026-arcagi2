#!/bin/bash
# Re-tune: the first tuning run let all 4 GPU workers write the same per-shape files, so only the last writer's batch
# sizes (256-2048) survived. Here each GPU writes to its own folder and the files are merged per shape.
# After the running bench (TUNE_DONE): re-tune, merge, install, re-bench (fp8_tuned2), then the vLLM 0.31 baseline probe.
cd /root/arc
trap - INT
for p in $(ps -eo pid,args | awk '/[c]hain_l4_next2.sh/ {print $1}'); do kill $p; done   # stop it before its baseline probe
until grep -q TUNE_DONE chain_tune.log 2>/dev/null; do sleep 30; done
B=/root/src/vllm_bench
python3 - <<'EOF'
src = open("/root/src/vllm_bench/tune_qwen38.py").read()
old = "    save_path = args.save_path\n"
new = "    save_path = os.path.join(args.save_path, f\"gpu{gpu_id}\")\n    os.makedirs(save_path, exist_ok=True)\n"
assert src.count(old) == 1
src = src.replace(old, new, 1)
if "\nimport os\n" not in src:
    src = "import os\n" + src
open("/root/src/vllm_bench/tune_qwen38_pergpu.py", "w").write(src)
EOF
rm -rf /root/arc/fp8_configs2 && mkdir -p /root/arc/fp8_configs2
t0=$(date +%s)
timeout 10800 /root/vllm-venv/bin/python $B/tune_qwen38_pergpu.py --tp-size 4 --out-dtype bfloat16 --save-path /root/arc/fp8_configs2 \
  > /root/arc/logs/tune_fp8_2.log 2>&1
echo "retune rc=$? in $(( $(date +%s) - t0 ))s"
python3 - <<'EOF'
import glob, json, os
out = "/root/arc/fp8_configs2/merged"; os.makedirs(out, exist_ok=True)
names = sorted({os.path.basename(f) for f in glob.glob("/root/arc/fp8_configs2/gpu*/*.json")})
for n in names:
    merged = {}
    for f in sorted(glob.glob(f"/root/arc/fp8_configs2/gpu*/{n}")):
        merged.update(json.load(open(f)))
    merged = {k: merged[k] for k in sorted(merged, key=int)}
    json.dump(merged, open(f"{out}/{n}", "w"), indent=4)
    print(n[:16], "batch sizes:", list(merged))
EOF
CFG=$(/root/vllm-venv/bin/python -c "import vllm, os; print(os.path.dirname(vllm.__file__) + '/model_executor/layers/quantization/utils/configs')")
cp /root/arc/fp8_configs2/merged/*.json $CFG/ && echo "installed merged configs"
N=fp8_tuned2; mkdir -p runs/$N
MODEL=Qwen/Qwen3.8-27B-FP8 VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
  > runs/$N/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback runs/$N/server.log && { echo "$N FAILED"; break; }; done
echo "$N config-missing warnings: $(grep -c 'Config file not found' runs/$N/server.log)"
/opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
echo "prefill: $(/opt/sglang/bin/python -c "
import json; r = json.load(open('runs/$N/prefill.json'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']), 'tok/s')")"
/opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
cat runs/$N/bench.log
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
echo RETUNE_DONE
unset NCCL_P2P_LEVEL
VENV=/root/vllmnew ATTN=TRITON_ATTN MTP=0 ./probe2.sh p31_base_align_c24 Qwen/Qwen3.8-27B-FP8 24 --language-model-only \
  --mamba-cache-mode align > base_probe.log 2>&1
echo RETUNE_CHAIN_DONE
