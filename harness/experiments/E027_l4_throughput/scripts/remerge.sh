#!/bin/bash
# Fix: merge the per-GPU tuned configs with glob.escape (the "[128,128]" in the names broke the glob), install,
# re-bench (fp8_tuned2), then the vLLM 0.31 baseline probe.
cd /root/arc
trap - INT
for p in $(ps -eo pid,args | awk '/[c]hain_retune.sh/ {print $1}'); do kill $p; done
for p in $(pgrep -f 'bench_decode.py|bench_decode2.py|vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
for p in $(pgrep -f 'VLLM::|vllm.v1.engine'); do kill -9 $p; done; sleep 5
python3 - <<'EOF'
import glob, json, os
base = "/root/arc/fp8_configs2"; out = base + "/merged"; os.makedirs(out, exist_ok=True)
names = sorted({os.path.basename(f) for f in glob.glob(base + "/gpu*/*.json")})
for n in names:
    merged = {}
    for d in sorted(glob.glob(base + "/gpu*")):
        f = os.path.join(d, n)
        if os.path.exists(f):
            merged.update(json.load(open(f)))
    merged = {k: merged[k] for k in sorted(merged, key=int)}
    assert merged, n
    json.dump(merged, open(os.path.join(out, n), "w"), indent=4)
    print(n[:16], "batch sizes:", list(merged))
EOF
CFG=$(/root/vllm-venv/bin/python -c "import vllm, os; print(os.path.dirname(vllm.__file__) + '/model_executor/layers/quantization/utils/configs')")
cp /root/arc/fp8_configs2/merged/*.json $CFG/ && echo "installed merged configs"
N=fp8_tuned2; rm -rf runs/$N; mkdir -p runs/$N
MODEL=Qwen/Qwen3.8-27B-FP8 VENV=/root/vllm-venv setsid nohup ./start_vllm.sh 0 --no-enable-prefix-caching --language-model-only \
  > runs/$N/server.log 2>&1 < /dev/null &
until curl -s -m 5 localhost:8001/v1/models > /dev/null; do sleep 15; grep -q Traceback runs/$N/server.log && { echo "$N FAILED"; break; }; done
echo "$N config-missing warnings: $(grep -c 'Config file not found' runs/$N/server.log)"
/opt/sglang/bin/python bench_decode2.py http://127.0.0.1:8001 ${N}_prefill runs/$N/prefill.json 4000 16 16 > runs/$N/prefill.log 2>&1
echo "prefill2: $(/opt/sglang/bin/python -c "
import json; r = json.load(open('runs/$N/prefill.json'))[0]; print(round(r['concurrency'] * r['prompt_tok'] / r['wall_s']), 'tok/s')")"
/opt/sglang/bin/python bench_decode.py http://127.0.0.1:8001 $N runs/$N/$N.json 8 16 32 48 > runs/$N/bench.log 2>&1
cat runs/$N/bench.log
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill $p; done; sleep 30
echo RETUNE_DONE
unset NCCL_P2P_LEVEL
VENV=/root/vllmnew ATTN=TRITON_ATTN MTP=0 ./probe2.sh p31_base_align_c24 Qwen/Qwen3.8-27B-FP8 24 --language-model-only \
  --mamba-cache-mode align > base_probe.log 2>&1
echo RETUNE_CHAIN_DONE
