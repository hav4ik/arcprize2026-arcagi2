#!/bin/bash
# E063 on the 4090 box: Manh's fork (L4 replica options) with and without VLLM_COMM_AUTOTUNE.
# Each arm: start the server, wait for it, decode bench at 56 / 16 streams, a prefill bench, greedy ids, stop.
set -u
cd /root/e063
PORT=18300
export E062_MODEL=/root/dsv41/model-e128k
arm() {
  local name=$1 env=$2
  echo "[$(date -u +%H:%M:%S)] arm $name start"
  E062_ENV="VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 VLLM_GPU_DRIVER_RESERVE_MB=512 $env" \
    bash /root/e062/serve.sh manh $PORT --linear-backend humming --cp-kv-cache-interleave-size 64 \
    --cpu-offload-gb 1 --cpu-offload-params embed_tokens
  local pidf=/root/e062/serve_manh_${PORT}.pid log=/root/e062/serve_manh_${PORT}.log
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break
    kill -0 $(cat $pidf) 2>/dev/null || { echo "server died"; break; }
    sleep 10
  done
  cp $log serve_$name.log
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then
    echo "[$(date -u +%H:%M:%S)] $name up"
    /root/manh/venv/bin/python decode_bench.py $PORT 56 2048 512 dec56_$name.json
    /root/manh/venv/bin/python decode_bench.py $PORT 16 2048 512 dec16_$name.json
    /root/manh/venv/bin/python decode_bench.py $PORT 32 8192 8 pre32_$name.json
    /root/manh/venv/bin/python - $PORT ids_$name.json <<'PY'
import json, sys, urllib.request, random
port, out = sys.argv[1], sys.argv[2]
res = []
for seed in range(4):
    rng = random.Random(seed)
    body = {"model": "dsv41", "prompt": [rng.randrange(1000, 100000) for _ in range(1500)],
            "max_tokens": 96, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    c = r["choices"][0]
    res.append(c.get("token_ids") or c["text"])
json.dump(res, open(out, "w"))
PY
    cp $log serve_$name.log
  fi
  kill -- -$(cat $pidf) 2>/dev/null; sleep 20; kill -9 -- -$(cat $pidf) 2>/dev/null
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 5; done
  echo "[$(date -u +%H:%M:%S)] arm $name done"
}
arm default ""
arm autotune "VLLM_COMM_AUTOTUNE=1 VLLM_COMM_AUTOTUNE_REPORT=/root/e063/at"
echo AB_DONE
