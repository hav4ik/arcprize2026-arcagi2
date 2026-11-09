#!/bin/bash
# E042: TPS sweep against a running serve.sh server with vLLM's own `vllm bench serve`.
# Usage: bench.sh <run_name> <input_len> <output_len> <concurrency...>
# Random token prompts, ignore_eos, temperature 0, prefix caching off on the server.
# Each concurrency C sends max(2C, 4) prompts (an initial batch plus one replacement wave).
set -u
NAME=$1; IN=$2; OUT=$3; shift 3
R=/root/dsv41/runs/$NAME; mkdir -p $R/bench
export PATH=/root/dsv41/venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for C in "$@"; do
  N=$((2 * C)); [ $N -lt 4 ] && N=4
  tag=in${IN}_out${OUT}_c${C}
  vllm bench serve --backend vllm --base-url http://127.0.0.1:${PORT:-8100} --endpoint /v1/completions \
    --model dsv41-e64 --tokenizer /root/dsv41/model-e64 --trust-remote-code \
    --dataset-name random --random-input-len $IN --random-output-len $OUT --random-range-ratio 0 \
    --ignore-eos --temperature 0 --num-prompts $N --max-concurrency $C --seed $C \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 \
    --save-result --result-dir $R/bench --result-filename $tag.json > $R/bench/$tag.log 2>&1
  python3 - $R/bench/$tag.json $C <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"c{sys.argv[2]}: FAILED ({e})"); sys.exit()
c = int(sys.argv[2]); tpot = d.get("mean_tpot_ms") or 0
print(f"c{c}: done {d['completed']}/{d['num_prompts']} out_tok/s {d['output_throughput']:.1f} "
      f"| TTFT mean {d['mean_ttft_ms']:.0f} ms p95 {d.get('p95_ttft_ms', 0):.0f} | TPOT mean {tpot:.1f} ms "
      f"p95 {d.get('p95_tpot_ms', 0):.1f} -> decode {1000 / tpot * c if tpot else 0:.1f} tok/s ({1000 / tpot if tpot else 0:.1f}/stream) "
      f"| {d['duration']:.0f} s")
PY
done | tee -a $R/bench/summary_in${IN}_out${OUT}.txt
