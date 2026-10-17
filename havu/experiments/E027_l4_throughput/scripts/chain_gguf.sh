#!/bin/bash
# L4 box: data-parallel llama.cpp decode benches (4 servers, 12 slots each, q8_0 KV), same prompts as bench_decode.py.
cd /root/arc
trap - INT
run() {  # label build model ctx
  ./serve_llama.sh $2 $3 12 $4 q8_0 $1 > runs/llama/$1.serve.log 2>&1
  cat runs/llama/$1.serve.log
  if grep -q FAILED runs/llama/$1.serve.log; then ./stop_llama.sh; return; fi
  grep -hE "llama_kv_cache|recurrent|CUDA0 (model|KV|compute) buffer" runs/llama/$1_gpu0.log | head -8
  /opt/sglang/bin/python bench_dp.py http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104 \
    $1 runs/llama/$1.json 8 16 32 48 > runs/llama/$1.bench.log 2>&1
  cat runs/llama/$1.bench.log
  ./stop_llama.sh
}
run q4km /root/src/llamacpp/build /workspace/gguf/q4km/Qwen3.8-27B-UD-Q4_K_M.gguf 73728
run bonsai /root/src/prismllama/build /workspace/gguf/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf 73728
echo GGUF_DONE
