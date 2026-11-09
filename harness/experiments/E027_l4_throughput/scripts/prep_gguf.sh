#!/bin/bash
# L4 box: download the GGUF candidates and build stock llama.cpp and the PrismML fork (CUDA, sm_89) in parallel.
mkdir -p /workspace/gguf /root/src /root/arc/logs
HF=/root/vllm-venv/bin/hf
( $HF download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --local-dir /workspace/gguf/bonsai > /root/arc/logs/dl_bonsai.log 2>&1 && echo "bonsai ok" >> /root/arc/logs/prep.log ) &
( $HF download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf --local-dir /workspace/gguf/q4km > /root/arc/logs/dl_q4km.log 2>&1 && echo "q4km ok" >> /root/arc/logs/prep.log ) &
build() {  # name repo
  cd /root/src && rm -rf $1 && git clone -q --depth 1 $2 $1 && cd $1 && git log -1 --format="$1 %h %cd" >> /root/arc/logs/prep.log &&
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF > /root/arc/logs/build_$1.log 2>&1 &&
  cmake --build build -j 48 --target llama-server llama-bench >> /root/arc/logs/build_$1.log 2>&1 && echo "build $1 ok" >> /root/arc/logs/prep.log ||
  echo "build $1 FAILED" >> /root/arc/logs/prep.log
}
build llamacpp https://github.com/ggml-org/llama.cpp &
build prismllama https://github.com/PrismML-Eng/llama.cpp &
wait
df -h / | tail -1 >> /root/arc/logs/prep.log
echo PREP_DONE >> /root/arc/logs/prep.log
