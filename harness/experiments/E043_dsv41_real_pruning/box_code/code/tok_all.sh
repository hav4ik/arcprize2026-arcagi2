#!/bin/bash
# Install the tools venv, then tokenize all calibration rollouts (32 shards) and the eval traces.
cd /workspace/dsv4
until uv pip install --python tvenv/bin/python numpy pyarrow pandas tokenizers sympy > logs/tvenv.log 2>&1; do sleep 10; done
M=/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017
R=data/rollouts_ds/rollouts
tvenv/bin/python code/scripts/tokenize_rollouts.py --model-dir $M --parquet data/e030_s20x60_hdrmin_eval.parquet --out data/tok_eval > logs/tok_eval.log 2>&1 &
for i in $(seq 0 31); do
  tvenv/bin/python code/scripts/tokenize_rollouts.py --model-dir $M --parquet $R/e031_arc2_train.parquet $R/e031_nvarc_b0.parquet $R/e036_nvarc_b1.parquet $R/e038_openrouter_deepinfra_b1r.parquet --out data/tok/shard$i --shard $i/32 > logs/tok_$i.log 2>&1 &
done
wait
echo TOK_DONE > logs/tok_done
