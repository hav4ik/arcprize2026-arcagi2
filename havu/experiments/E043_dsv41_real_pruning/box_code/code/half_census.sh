#!/bin/bash
# Data-sufficiency check: census on half / quarter of the calibration shards, same budgets, held-out coverage.
cd /workspace/dsv4
for frac in half quarter; do
  if [ $frac = half ]; then dirs=$(ls -d data/tok/shard{0..15}); else dirs=$(ls -d data/tok/shard{0..7}); fi
  venv/bin/python code/scripts/engram_census.py count --tok $dirs --workers 8 --out data/engram/counts_$frac.npz
  venv/bin/python code/scripts/engram_census.py select --counts data/engram/counts_$frac.npz --budget-gib 16 32 64 96 112 --out-dir data/engram/masks_$frac
  venv/bin/python code/scripts/engram_census.py coverage --masks data/engram/masks_$frac/engram_top{16,32,64,96,112}gib.npz --tok data/tok_eval
done
