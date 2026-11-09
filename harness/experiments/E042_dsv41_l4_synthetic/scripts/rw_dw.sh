#!/bin/bash
# E042 RW (Runpod, 16:00): the user: "I need to quickly measure the TPS of this E128 model on 4XL4" (no DSpark).
# decode_window.py on the real tool E128 2-bit server (rw_e128t2), the same points as the dummy-weight ds_x128_u93
# (8K x 32 334.9, 8K x 64 487.5 tok/s), then 8K x 128. Log: /root/dsv41/runs/rw.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/rw.log; PY=/root/dsv41/venv/bin/python; R=/root/dsv41/runs/rw_e128t2
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
for c in 32 64 128; do
  log "rw_e128t2 ctx 8192 C $c: $(timeout 1800 $PY $S/decode_window.py --model dsv41 --max-model-len 163840 --ctx 8192 --conc $c --window 60 --seed $c --json $R/dw_8192_c$c.json 2>&1 | tail -1 | cut -c1-700)"
done
log "rw_e128t2 health: $($PY $S/run_health.py $R 2>&1 | cut -c1-500)"
log "RW_DW_DONE"
