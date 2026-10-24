#!/bin/bash
# E042: steady-decode grid (decode_window.py) against the running server. Usage: dw_grid.sh <run_name> "<ctx:conc ...>"
R=/root/dsv41/runs/$1/dw; mkdir -p $R
for p in $2; do
  ctx=${p%%:*}; c=${p##*:}
  echo "[$(date -u +%H:%M:%S)] $(/root/dsv41/venv/bin/python /root/dsv41/scripts/decode_window.py --ctx $ctx --conc $c --window 30 --seed $c --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1)" >> $R/grid.log
done
echo "GRID_DONE" >> $R/grid.log
