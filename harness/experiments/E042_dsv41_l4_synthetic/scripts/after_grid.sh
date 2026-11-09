#!/bin/bash
until grep -q GRID_DONE /root/dsv41/runs/s1_full/dw/grid.log 2>/dev/null; do sleep 15; done
sleep 10
exec /root/dsv41/scripts/run_followups.sh
