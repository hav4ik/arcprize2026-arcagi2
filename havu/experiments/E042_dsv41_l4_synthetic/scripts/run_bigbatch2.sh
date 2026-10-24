#!/bin/bash
# E042 S4 resume: the first C=128 point stalled at 100 streams (aiohttp's default 100-connection cap in
# decode_window.py, now lifted). Re-run C=128 and the 8K point on the running s4_c128 server, then stop it.
S=/root/dsv41/scripts; L=/root/dsv41/runs/bigbatch.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
log "resume: 1024:128 (connection cap fixed) + 8192:64"
$S/dw_grid.sh s4_c128 "1024:128 8192:64"
$S/stop.sh s4_c128 | tee -a $L
log "BIGBATCH_DONE (server stopped; box idle)"
