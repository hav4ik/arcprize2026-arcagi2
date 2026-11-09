#!/bin/bash
# E068 chain 2 (node 5, 2026-10-30): the already-started kvauto server is identical to ref (--kv-cache-dtype auto ->
# fp8_ds_mla), so it runs the probe at temperature 0.6 / top_p 0.95; then the vitdp and noprefix arms of box_run.sh.
cd /workspace/dsv4
S=repo/experiments/E068_selfhost_image_gap/scripts
for i in $(seq 1 80); do curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo "SERVE_FAILED kvauto"; tail -20 runs/e068_kvauto/server.log; exit 1; }
echo "SERVER_READY ref_dup $(date -u +%H:%M)"
tvenv/bin/python $S/perception_probe.py --out results/e068/probe_ref_t06.jsonl --base-url http://127.0.0.1:18001/v1 \
  --workers 64 --temperature 0.6 --top-p 0.95 > results/e068/probe_ref_t06.log 2>&1
tvenv/bin/python $S/perception_probe.py --score results/e068/probe_ref_t06.jsonl
echo "ARM_DONE ref_t06 $(date -u +%H:%M)"
bash $S/box_run.sh vitdp noprefix
