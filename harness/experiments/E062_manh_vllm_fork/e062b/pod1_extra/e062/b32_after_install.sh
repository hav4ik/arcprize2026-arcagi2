#!/bin/bash
# E062 (on the box): after E042's box_install.sh, finish B32's stack: podprep (kit, 12-file overlay, tool venv, data),
# then B32's 16-file overlay (E056 kit) over /root/dsv41/venv with every sha checked, then /root/kit_b = kit + the
# pinned_candb files at their kit paths (as the B32 notebook's cell 0b stages them).
set -uo pipefail
for _ in $(seq 360); do
  grep -q "INSTALL OK" /root/dsv41/install.log 2>/dev/null && break
  grep -qE "^(ERROR|Traceback)|exit 1|FAILED" /root/dsv41/install.log 2>/dev/null && { echo "install failed"; exit 1; }
  sleep 10
done
grep -q "INSTALL OK" /root/dsv41/install.log || { echo "install not done after 1 h"; exit 1; }
set -e
bash /root/podkit/podprep.sh
SPK=/root/dsv41/venv/lib/python3.12/site-packages
tar xzf /root/e062/b32_overlay.tgz -C $SPK
while read -r h rel; do echo "$h  $SPK/$rel" | sha256sum -c --quiet -; done < /root/e062/b32_overlay.sha256
echo "B32 16-file overlay ok"
rm -rf /root/kit_b && cp -a /root/kit /root/kit_b
cp /root/e062/pinned_candb/solver_dsv41.py /root/kit_b/solver_dsv41.py
for f in run_deepseek_tools.py train_check.py auto_repair.py auto_repair_names.json; do
  cp /root/e062/pinned_candb/$f /root/kit_b/repo/evals/$f; done
sha256sum /root/kit_b/solver_dsv41.py /root/kit_b/repo/evals/{run_deepseek_tools,train_check,auto_repair}.py | cut -c1-8,65-
/root/dsv41/venv/bin/python -I -c "import vllm, sys; print('vllm', vllm.__version__, vllm.__file__)"
echo "B32 STACK OK"
