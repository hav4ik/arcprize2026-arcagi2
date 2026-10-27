#!/usr/bin/env bash
# E056 step a on a dev box with the standard stack: copy the stack venv to venv_candb (the source venv is never
# modified), point the copy's bin/ scripts at the copy (the loop of E051 gpu/setup_offload_venv.sh), install the
# candidate-B overlay through the notebook's own cells (scripts/notebook_install.py: cells 00, 0 and 0b verbatim,
# then cell 2's overlay block verbatim), and check: all 16 sha lines, imports resolve to the copy, the source venv is
# unchanged, E051's 26 CPU tests pass against the installed tree.
#
#   bash /root/e056/gpu/setup_candb_venv.sh            # layout under /root/e056 (README "Box layout")
set -euo pipefail
K=/root/e056
SRC_GIVEN=/root/dsv41/venv
SRC=$(readlink -f "$SRC_GIVEN")
DST=/root/dsv41/venv_candb
SP=lib/python3.12/site-packages
sha() { sha256sum "$1" | cut -c1-64; }
E051_WHEEL=(  # path, sha256 of the fork wheel's original (E051 gpu/install_overlay.sh)
  "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py 04a314743e1e90ab5f285a62268aab9be89c3b17f1888b760b4999389f4f1eff"
  "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py cd43cf5f0c22e4e5616baf9c27a785bac16559d806f21f9e1b7bec056cb97a2d"
  "vllm/v1/kv_offload/cpu/manager.py 3b8b9f6a60f5e69aee1101bd491412cb6a9c9763dfbd027707eeb5605cea192f"
  "vllm/v1/kv_offload/cpu/spec.py 240d686342837804ca34e43ab7d48b52a16f747d737348f5187d7b0e8e0d5a8d"
)
# 1. The source must be the standard stack: the kit overlay's 12 files (377d83e0 list) + the 4 wheel originals.
while read -r h rel; do
  [ "$(sha "$SRC/$SP/$rel")" = "$h" ] || { echo "source venv: $rel is not the kit version"; exit 1; }
done < "$K/kit_base/vllm_overlay_rw.sha256"
for row in "${E051_WHEEL[@]}"; do
  set -- $row
  [ "$(sha "$SRC/$SP/$1")" = "$2" ] || { echo "source venv: $1 is not the wheel original"; exit 1; }
done
(cd "$SRC/$SP" && sha256sum $(awk '{print $2}' "$K/kit/vllm_overlay_rw.sha256")) > "$K/runs/src_venv_before.sha256"
echo "source venv $SRC: 12 kit files + 4 wheel originals ok"
# 2. Copy and rewrite paths (verbatim logic of setup_offload_venv.sh).
[ ! -e "$DST" ] || { echo "$DST exists; remove it or pick another DST"; exit 1; }
cp -a "$SRC" "$DST"
DSTR=$(readlink -f "$DST")
n=0
for f in "$DST"/bin/*; do
  [ -f "$f" ] && [ ! -L "$f" ] || continue
  for s in "$SRC_GIVEN" "$SRC"; do
    if grep -Iq "$s" "$f" 2>/dev/null; then
      sed -i "s#$s/#$DSTR/#g; s#$s\([\"' ]\)#$DSTR\1#g; s#$s\$#$DSTR#g" "$f"; n=$((n + 1))
    fi
  done
done
echo "rewrote paths in $n bin/ files"
if grep -rIl -E "($SRC_GIVEN|$SRC)(/|[\"' ]|$)" "$DST/bin" 2>/dev/null | grep -q .; then
  echo "WARNING: source venv path still referenced in:"; grep -rIl -E "($SRC_GIVEN|$SRC)(/|[\"' ]|$)" "$DST/bin"
fi
head -1 "$DST/bin/vllm"
# 3. The notebook's install path (cells 00 / 0 / 0b verbatim + cell 2's overlay block).
"$DST/bin/python" -I "$K/scripts/notebook_install.py" --notebook "$K/nb/e056-candb-installtest.ipynb" \
  --input "$K/input" --scratch "$K/nbscratch" --site-packages "$DST/$SP" > "$K/runs/notebook_install.log" 2>&1 \
  || { tail -20 "$K/runs/notebook_install.log"; echo "notebook install FAILED"; exit 1; }
tail -1 "$K/runs/notebook_install.log"
# 4. Checks.
while read -r h rel; do
  [ "$(sha "$DST/$SP/$rel")" = "$h" ] || { echo "installed: $rel mismatch"; exit 1; }
done < "$K/kit/vllm_overlay_rw.sha256"
echo "installed: all $(wc -l < "$K/kit/vllm_overlay_rw.sha256") sha lines match"
(cd "$SRC/$SP" && sha256sum $(awk '{print $2}' "$K/kit/vllm_overlay_rw.sha256")) > "$K/runs/src_venv_after.sha256"
cmp -s "$K/runs/src_venv_before.sha256" "$K/runs/src_venv_after.sha256" && echo "source venv unchanged" \
  || { echo "ERROR: source venv changed"; exit 1; }
"$DST/bin/python" -I -c "
import sys, vllm
assert vllm.__file__.startswith(sys.prefix), vllm.__file__
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as s
assert hasattr(s, 'SWA_STORE_CHECKPOINT') and s.__file__.startswith(sys.prefix), s.__file__
print('vllm from', vllm.__file__, '| offload scheduler', s.__file__)"
"$DST/bin/vllm" --version 2>/dev/null | tail -1 || true
rc=0
cd "$K/e051/tests" && PYTHONPATH="$K/e051/tests" CUDA_VISIBLE_DEVICES="" "$DST/bin/python" -m unittest test_dsv41_kv_offload \
  > "$K/runs/e051_cpu_tests.log" 2>&1 || rc=$?
tail -3 "$K/runs/e051_cpu_tests.log"
exit $rc
