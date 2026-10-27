#!/usr/bin/env bash
# One-step E051 install on a box or pod that has the standard stack (fork wheel + 12-file kit overlay in SRC):
# copy the venv to DST, point the copy's bin/ scripts at the copy, install the 4 offload files (sha-checked), run the
# 26 CPU tests against the installed tree. The original venv is never modified.
#
#   bash setup_offload_venv.sh [SRC=/root/dsv41/venv] [DST=/root/dsv41/venv_offload]
# Needs this kit at /root/e051 (overlay/, gpu/, tests/). About 1-3 min (cp of ~8 GB + tests).
set -euo pipefail
SRC=$(readlink -f "${1:-/root/dsv41/venv}")
DST=${2:-/root/dsv41/venv_offload}
K=/root/e051
[ -x "$SRC/bin/python" ] || { echo "no venv at $SRC"; exit 1; }
[ ! -e "$DST" ] || { echo "$DST exists; remove it or pick another DST"; exit 1; }
cp -a "$SRC" "$DST"
DSTR=$(readlink -f "$DST")
# A copied venv keeps the source path in every bin/ script's shebang (and in activate*), so the copy's `vllm` would
# run the source venv's (unpatched) packages. Rewrite both the given and the resolved source paths.
n=0
for f in "$DST"/bin/*; do
  [ -f "$f" ] && [ ! -L "$f" ] || continue
  for s in "${1:-/root/dsv41/venv}" "$SRC"; do
    if grep -Iq "$s" "$f" 2>/dev/null; then
      sed -i "s#$s/#$DSTR/#g; s#$s\([\"' ]\)#$DSTR\1#g; s#$s\$#$DSTR#g" "$f"; n=$((n + 1))
    fi
  done
done
echo "rewrote paths in $n bin/ files"
if grep -rIl -E "(${1:-/root/dsv41/venv}|$SRC)(/|[\"' ]|$)" "$DST/bin" 2>/dev/null | grep -q .; then
  echo "WARNING: source venv path still referenced in:"; grep -rIl -E "(${1:-/root/dsv41/venv}|$SRC)(/|[\"' ]|$)" "$DST/bin"
fi
head -1 "$DST/bin/vllm"
bash "$K/gpu/install_overlay.sh" "$DST"
"$DST/bin/python" -I -c "import vllm, sys; assert vllm.__file__.startswith(sys.prefix), vllm.__file__; print('vllm from', vllm.__file__)"
# The source venv must be untouched.
for f in distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py v1/kv_offload/cpu/spec.py; do
  if cmp -s "$SRC/lib/python3.12/site-packages/vllm/$f" "$K/overlay/vllm/$f"; then echo "ERROR: $SRC is patched ($f)"; exit 1; fi
done
cd "$K/tests" && PYTHONPATH="$K/tests" CUDA_VISIBLE_DEVICES="" "$DST/bin/python" -m unittest test_dsv41_kv_offload 2>&1 | tail -3
