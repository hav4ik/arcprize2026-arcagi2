#!/usr/bin/env bash
# One-step E051 install on a box or pod that has the standard stack (fork wheel + 12-file kit overlay in SRC):
# copy the venv to DST, point the copy's bin/ scripts at the copy, install the 4 offload files (sha-checked), run the
# 26 CPU tests against the installed tree. The original venv is never modified.
#
#   bash setup_offload_venv.sh [SRC=/root/dsv41/venv] [DST=/root/dsv41/venv_offload]
#   OVERLAY=v2 bash setup_offload_venv.sh ...   # overlay v2 (incremental update_offload_keys); default v1
# Needs this kit at /root/e051 (overlay/, gpu/, tests/). About 1-3 min (cp of ~8 GB + tests).
set -euo pipefail
SRC=$(readlink -f "${1:-/root/dsv41/venv}")
DST=${2:-/root/dsv41/venv_offload}
K=/root/e051
case "${OVERLAY:-v1}" in
  v1) INSTALLER=install_overlay.sh; OV=overlay; TESTS="test_dsv41_kv_offload" ;;
  v2) INSTALLER=install_overlay_v2.sh; OV=overlay_v2; TESTS="test_dsv41_kv_offload test_dsv41_kv_offload_v2" ;;
  *) echo "OVERLAY must be v1 or v2"; exit 1 ;;
esac
[ -x "$SRC/bin/python" ] || { echo "no venv at $SRC"; exit 1; }
[ ! -e "$DST" ] || { echo "$DST exists; remove it or pick another DST"; exit 1; }
OFF="distributed/kv_transfer/kv_connector/v1/offloading/config.py distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py v1/kv_offload/cpu/manager.py v1/kv_offload/cpu/spec.py"
SRCSITE=$("$SRC/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
src_shas() { for f in $OFF; do sha256sum "$SRCSITE/vllm/$f"; done; }
BEFORE=$(src_shas)
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
bash "$K/gpu/$INSTALLER" "$DST"
"$DST/bin/python" -I -c "import vllm, sys; assert vllm.__file__.startswith(sys.prefix), vllm.__file__; print('vllm from', vllm.__file__)"
# The source venv must be untouched (it may legitimately carry an earlier overlay, e.g. venv_candb has v1).
[ "$(src_shas)" = "$BEFORE" ] || { echo "ERROR: $SRC offload files changed during setup"; exit 1; }
echo "source venv offload files unchanged"
cd "$K/tests" && PYTHONPATH="$K/tests" CUDA_VISIBLE_DEVICES="" "$DST/bin/python" -m unittest $TESTS 2>&1 | tail -3
