#!/usr/bin/env bash
# E057b: build B+'s venv from candidate B's: copy /root/dsv41/venv_candb to /root/dsv41/venv_candbp (the source is
# never modified), point the copy's bin/ scripts at the copy (the loop of E051 gpu/setup_offload_venv.sh and E056
# gpu/setup_candb_venv.sh), install E051 overlay v2 (gpu/install_overlay_v2.sh: only offloading/scheduler.py changes,
# dffd7d86 -> 94b06f85), then check: the 16-file list with the scheduler replaced, imports from the copy, the source
# venv unchanged, and E051's CPU tests (test_dsv41_kv_offload + test_dsv41_kv_offload_v2) against the copy.
#   bash /root/e057b/scripts/setup_candbp_venv.sh > /root/e057b/runs/setup_candbp_venv.log 2>&1
set -euo pipefail
SRC_GIVEN=/root/dsv41/venv_candb
SRC=$(readlink -f "$SRC_GIVEN")
DST=/root/dsv41/venv_candbp
SP=lib/python3.12/site-packages
R=/root/e057b/runs
SCHED=distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
SCHED_V2=94b06f850d3706d154d7d51b835e913af923cee1b5784bed0825c96ba489c8fa
IDX=model_executor/kernels/attention/dsa/dsv41_indexer.py
IDX_SHA=9f13485b
sha() { sha256sum "$1" | cut -c1-64; }
mkdir -p "$R"
# 1. the source must be candidate B exactly (16 sha lines of the E056 kit)
while read -r h rel; do
  [ "$(sha "$SRC/$SP/$rel")" = "$h" ] || { echo "source venv: $rel is not candidate B's"; exit 1; }
done < /root/e056/kit/vllm_overlay_rw.sha256
(cd "$SRC/$SP" && sha256sum $(awk '{print $2}' /root/e056/kit/vllm_overlay_rw.sha256)) > "$R/candb_before.sha256"
echo "source $SRC: all 16 candidate-B sha lines match"
# 2. copy + path rewrite
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
# 3. overlay v2
bash /root/e051/gpu/install_overlay_v2.sh "$DST"
# 4. checks
[ "$(sha "$DST/$SP/vllm/$SCHED")" = "$SCHED_V2" ] || { echo "scheduler is not v2 94b06f85"; exit 1; }
[ "$(sha "$DST/$SP/vllm/$IDX" | cut -c1-8)" = "$IDX_SHA" ] || { echo "indexer is not $IDX_SHA"; exit 1; }
nok=0
while read -r h rel; do
  if [ "$rel" = "vllm/$SCHED" ]; then continue; fi
  [ "$(sha "$DST/$SP/$rel")" = "$h" ] || { echo "installed: $rel mismatch"; exit 1; }
  nok=$((nok + 1))
done < /root/e056/kit/vllm_overlay_rw.sha256
echo "venv_candbp: scheduler $(sha "$DST/$SP/vllm/$SCHED" | cut -c1-8), indexer $(sha "$DST/$SP/vllm/$IDX" | cut -c1-8), other $nok candidate-B lines match"
(cd "$SRC/$SP" && sha256sum $(awk '{print $2}' /root/e056/kit/vllm_overlay_rw.sha256)) > "$R/candb_after.sha256"
cmp -s "$R/candb_before.sha256" "$R/candb_after.sha256" && echo "source venv_candb unchanged" \
  || { echo "ERROR: venv_candb changed"; exit 1; }
"$DST/bin/python" -I -c "
import sys, inspect, vllm
assert vllm.__file__.startswith(sys.prefix), vllm.__file__
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as s
assert s.__file__.startswith(sys.prefix) and hasattr(s, 'SWA_STORE_CHECKPOINT'), s.__file__
assert '[dsv41-offload v2]' in inspect.getsource(s.RequestOffloadState.update_offload_keys)
print('vllm from', vllm.__file__, '| offload scheduler v2 from', s.__file__)"
"$DST/bin/vllm" --version 2>/dev/null | tail -1 || true
rc=0
cd /root/e051/tests && PYTHONPATH=/root/e051/tests CUDA_VISIBLE_DEVICES="" "$DST/bin/python" -m unittest \
  test_dsv41_kv_offload test_dsv41_kv_offload_v2 > "$R/e051_tests_candbp.log" 2>&1 || rc=$?
tail -3 "$R/e051_tests_candbp.log"
grep -q "^Ran 31 tests" "$R/e051_tests_candbp.log" || { echo "expected 31 tests"; rc=1; }
exit $rc
