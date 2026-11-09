#!/usr/bin/env bash
# Install the [dsv41-offload] overlay v2 over a venv with the fork wheel
# (vllm-0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130) + the 12-file kit overlay.
# v2 = v1 + incremental update_offload_keys in offloading/scheduler.py (the other 3 files are unchanged).
#
#   gpu/install_overlay_v2.sh /path/to/venv            # install; also upgrades a v1-patched venv in place
#   gpu/install_overlay_v2.sh /path/to/venv --revert   # restore the wheel originals (*.orig-dsv41off backups)
#
# Each target must be the wheel original, the v1 overlay or the v2 overlay (sha256 below); anything else is refused.
set -euo pipefail
VENV=${1:?usage: install_overlay_v2.sh VENV [--revert]}
MODE=${2:-install}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SITE=$("$VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
DST="$SITE/vllm"
[ -d "$DST" ] || { echo "no vllm package in $SITE"; exit 1; }

# path  sha256(wheel original)  sha256(v1 overlay)  sha256(v2 overlay)
FILES=(
  "distributed/kv_transfer/kv_connector/v1/offloading/config.py 04a314743e1e90ab5f285a62268aab9be89c3b17f1888b760b4999389f4f1eff 664b4b0118e64194c62fe99658a7f84000fe4b2264b956228006783278e09c51 664b4b0118e64194c62fe99658a7f84000fe4b2264b956228006783278e09c51"
  "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py cd43cf5f0c22e4e5616baf9c27a785bac16559d806f21f9e1b7bec056cb97a2d dffd7d86e990fe45a27fba017f53befee4ea52f281407cb608f24168a54034a6 94b06f850d3706d154d7d51b835e913af923cee1b5784bed0825c96ba489c8fa"
  "v1/kv_offload/cpu/manager.py 3b8b9f6a60f5e69aee1101bd491412cb6a9c9763dfbd027707eeb5605cea192f eda86f25e7eb20a8b4af686dce0e5d005b5ce3687557914293f400d0475d7dfd eda86f25e7eb20a8b4af686dce0e5d005b5ce3687557914293f400d0475d7dfd"
  "v1/kv_offload/cpu/spec.py 240d686342837804ca34e43ab7d48b52a16f747d737348f5187d7b0e8e0d5a8d a0dbbafc1e234f3bb937b38fe9dac6b3d1e7c48003ba0015e0f1ab8d717e785f a0dbbafc1e234f3bb937b38fe9dac6b3d1e7c48003ba0015e0f1ab8d717e785f"
)

sha() { sha256sum "$1" | cut -c1-64; }

if [ "$MODE" = "--revert" ]; then
  for row in "${FILES[@]}"; do
    set -- $row
    if [ -f "$DST/$1.orig-dsv41off" ]; then mv "$DST/$1.orig-dsv41off" "$DST/$1"; echo "reverted $1"; fi
  done
  find "$DST" -name '__pycache__' -path '*offload*' -prune -exec rm -rf {} + 2>/dev/null || true
  exit 0
fi

for row in "${FILES[@]}"; do
  set -- $row
  cur=$(sha "$DST/$1")
  src=$(sha "$HERE/overlay_v2/vllm/$1")
  [ "$src" = "$4" ] || { echo "overlay_v2 file $1 does not match its recorded sha256"; exit 1; }
  if [ "$cur" = "$4" ]; then echo "already v2: $1"; continue; fi
  [ "$cur" = "$2" ] || [ "$cur" = "$3" ] || { echo "REFUSING: $DST/$1 is neither the wheel original nor overlay v1"; exit 1; }
done
for row in "${FILES[@]}"; do
  set -- $row
  [ "$(sha "$DST/$1")" = "$4" ] && continue
  # Keep the wheel original as the backup (a v1-patched venv already has it).
  [ -f "$DST/$1.orig-dsv41off" ] || cp -p "$DST/$1" "$DST/$1.orig-dsv41off"
  cp "$HERE/overlay_v2/vllm/$1" "$DST/$1"
  echo "patched (v2) $1"
done
find "$DST/distributed/kv_transfer/kv_connector/v1/offloading" "$DST/v1/kv_offload/cpu" \
  -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
"$VENV/bin/python" -I -c "
import inspect, vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as s
assert hasattr(s, 'SWA_STORE_CHECKPOINT')
assert '[dsv41-offload v2]' in inspect.getsource(s.RequestOffloadState.update_offload_keys)
print('import ok (v2)')"
