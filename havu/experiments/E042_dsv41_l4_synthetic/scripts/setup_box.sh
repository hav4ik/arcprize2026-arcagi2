#!/bin/bash
# E042: ship the bundle to a fresh 4xL4 box and run box_install.sh there (detached; log /root/dsv41/install.log).
# Usage (from the workspace): setup_box.sh <user@host> <ssh port>
# If the box has a bigger volume at /workspace, /root/dsv41 becomes a symlink to /workspace/dsv41 (scripts use
# /root/dsv41 paths).
set -euo pipefail
T=$1; P=$2
E=$(cd "$(dirname "$0")/.." && pwd); K=$E/kaggle
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$P" "$T")
"${SSH[@]}" 'set -e; if [ ! -e /root/dsv41 ] && [ -d /workspace ]; then mkdir -p /workspace/dsv41 && ln -s /workspace/dsv41 /root/dsv41; fi
  mkdir -p /root/dsv41/bundle/scripts /root/dsv41/bundle/trace_shapes; command -v curl tar sha256sum > /dev/null'
tar czf - -C $E/patches apply_patches.py synth_init.py humming_v41.py \
  -C $K/box_artifacts dsv41_indexer.pr154.py flashinfer_jit_cache_fork_sm89.tgz model-e64-meta.tgz \
  -C $K requirements.box.freeze -C $E/metadata synthetic-e48 synthetic-e72 \
  | "${SSH[@]}" 'tar xzf - -C /root/dsv41/bundle'
sha256sum < $K/box_artifacts/dsv41_indexer.py.orig | sed 's/-$/dsv41_indexer.py/' \
  | "${SSH[@]}" 'cat > /root/dsv41/bundle/dsv41_indexer.orig.sha256'
tar czf - -C $E/scripts --exclude __pycache__ . | "${SSH[@]}" 'tar xzf - -C /root/dsv41/bundle/scripts'
tar czf - -C $E/metadata/trace_shapes . | "${SSH[@]}" 'tar xzf - -C /root/dsv41/bundle/trace_shapes'
"${SSH[@]}" 'cp /root/dsv41/bundle/scripts/box_install.sh /root/dsv41/ && chmod +x /root/dsv41/box_install.sh &&
  (setsid nohup /root/dsv41/box_install.sh > /dev/null 2>&1 < /dev/null &) && echo "install started; log /root/dsv41/install.log"'
