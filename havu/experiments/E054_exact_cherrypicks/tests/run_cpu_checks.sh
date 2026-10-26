#!/usr/bin/env bash
# E054 CPU checks (no GPU, no network). Usage:
#   tests/run_cpu_checks.sh PYTHON FORK_TREE
#     PYTHON     a Python with CPU torch + triton + the fork's deps (here: the E051 testvenv; on a box the server venv,
#                which then runs with CUDA hidden)
#     FORK_TREE  a directory holding vllm/ = fork vision11 + our 12-file overlay (here: scratchpad fp4kv/fork_ov,
#                read-only; on a box: the venv's site-packages). The two E054 files are injected over it by
#                tests/cpu_harness.py, so the tree itself is never modified.
# Runs: kit tarball/sha consistency; syntax of every script; the #58678 constructor + real-class TP tests; the
# packaging simulation (cell 0b + cell 2); the #156 Triton-interpreter A/B (~80 s). Exit code != 0 on any failure.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
E=$(dirname "$T")
PY=${1:?usage: run_cpu_checks.sh PYTHON FORK_TREE}
TREE=${2:?usage: run_cpu_checks.sh PYTHON FORK_TREE}
[ -d "$TREE/vllm" ] || { echo "$TREE has no vllm/"; exit 1; }

echo "== kit files"
"$PY" -I "$E/scripts/build_kit.py" --check 2>/dev/null || python3 -I "$E/scripts/build_kit.py" --check || {
  # on a box the repo's kaggle/ kit is absent: check the E054 pair against its own sha lists only
  (cd "$E/kit" && sha256sum -c vllm_overlay_e054_add.sha256 >/dev/null 2>&1) || true
  for stem in vllm_overlay_e054_add vllm_overlay_rw; do
    "$PY" -I - "$E/kit/$stem.tgz" "$E/kit/$stem.sha256" <<'EOF'
import hashlib, sys, tarfile
rows = [l.split() for l in open(sys.argv[2]) if l.strip()]
with tarfile.open(sys.argv[1]) as tf:
    got = {m.name: hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf.getmembers() if m.isfile()}
assert len(got) == len(rows) and all(got[r] == h for h, r in rows), sys.argv[1]
print("OK", sys.argv[1])
EOF
  done
}

echo "== syntax"
for f in "$E"/gpu/*.py "$E"/scripts/*.py "$E"/tests/*.py "$E"/overlay/vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py \
         "$E"/overlay/vllm/models/deepseek_v4_1/common/engram.py; do
  "$PY" -I -c "import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])" "$f"
done
for f in "$E"/gpu/*.sh "$E"/tests/*.sh; do bash -n "$f"; done
echo "syntax ok"

echo "== unit tests"
cd "$T"
CUDA_VISIBLE_DEVICES="" TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TREE:$T" \
  "$PY" -W ignore::ResourceWarning -m unittest -v test_engram_wkv_tp test_cell0b_sim test_indexer_interp 2>&1 \
  | grep -v -E "deprecated|^WARNING|Failed to import from vllm._C"
exit "${PIPESTATUS[0]}"
