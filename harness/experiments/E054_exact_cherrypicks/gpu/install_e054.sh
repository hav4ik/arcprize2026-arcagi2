#!/usr/bin/env bash
# E054: install the two exact cherry-picks over a box venv that has the fork wheel + the kit's 12-file overlay.
#
#   gpu/install_e054.sh [VENV] [--only indexer|engram] [--force]   # install (default VENV /root/dsv41/venv)
#   gpu/install_e054.sh [VENV] --revert                             # restore the kit versions
#   gpu/install_e054.sh [VENV] --status                             # show which version is installed
#
# - Source: kit/vllm_overlay_e054_add.tgz (paths relative to site-packages), checked against
#   kit/vllm_overlay_e054_add.sha256 before anything is touched.
# - Refuses unless every target is the kit version (sha256 below) or already the E054 version
#   (--force overrides, after backing up whatever is there).
# - Backups: $VENV/e054_backup/<path> (first install only; never overwritten).
#   gpu/indexer_ab.py reads the old indexer from there.
# - Removes the two modules' stale .pyc. Triton and vLLM compile caches key on source contents,
#   so no cache needs clearing (the first server start after a switch recompiles; ~1-2 min).
# - Runtime kill switches need no reinstall: DSV41_INDEXER_SKIP=0, DSV41_ENGRAM_WKV_TP=0 (server env).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VENV=/root/dsv41/venv
MODE=install
ONLY=""
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --revert) MODE=revert ;;
    --status) MODE=status ;;
    --only) ONLY=$2; shift ;;
    --force) FORCE=1 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) VENV=$1 ;;
  esac
  shift
done
PY="$VENV/bin/python"
SITE=${E054_SITE:-$("$PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}  # E054_SITE: tests only
[ -d "$SITE/vllm" ] || { echo "no vllm package in $SITE"; exit 1; }
BAK="$VENV/e054_backup"
ADD="$HERE/kit/vllm_overlay_e054_add.tgz"
ADDSHA="$HERE/kit/vllm_overlay_e054_add.sha256"

# path  kit (base) sha256  key
ROWS=(
  "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py 3da076e17cb043c57c75f660b0f1b1d2fd15a5d431fb1d7730d192520e629d2f indexer"
  "vllm/models/deepseek_v4_1/common/engram.py 5f1e7230e52f6904c4de834fe97cc57160fedbc30c250968788aab42bf5e9561 engram"
)
sha() { sha256sum "$1" | cut -c1-64; }
newsha() { awk -v p="$1" '$2 == p {print $1}' "$ADDSHA"; }
selected() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
clear_pyc() {
  local d; d=$(dirname "$SITE/$1")/__pycache__
  local b; b=$(basename "$1" .py)
  [ -d "$d" ] && find "$d" -maxdepth 1 -type f -name "$b.*.pyc" -delete || true
}

if [ "$MODE" = status ]; then
  for row in "${ROWS[@]}"; do
    set -- $row
    cur=$(sha "$SITE/$1")
    if [ "$cur" = "$2" ]; then v=kit; elif [ "$cur" = "$(newsha "$1")" ]; then v=E054; else v="UNKNOWN ${cur:0:8}"; fi
    echo "$3: $v  ($1)"
  done
  exit 0
fi

if [ "$MODE" = revert ]; then
  for row in "${ROWS[@]}"; do
    set -- $row
    selected "$3" || continue
    if [ "$(sha "$SITE/$1")" = "$2" ]; then echo "already kit: $1"; continue; fi
    [ -f "$BAK/$1" ] || { echo "no backup for $1 in $BAK"; exit 1; }
    [ "$(sha "$BAK/$1")" = "$2" ] || { echo "backup of $1 is not the kit version; not restoring"; exit 1; }
    cp -p "$BAK/$1" "$SITE/$1"
    [ "$(sha "$SITE/$1")" = "$2" ] || { echo "restore of $1 failed its sha check"; exit 1; }
    clear_pyc "$1"
    echo "reverted $1"
  done
  exit 0
fi

# install: check the source first, then every target, then write
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/e054_stage.XXXXXX")
trap 'rm -rf -- "${STAGE:?}"' EXIT
tar -xzf "$ADD" -C "$STAGE"
for row in "${ROWS[@]}"; do
  set -- $row
  want=$(newsha "$1")
  [ -n "$want" ] && [ "$(sha "$STAGE/$1")" = "$want" ] || { echo "kit tarball member $1 does not match its sha file"; exit 1; }
  selected "$3" || continue
  cur=$(sha "$SITE/$1")
  if [ "$cur" = "$want" ]; then echo "already E054: $1"; continue; fi
  if [ "$cur" != "$2" ] && [ "$FORCE" != 1 ]; then
    echo "REFUSING: $SITE/$1 is ${cur:0:8}, neither the kit version ${2:0:8} nor E054 ${want:0:8} (--force to override)"
    exit 1
  fi
done
for row in "${ROWS[@]}"; do
  set -- $row
  selected "$3" || continue
  want=$(newsha "$1")
  [ "$(sha "$SITE/$1")" = "$want" ] && continue
  mkdir -p "$(dirname "$BAK/$1")"
  [ -f "$BAK/$1" ] || cp -p "$SITE/$1" "$BAK/$1"
  cp "$STAGE/$1" "$SITE/$1"
  [ "$(sha "$SITE/$1")" = "$want" ] || { echo "install of $1 failed its sha check"; exit 1; }
  clear_pyc "$1"
  echo "installed $1 (${want:0:8}); backup in $BAK"
done
"$PY" -I - "$SITE" <<'EOF'
import ast, sys
site = sys.argv[1]
for rel, needle in (("vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py", "DSV41_INDEXER_SKIP"),
                    ("vllm/models/deepseek_v4_1/common/engram.py", "DSV41_ENGRAM_WKV_TP")):
    src = open(f"{site}/{rel}").read()
    ast.parse(src)
    print(f"{rel}: parses, {'E054' if needle in src else 'kit'} version")
EOF
echo "status:"; bash "$0" "$VENV" --status
