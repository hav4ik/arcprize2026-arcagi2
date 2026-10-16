#!/usr/bin/env bash
# Mirror this workspace into the GitHub snapshot repo (hav4ik/arcprize2026-arcagi2, folder havu/).
# Usage:
#   data_tools/sync_github.sh                 # sync + stage + secret scan, show the diff stat (no commit)
#   data_tools/sync_github.sh "havu: message" # same, then commit (unsigned, like the history) and push to main
# Never synced: data/, .venv/, SECRETS.yaml, ENDPOINTS.yaml, caches, nested .git dirs. The repo's .gitignore
# also skips *.log, *.pid and vendor/ copies.
set -euo pipefail
WS=/home/vu/Workspace/ARC-AGI-2
REPO=${REPO:-/home/vu/Workspace/arcprize2026-arcagi2}
cd "$REPO"
git pull -q --ff-only origin main
rsync -a --delete --exclude=/data/ --exclude=/.venv/ --exclude=/SECRETS.yaml --exclude=/ENDPOINTS.yaml \
  --exclude=__pycache__/ --exclude='*.pyc' --exclude=.git/ "$WS/" "$REPO/havu/"
git add -A havu
# Scan exactly what would be committed (added or modified files).
git diff --cached --name-only --diff-filter=AM | sed "s#^#$REPO/#" | "$WS/.venv/bin/python" "$WS/data_tools/secret_scan.py" - \
  || { echo "secret scan failed: nothing committed; fix and re-run (git -C $REPO reset to unstage)"; exit 1; }
git diff --cached --stat | tail -1
if [ $# -ge 1 ]; then
  git -c commit.gpgsign=false commit -q -m "$1"
  git push -q origin main
  git log --oneline -1
fi
