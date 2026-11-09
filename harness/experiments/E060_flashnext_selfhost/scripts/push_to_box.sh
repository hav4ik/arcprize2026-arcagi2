#!/bin/bash
# E060: copy what the box needs from this workstation into /root/e060/repo on a (fresh) box.
#   bash experiments/E060_flashnext_selfhost/scripts/push_to_box.sh HOST PORT
# Copies: evals/ (no results), the 120 public eval tasks, notes/prompts (the harness reads the ARC Prize template),
# data_tools/check_image_tokens.py, and this experiment's scripts/, patches/ and arm configs (into repo/exp/).
# No secrets are copied. Then run scripts/setup_box.sh on the box.
set -euo pipefail
HOST=$1; PORT=$2
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
X=$ROOT/experiments/E060_flashnext_selfhost
SSH="ssh -p $PORT"
$SSH root@$HOST 'mkdir -p /root/e060/repo/data/arc-agi-2-github/data /root/e060/repo/notes /root/e060/repo/data_tools /root/e060/repo/exp /root/e060/runs'
rsync -a -e "$SSH" --exclude results --exclude __pycache__ --exclude sandbox_docker "$ROOT/evals" root@$HOST:/root/e060/repo/
rsync -a -e "$SSH" "$ROOT/data/arc-agi-2-github/data/evaluation" root@$HOST:/root/e060/repo/data/arc-agi-2-github/data/
rsync -a -e "$SSH" "$ROOT/notes/prompts" root@$HOST:/root/e060/repo/notes/
rsync -a -e "$SSH" "$ROOT/data_tools/check_image_tokens.py" root@$HOST:/root/e060/repo/data_tools/
rsync -a -e "$SSH" "$X/scripts" "$X/patches" root@$HOST:/root/e060/repo/
rsync -a -e "$SSH" "$X"/cfg_*.txt "$X"/extra_*.json "$X/smoke_tasks.txt" root@$HOST:/root/e060/repo/exp/
echo "pushed to $HOST:$PORT /root/e060/repo"
