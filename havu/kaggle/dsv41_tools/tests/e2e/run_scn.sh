#!/bin/bash
# run_scn.sh <name> <challenges.json> <notebook.ipynb> <port> [ENV=VALUE ...]: one emulated competition rerun.
A=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/e2e_next
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad
N=$1; CH=$2; NB=$3; PORT=$4; shift 4
R=$A/runs/$N; rm -rf $R; mkdir -p $R/input/arc-prize-2026-arc-agi-2 $R/input/dsv41-tools-kit $R/work $R/scratch
cp $CH $R/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json
KIT_SRC=${KIT_SRC:-$A/realkit}; cp $KIT_SRC/dsv41_kit.tgz $R/input/dsv41-tools-kit/; [ -f $KIT_SRC/MANIFEST.json ] && cp $KIT_SRC/MANIFEST.json $R/input/dsv41-tools-kit/
mkdir -p $R/input/models/e128; cp -r /tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_e2e/model_e128/. $R/input/models/e128/
cp $NB $R/notebook.ipynb
(
  export KAGGLE_IS_COMPETITION_RERUN=1 ARC_INPUT_ROOT=$R/input ARC_OUT=$R/work ARC_SCRATCH=$R/scratch ARC_SKIP_INSTALL=1 \
    ARC_VENV=$A/venv ARC_TOOL_PYTHON=$S/toolvenv/bin/python ARC_PORT=$PORT PATH=$A/fakebin:$PATH \
    FAKE_LOG=$R/fake.jsonl FAKE_STATE=$R
  for kv in "$@"; do export "$kv"; done
  env | grep -E '^(ARC_|FAKE_|KAGGLE)' | sort > $R/env.txt
  cd $R && exec setsid /home/vu/Workspace/ARC-AGI-2/.venv/bin/python -u $A/tools/run_nb.py $R/notebook.ipynb > $R/run.log 2>&1 < /dev/null
) &
echo $! > $R/nb.pid
echo "started $N pid $(cat $R/nb.pid)"
