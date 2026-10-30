#!/bin/bash
# "SFT 025ep on V12": V12's launch (launch_nb.sh rerun, same runner, KAGGLE_IS_COMPETITION_RERUN=1, evalx2t_s2026) with
# the model input mapped to the staged SFT dir and the vllm-wheels input to the new full set; then the V12 sampler and
# the V12 pod accommodation (uv cache deleted after the install cells).
set -euo pipefail
N=s025_rerun; R=/root/nbruns/$N; NB=/root/e062pl/sft_025ep_v12/dsv41-tools-submit-b32m-external-server.ipynb
export KT_MODEL_SLUG=dsv41-flash-1017-arc-tool-e128-2bit KT_MODEL_DIR=/root/sft_025ep_model \
  KT_SRC_DIR=/root/kin/src_latest/datasets/nguyennguyen599/arc2-2026-src/versions/23 KT_WHEELS_DIR=/root/kin/vllm_wheels_sft
bash /root/e062pl/launch_nb.sh rerun $N $NB
setsid nohup python3 -I /root/e062pl/v12_sampler.py /root/kwork_$N $R $R/samp > $R/samp.out 2>&1 < /dev/null &
echo $! > $R/sampler.pid
setsid nohup bash /root/sftv01/scripts/uvcache_cleanup.sh $R > $R/uvcache_cleanup.log 2>&1 < /dev/null &
echo "sampler pid $(cat $R/sampler.pid); uv-cache cleanup armed"
