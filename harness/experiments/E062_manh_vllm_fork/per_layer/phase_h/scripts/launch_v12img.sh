#!/bin/bash
# "V12 + images (causal)": the nb_v12_img notebook (790b1bd9d7b1) launched exactly like V12 / SFT 025ep (launch_nb rerun,
# same runner, KAGGLE_IS_COMPETITION_RERUN=1, evalx2t_s2026). Model input: the staged HF causal model (+ E128's 16 Engram
# shards) mounted at /kaggle/input/models/chankhavu/deepseek-v41-flash-1017-arc-tool-img-e128-2bit-causal/pytorch/default/1
# (the only model mounted; the notebook's bounded lookup finds it at depth models/*/*/*/*/*). Wheels = the g22df163d2d set,
# arc2-2026-src v23. Usage: launch_v12img.sh [run name] [notebook]. Then the V12 sampler and the V12 pod accommodation (uv cache deleted after the install cells).
set -euo pipefail
N=${1:-v12img_rerun}; R=/root/nbruns/$N; NB=${2:-/root/e062pl/v12_img/dsv41-tools-v12-img.ipynb}
export KT_MODEL_OWNER=chankhavu KT_MODEL_SLUG=deepseek-v41-flash-1017-arc-tool-img-e128-2bit-causal KT_MODEL_DIR=/root/img_causal_model \
  KT_SRC_DIR=/root/kin/src_latest/datasets/nguyennguyen599/arc2-2026-src/versions/23 KT_WHEELS_DIR=/root/kin/vllm_wheels_sft
bash /root/sftv01/scripts/launch_nb_owner.sh rerun $N $NB
setsid nohup python3 -I /root/e062pl/v12_sampler.py /root/kwork_$N $R $R/samp > $R/samp.out 2>&1 < /dev/null &
echo $! > $R/sampler.pid
setsid nohup bash /root/sftv01/scripts/uvcache_cleanup.sh $R > $R/uvcache_cleanup.log 2>&1 < /dev/null &
echo "sampler pid $(cat $R/sampler.pid); uv-cache cleanup armed"
