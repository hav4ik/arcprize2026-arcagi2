#!/bin/bash
# Run the exact B32-M v3 notebook end to end in RERUN mode with papermill, in a clean environment (env -i) with the
# Kaggle-like kernel venv /root/kenv first on PATH, cwd /kaggle/working; plus the 60 s monitor sampler, mem.log, the
# metrics scraper and the GPU holder trigger. Usage: launch.sh [MiB held per GPU, default 410]
D=/root/e066/b32mv3; cd /kaggle/working
setsid nohup /usr/bin/python3 $D/mon_sampler.py $D/mon.jsonl 60 > $D/mon.out 2>&1 < /dev/null &
setsid nohup bash /root/memlog.sh $D/mem.log > /dev/null 2>&1 < /dev/null &
setsid nohup $D/scrape_loop.sh > /dev/null 2>&1 < /dev/null &
HOLD_MIB=${1:-410}  # gpu_hold.py: MiB held per GPU from cell 2 (after 2r) on, for Kaggle-like free VRAM
setsid nohup $D/hold_trigger.sh $HOLD_MIB > /dev/null 2>&1 < /dev/null &
# The notebook settings cell uses os.environ.setdefault: the same ARC_SERVER_ENV_JSON as the notebook's, with
# VLLM_GPU_MEMORY_LOG_INTERVAL 60 instead of 300 (monitoring only; coordinator request)
SERVER_ENV='{"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "60", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_PLUGINS": ""}'
echo "nb start $(date -u +%T)" >> $D/run.log
setsid nohup env -i HOME=/root LANG=C.UTF-8 TERM=dumb VIRTUAL_ENV=/root/kenv KAGGLE_IS_COMPETITION_RERUN=1 ARC_SERVER_ENV_JSON="$SERVER_ENV" \
  PATH=/root/kenv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -c "/root/kgl/bin/papermill --kernel py313b32m --log-output --request-save-on-cell-execute --cwd /kaggle/working $D/nb_rerun.ipynb $D/nb_rerun_out.ipynb >> $D/run.log 2>&1; echo \"nb end rc=\$? \$(date -u +%T)\" >> $D/run.log" > /dev/null 2>&1 < /dev/null &
sleep 2; pgrep -af "mon_sampler|memlog.sh $D|papermill" | cut -c1-100
