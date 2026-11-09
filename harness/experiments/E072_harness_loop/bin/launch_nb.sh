#!/bin/bash
# Launch a notebook exactly like phase_h launch_nb_owner.sh (rerun mode) + the job's extra env. Usage: launch_nb.sh <commit|rerun> <run> <notebook> [K=V ...]
set -euo pipefail
PH=$1; N=$2; NB=$3; shift 3; R=/root/nbruns/$N
mkdir -p $R; [ -e $R/nb.timeline ] && { echo "$R/nb.timeline exists"; exit 1; }
rm -rf /tmp/dsv41 /tmp/arc_vision /dev/shm/wheels
rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv
bash /root/queue/bin/make_tree.sh $PH /root/kwork_$N > $R/tree.txt 2>&1
sha256sum $NB > $R/notebook.sha256
EXTRA=("$@"); [ "$PH" = rerun ] && EXTRA+=(KAGGLE_IS_COMPETITION_RERUN=1)
setsid nohup bash /root/queue/bin/memlog.sh $R/mem.log > /dev/null 2>&1 < /dev/null &
echo $! > $R/memlog.pid
cd /root/kwork_$N
setsid nohup env -i HOME=/root LANG=C.UTF-8 TERM=dumb VIRTUAL_ENV=/root/kenv \
  PATH=/root/kenv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "${EXTRA[@]}" \
  /root/kenv/bin/python -I /root/e062pl/run_nb.py $NB $R/nb > $R/run_nb.log 2>&1 < /dev/null &
echo $! > $R/run_nb.pid
setsid nohup python3 -I /root/e062pl/v12_sampler.py /root/kwork_$N $R $R/samp > $R/samp.out 2>&1 < /dev/null &
echo $! > $R/sampler.pid
setsid nohup bash /root/queue/bin/uvcache_cleanup.sh $R > $R/uvcache_cleanup.log 2>&1 < /dev/null &
echo "started $N ($PH) pid $(cat $R/run_nb.pid) at $(date -u +%T)"; cat $R/tree.txt
