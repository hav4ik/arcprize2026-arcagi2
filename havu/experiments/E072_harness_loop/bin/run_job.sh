#!/bin/bash
# run_job.sh <jobfile>: one job, start to finish. Job file = shell env (see ../README.md). Exit 0 = scored; 10+ = failed before launch.
Q=/root/queue; B=$Q/bin; J=$1
set -a; . /root/queue/pod.env; . "$J"; set +a
: "${JOB_LABEL:?}" "${NB_PATH:?}" "${NB_SHA256:?}" "${WORKLOAD:?}" "${KT_WHEELS_DIR:?}"
MAX_KAGGLE_H=${MAX_KAGGLE_H:-7}; MODEL_KIND=${MODEL_KIND:-e128}; EXTRA_ENV=${EXTRA_ENV:-}
RUN=q_$JOB_LABEL; [ -e /root/nbruns/$RUN ] && RUN=${RUN}_$(date -u +%m%d%H%M)
log() { echo "$(date -u +%FT%TZ) [$RUN] $*" | tee -a $Q/logs/runner.log; }
log "job $J: nb $NB_PATH workload $WORKLOAD env '$EXTRA_ENV' max ${MAX_KAGGLE_H}h"
[[ "$(sha256sum "$NB_PATH" | cut -d' ' -f1)" == $NB_SHA256* ]] || { log "FAIL notebook sha mismatch"; exit 11; }
WD=$KT_WHEELS_DIR/wheels/wheels
if [ -n "${WHEELS_EXPECT_COUNT:-}" ]; then
  n=$(ls $WD 2>/dev/null | wc -l); ls $WD 2>/dev/null | grep -q "^vllm-.*${WHEELS_EXPECT_VLLM:-vllm}" && [ "$n" -eq "$WHEELS_EXPECT_COUNT" ] \
    || { log "FAIL wheels $WD: $n files (expect $WHEELS_EXPECT_COUNT, vllm ${WHEELS_EXPECT_VLLM:-any})"; exit 15; }
fi
mkdir -p $Q/nb /root/nbruns/$RUN
V=$Q/nb/$RUN.ipynb
python3 -I $B/mk_variant_nb.py "$NB_PATH" "$EXTRA_ENV" $V "${NB_PATCHES:-}" > /root/nbruns/$RUN/variant.json || { log "FAIL variant"; exit 12; }
log "variant: $(cat /root/nbruns/$RUN/variant.json)"
cp "$J" /root/nbruns/$RUN/job.env
bash $B/stage_model.sh "$MODEL_KIND" >> $Q/logs/stage_$RUN.log 2>&1 || { log "FAIL model staging (see logs/stage_$RUN.log)"; exit 13; }
export KT_MODEL_OWNER=${KT_MODEL_OWNER:-chankhavu} KT_MODEL_SLUG=${KT_MODEL_SLUG:-dsv41-flash-1017-arc-tool-e128-2bit} \
  KT_MODEL_DIR=${KT_MODEL_DIR:-/root/models/e128} KT_SRC_DIR KT_WHEELS_DIR KT_WORKLOAD_CH=$Q/wl/${WORKLOAD}_challenges.json
[ -n "${DRY_RUN:-}" ] && { log "DRY_RUN: pre-steps ok, not launching"; exit 0; }
printf '%s\t%s\t%s\t%s\n' "$JOB_LABEL" "$RUN" "$WORKLOAD" "$(date -u +%FT%TZ)" >> $Q/index.tsv
read -ra JE <<< "$EXTRA_ENV"
bash $B/launch_nb.sh rerun $RUN $V "${JE[@]}" >> $Q/logs/launch_$RUN.log 2>&1 || { log "FAIL launch (see logs/launch_$RUN.log)"; exit 14; }
R=/root/nbruns/$RUN; T0=$(date +%s); log "launched; waiting"; last_score=0; why=done
while true; do
  sleep 30
  grep -q NOTEBOOK_DONE $R/nb.timeline 2>/dev/null && { why="NOTEBOOK_DONE: $(grep NOTEBOOK_DONE $R/nb.timeline | cut -c1-200)"; break; }
  kill -0 "$(cat $R/run_nb.pid)" 2>/dev/null || { why="runner process gone"; break; }
  if [ $(( $(date +%s) - T0 )) -gt $(python3 -c "print(int($MAX_KAGGLE_H*3600))") ]; then why="max Kaggle clock ${MAX_KAGGLE_H}h reached"; python3 $B/stop_run.py $RUN >> $Q/logs/launch_$RUN.log 2>&1; break; fi
  if [ -n "${MEM_ABORT_GIB:-}" ]; then  # host RAM guard: cgroup rss+shmem+7 (Kaggle-eq) above the limit -> abort the run
    g=$(awk '$1=="rss"||$1=="shmem"{s+=$2} END{printf "%d", s/1073741824+7}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
    if [ "${g:-0}" -gt "$MEM_ABORT_GIB" ]; then why="ABORT host RAM guard: Kaggle-eq ${g} GiB > $MEM_ABORT_GIB"; python3 $B/stop_run.py $RUN >> $Q/logs/launch_$RUN.log 2>&1; break; fi
  fi
  if [ $(( $(date +%s) - last_score )) -gt 600 ]; then last_score=$(date +%s); python3 -I $B/score_job.py $RUN $WORKLOAD > /dev/null 2>&1; fi
done
log "ended: $why"
python3 $B/reap.py $RUN >> $Q/logs/launch_$RUN.log 2>&1
python3 -I $B/score_job.py $RUN $WORKLOAD > $R/score.txt 2>&1; log "score: $(cut -c1-400 $R/score.txt)"
echo "$why" > $R/END_REASON
exit 0
