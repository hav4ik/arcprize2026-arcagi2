#!/bin/bash
# Run over ssh by the LOCAL watcher with the Kaggle token on stdin (never stored). Reads /root/queue/need_kaggle_token (path<TAB>size per
# line; first line "DRY" = handshake test only), downloads in two streams, writes token_ok / token_fail.
Q=/root/queue; B=$Q/bin; D=/root/models/e128; H=chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1
read -r tok; [ -n "$tok" ] || { touch $Q/token_fail; exit 1; }
mv $Q/need_kaggle_token $Q/token_inprogress
if [ "$(head -1 $Q/token_inprogress)" = DRY ]; then echo "dry: token length ${#tok}"; rm -f $Q/token_inprogress; touch $Q/token_ok; exit 0; fi
export KAGGLE_API_TOKEN="$tok"; unset tok; T=$Q/dl_tmp; mkdir -p $T
awk 'NR%2==1' $Q/token_inprogress > $T/a.tsv; awk 'NR%2==0' $Q/token_inprogress > $T/b.tsv
for x in a b; do KAGGLEHUB_CACHE=$D/.khcache_$x /root/kgl/bin/python -I $B/dl_kaggle_files.py model $H $T/$x.tsv $D > $Q/logs/stage_$x.log 2>&1 & pids+=($!); sleep 5; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
unset KAGGLE_API_TOKEN; rm -rf $D/.khcache_a $D/.khcache_b $T $Q/token_inprogress
[ $rc -eq 0 ] && touch $Q/token_ok || touch $Q/token_fail; exit $rc
