# usage: bash -s <run>
N=$1; R=/root/runs/$N; L=$R/run.log
echo "SERVER $(grep -E '\] server: ' $L | tail -1)"
echo "UP $(grep -E 'server up after' $L | tail -1)"
echo "NSTART $(grep -c 'server start' $L) NDIED $(grep -c 'server died' $L) NGATE $(grep -c 'KV gate:' $L) DONE $(grep -c '\] done$' $L)"
echo "GATE $(grep 'KV gate:' $L | tail -1)"
echo "EVENTS $(grep -E 'server died|server exited|server failed|signal |hard stop|force deadline|local score|stop-new|no restart|Traceback|warning' $L | grep -v 'deadlines: stop-new' | tail -3 | tr '\n' '|')"
echo "LAST $(tail -1 $L)"
echo "ATT $(wc -l < $R/out/dsv41_logs/attempts.jsonl 2>/dev/null)"
S=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /solver_dsv41.py/ {print $1}' | tr '\n' ' ')
echo "SOLVER ${S:-none}"
echo "MEM $(tail -1 $R/mem.log 2>/dev/null) maxrss=$(awk '{split($3,a,"="); if (a[2]>m) m=a[2]} END {print m}' $R/mem.log 2>/dev/null)"
echo "CG rss+shmem_gib=$(awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} END {printf "%.1f", (r+s)/1073741824}' /sys/fs/cgroup/memory/memory.stat) limit_gib=$(awk '{printf "%.1f", $1/1073741824}' /sys/fs/cgroup/memory/memory.limit_in_bytes) failcnt=$(cat /sys/fs/cgroup/memory/memory.failcnt)"
