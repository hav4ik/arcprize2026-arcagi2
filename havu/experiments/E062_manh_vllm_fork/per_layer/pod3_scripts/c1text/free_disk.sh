#!/bin/bash
# Free regenerable data before the c1text commit run (coordinator, >= 20 GB): run 2/3's archived /tmp/dsv41 copies
# (uv cache + extracted kit; their stage.log files kept), the failed commit attempt's /tmp/dsv41 and /tmp/arc_vision,
# /dev/shm/wheels (tmpfs, recreated by Manh's cell 3), and /root/b32m/venv (runs 2-3's kernel venv; rebuildable from
# /root/b32m/dl with install_b32m.sh). Keeps the C1 model, the v10/v11 wheel trees and the kit. Prints df before/after.
set -u
echo "free_disk $(date -u +%T): $(df -h / | tail -1)"
mkdir -p /root/e066/c1text/kept_logs
for d in /root/e066/run2_tmp_dsv41 /root/e066/run3_tmp_dsv41 /tmp/dsv41; do
  [ -f $d/stage.log ] && cp $d/stage.log /root/e066/c1text/kept_logs/stage$(echo $d | tr / _).log
  echo "rm $d ($(du -xsh $d 2>/dev/null | cut -f1))"; rm -rf $d
done
rm -rf /tmp/arc_vision; echo "rm /dev/shm/wheels ($(du -sh /dev/shm/wheels 2>/dev/null | cut -f1))"; rm -rf /dev/shm/wheels
echo "rm /root/b32m/venv ($(du -xsh /root/b32m/venv | cut -f1))"; rm -rf /root/b32m/venv /root/b32m/uvcache
echo "after: $(df -h / | tail -1)"
