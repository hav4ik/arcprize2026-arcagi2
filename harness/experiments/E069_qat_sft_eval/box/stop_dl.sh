#!/bin/bash
# E068 box: stop the running checkpoint download (its whole process group), found by the exact script path.
for pg in $(ps -eo pgid=,args= | awk '$2=="bash" && $3=="/root/e068/scripts/dl_e128t.sh" {print $1}' | sort -u); do
  echo "stopping download process group $pg"; kill -- -$pg
done
