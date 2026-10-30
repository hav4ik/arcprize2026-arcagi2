#!/bin/bash
# v12_cycle.sh <until "YYYY-mm-dd HH:MM"> <known alerts> <short|full>: wait (v12_wait.sh), then sync the dashboard.
D=$(dirname $0); $D/v12_wait.sh "$1" "$2" "$3"; echo "== dashboard sync"; $D/dash_sync.sh 2>&1 | tail -2
