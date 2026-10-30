#!/bin/bash
# Start gpu_hold.py once cell 2r has finished (papermill prints "Executing Cell 13" for cell 2, the cell after 2r) and
# before cell 3 starts the solver's server ("Executing Cell 17"); log both times. Usage: hold_trigger.sh <MiB per GPU>
D=/root/e066/b32mv3
until grep -aq "Executing Cell 13-" $D/run.log 2>/dev/null; do sleep 1; done
echo "$(date -u +%T) trigger: cell 2 started (2r done); cell 3 started already: $(grep -ac 'Executing Cell 17-' $D/run.log)" >> $D/gpu_hold.log
setsid nohup /usr/bin/python3 $D/gpu_hold.py $1 $D/gpu_hold.log > $D/gpu_hold.out 2>&1 < /dev/null &
until grep -aq "Executing Cell 17-" $D/run.log 2>/dev/null; do sleep 1; done
echo "$(date -u +%T) cell 3 (solver + server) started; holder alive: $(pgrep -fc gpu_hold.py)" >> $D/gpu_hold.log
