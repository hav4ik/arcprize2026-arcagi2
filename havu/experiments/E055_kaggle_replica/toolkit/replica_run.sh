#!/bin/bash
# E055 toolkit: launch a Kaggle-fidelity 12 h replica of the dsv41 solver on a Runpod 4xL4 pod, through the pod's
# /root/podrun.sh (experiments/E046_multi_test_rollout/pods/podrun.sh: Kaggle notebook cells as local_run.py,
# ARC_SKIP_INSTALL=1, local model).
#
# Usage: replica_run.sh <pod_ssh_port> <pod_ip> <name> <workload> [ENV=VAL ...]
#   workload  a name in toolkit/workloads/ (e.g. evalx2t_s2026) or a path prefix; <prefix>_challenges.json and
#             <prefix>_solutions.json are copied to the pod's /root/data/
#   ENV=VAL   solver settings passed to podrun.sh (ARC_WORKERS=44 ARC_MULTI_TEST=1 ARC_KIT=/root/kit_v5 ...)
#
# Fidelity corrections applied here:
#   - deadlines 600 s earlier than Kaggle's (stop-new / force / hard stop / kill switch = 10.83 / 11.17 / 11.5 / 11.75 h
#     from the notebook's first cell, T0 = ARC_T0 / time.time() in local_run.py cell 0). On Kaggle the offline install
#     (256 s) and the slower FUSE model load (590 s vs ~285 s on a pod) happen after T0, so its first attempt starts
#     ~600 s later (Kaggle bench V1: server up at +0.24 h vs +0.06-0.10 h on pods). A deadline passed as ENV is taken
#     on Kaggle's clock and shifted too.
#   - KV pool: ARC_GPU_UTIL=0.938 (1,556,850 tokens on pods = Kaggle's 1,554,183 at 0.945).
#   - memlog with the cgroup's memory.current (what Kaggle's 173 GiB limit applies to) and GPU power / SM clock.
# The remaining gaps (pod prefill ~1.45x and decode ~1.06x faster than Kaggle) are corrected after the run by
# replica_report.py's Kaggle-clock section; they are listed in the fidelity.md written beside the run.
#
# Local options (environment of this script):
#   REPLICA_SHIFT_S=600   deadline shift in seconds (0: Kaggle's deadlines on the pod clock)
#   REPLICA_DRY_RUN=1     print every remote command instead of running it (no ssh)
#   REPLICA_POWER_W=50    try `nvidia-smi -pl 50` on every GPU first (Runpod containers usually may not; the result is
#                         recorded). Off by default: it changes the pod's state until reset.
#   REPLICA_SSH_USER=root
#   REPLICA_LOCAL_DIR     where replica.json / fidelity.md go locally (default toolkit/runs/<name>)
#   REPLICA_POST_LAUNCH=<UTC launch time, e.g. 2026-10-27T04:07:50Z>
#                         the run is already started (same name and settings): skip the refusals, the workload copy
#                         (its checksum is still verified) and the launch; write only the memlog and the records
# The script never stops or kills anything on the pod: it refuses to start when a solver is already running, a GPU
# holds memory, or the run directory exists.
set -euo pipefail

if [ $# -lt 4 ]; then
  sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
PORT=$1; IP=$2; NAME=$3; WL=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SHIFT=${REPLICA_SHIFT_S:-600}
DRY=${REPLICA_DRY_RUN:-0}
USER_=${REPLICA_SSH_USER:-root}
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad run name: $NAME" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "bad port: $PORT" >&2; exit 2; }

# ---- workload
if [ -f "$HERE/workloads/${WL}_challenges.json" ]; then PREFIX="$HERE/workloads/$WL"; else PREFIX="$WL"; fi
CH="${PREFIX}_challenges.json"; SO="${PREFIX}_solutions.json"
[ -f "$CH" ] && [ -f "$SO" ] || { echo "workload files not found: $CH / $SO" >&2; exit 2; }
WLN=$(basename "$PREFIX")
CH_SHA=$(sha256sum "$CH" | cut -d' ' -f1); SO_SHA=$(sha256sum "$SO" | cut -d' ' -f1)
REL_CH=$(realpath --relative-to="$ROOT" "$CH"); REL_SO=$(realpath --relative-to="$ROOT" "$SO")

# ---- settings: defaults, then the caller's, then the deadline shift
declare -A ENV
ENV[ARC_CHALLENGES]=/root/data/${WLN}_challenges.json
ENV[ARC_SOLUTIONS]=/root/data/${WLN}_solutions.json
ENV[ARC_GPU_UTIL]=0.938
ENV[ARC_MAX_NUM_SEQS]=64
ENV[ARC_CACHE_SALT]=${NAME}_$(date -u +%H%M%S)
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "not ENV=VAL: $kv" >&2; exit 2; }
  k=${kv%%=*}; [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "bad variable name: $k" >&2; exit 2; }
  ENV[$k]=${kv#*=}
done
declare -A KAGGLE=([ARC_STOP_NEW_S]=39000 [ARC_FORCE_AT_S]=40200 [ARC_HARD_STOP_S]=41400 [ARC_KILL_AT_S]=42300)
declare -A KCLOCK
for k in ARC_STOP_NEW_S ARC_FORCE_AT_S ARC_HARD_STOP_S ARC_KILL_AT_S; do
  v=${ENV[$k]:-${KAGGLE[$k]}}
  KCLOCK[$k]=$v
  ENV[$k]=$(python3 -c "print(int(float('$v')) - int('$SHIFT'))")
done
ARGS=()
for k in $(printf '%s\n' "${!ENV[@]}" | sort); do ARGS+=("$k=${ENV[$k]}"); done

SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -p "$PORT" "$USER_@$IP")
SCP=(scp -q -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")
remote() {  # remote <command string>: run on the pod (or print it)
  if [ "$DRY" = 1 ]; then echo "[dry-run] ssh -p $PORT $USER_@$IP $1" >&2; echo "dry-run"; else "${SSH[@]}" "$1"; fi
}

LOCAL=${REPLICA_LOCAL_DIR:-"$HERE/runs/$NAME"}
[ -e "$LOCAL/replica.json" ] && { echo "local run record exists: $LOCAL (pick another name)" >&2; exit 2; }
mkdir -p "$LOCAL"
R=/root/runs/$NAME
KIT=${ENV[ARC_KIT]:-/root/kit}

# ---- pre-flight (read-only)
echo "pre-flight on $IP:$PORT ..."
PRE=$(remote "set -u; test -x /root/podrun.sh && test -f /root/local_run.py && echo podrun_ok || echo podrun_missing;
  test -e $R && echo run_dir_exists || echo run_dir_free;
  pgrep -af 'python[0-9.]* -u [^ ]*/[l]ocal_run[.]py|python[0-9.]* -u [^ ]*/[s]olver_dsv41[.]py' | cut -c1-160 | sed 's/^/busy: /';
  pgrep -f 'python[0-9.]* -u [^ ]*/[l]ocal_run[.]py|python[0-9.]* -u [^ ]*/[s]olver_dsv41[.]py' >/dev/null && echo solver_running || echo solver_idle;
  nvidia-smi --query-gpu=index,name,memory.used,power.limit,power.max_limit,power.min_limit,clocks.max.sm --format=csv,noheader 2>&1 | sed 's/^/gpu: /';
  for f in $KIT/solver_dsv41.py $KIT/repo/evals/run_deepseek_tools.py $KIT/repo/evals/train_check.py; do
    [ -f \$f ] && echo \"kit: \$(sha256sum \$f | cut -c1-8) \$f\" || echo \"kit: missing \$f\"; done;
  echo \"cgroup_max: \$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)\";
  echo \"nproc: \$(nproc)\"; echo \"mem_total_gib: \$(free -g | awk 'NR==2{print \$2}')\"")
echo "$PRE" | sed 's/^/  /'
POST=${REPLICA_POST_LAUNCH:-}
if [ "$DRY" != 1 ] && [ -z "$POST" ]; then
  grep -q podrun_ok <<<"$PRE" || { echo "pod lacks /root/podrun.sh or /root/local_run.py" >&2; exit 3; }
  grep -q run_dir_free <<<"$PRE" || { echo "$R exists on the pod: pick another name" >&2; exit 3; }
  grep -q solver_idle <<<"$PRE" || { echo "a solver is already running on the pod: not starting (never kill it from here)" >&2; exit 3; }
  if grep '^gpu:' <<<"$PRE" | awk -F', ' '{gsub(/ MiB/,"",$3); if ($3+0 > 1000) bad=1} END {exit !bad}'; then
    echo "a GPU already holds > 1 GiB: not starting" >&2; exit 3; fi
fi

# ---- workload to the pod, checked
echo "copying workload $WLN ..."
if [ "$DRY" = 1 ]; then echo "[dry-run] scp $CH $SO -> /root/data/" >&2; else
  if [ -z "$POST" ]; then remote "mkdir -p /root/data" >/dev/null; "${SCP[@]}" "$CH" "$SO" "$USER_@$IP:/root/data/"; fi
  GOT=$(remote "sha256sum /root/data/${WLN}_challenges.json /root/data/${WLN}_solutions.json | cut -d' ' -f1 | tr '\n' ' '")
  [ "$GOT" = "$CH_SHA $SO_SHA " ] || { echo "workload checksum mismatch on the pod: $GOT" >&2; exit 3; }
fi

# ---- optional power cap
POWER="not requested (pods run at their 72 W limit; Kaggle's L4s at 50 W)"
if [ -n "${REPLICA_POWER_W:-}" ]; then
  POWER=$(remote "nvidia-smi -pl ${REPLICA_POWER_W} 2>&1 | tail -4; nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' '" || true)
  echo "power cap ${REPLICA_POWER_W} W: $POWER"
fi

# ---- launch
CMD=$(printf '%q ' /root/podrun.sh "$NAME" "${ARGS[@]}")
if [ -n "$POST" ]; then
  LAUNCHED=$POST; echo "already launched at $POST: $CMD"
else
  echo "launching: $CMD"
  # detached: podrun.sh's children inherit its stdout, so a plain ssh would wait for the whole 12 h run
  remote "setsid nohup $CMD > /root/runs/$NAME.launch.out 2>&1 < /dev/null & sleep 5; echo launched"
  LAUNCHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
fi
# memlog: host used / RSS (as /root/memlog.sh) + cgroup memory.current + GPU power and SM clocks, every 60 s
remote "sleep 2; cat > $R/memlog_cg.sh <<'EOS'
#!/bin/bash
while true; do
  cg=\$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
  g=\$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';')
  echo \"\$(date -u +%T) used_gib=\$(free -g | awk 'NR==2{print \$3}') rss_gib=\$(ps -eo rss= | awk '{s+=\$1} END {printf \"%.1f\", s/1048576}') mlocked_kb=\$(grep Mlocked /proc/meminfo | awk '{print \$2}') cg_gib=\$(awk -v b=\$cg 'BEGIN{printf \"%.1f\", b/1073741824}') gpu_w_mhz=\$g\" >> \$1
  sleep 60
done
EOS
chmod +x $R/memlog_cg.sh; setsid nohup $R/memlog_cg.sh $R/mem.log > /dev/null 2>&1 < /dev/null & echo memlog started" >/dev/null

# ---- local record: replica.json + fidelity.md (also copied beside the run on the pod)
export NAME IP PORT LAUNCHED WLN CH_SHA SO_SHA REL_CH REL_SO SHIFT POWER PRE DRY
ENV_JSON=$(for a in "${ARGS[@]}"; do printf '%s\n' "$a"; done | python3 -c 'import json,sys; print(json.dumps(dict(l.rstrip("\n").split("=",1) for l in sys.stdin if "=" in l)))')
KCL_JSON=$(for k in "${!KCLOCK[@]}"; do printf '%s=%s\n' "$k" "${KCLOCK[$k]}"; done | python3 -c 'import json,sys; print(json.dumps({k: int(float(v)) for k, v in (l.strip().split("=") for l in sys.stdin)}))')
export ENV_JSON KCL_JSON
python3 - "$LOCAL" <<'PY'
import json, os, sys
from pathlib import Path
d = Path(sys.argv[1]); E = os.environ
env = json.loads(E["ENV_JSON"]); kcl = json.loads(E["KCL_JSON"])
pre = [l.strip() for l in E["PRE"].splitlines() if l.strip()]
rec = {"name": E["NAME"], "pod": f"{E['IP']}:{E['PORT']}", "launched_utc": E["LAUNCHED"], "dry_run": E["DRY"] == "1",
       "workload": E["WLN"], "challenges_local": E["REL_CH"], "solutions_local": E["REL_SO"],
       "challenges_sha256": E["CH_SHA"], "solutions_sha256": E["SO_SHA"], "shift_s": int(E["SHIFT"]),
       "kaggle_clock_deadlines_s": kcl, "env": env, "preflight": pre, "power_cap": E["POWER"]}
(d / "replica.json").write_text(json.dumps(rec, indent=1))
h = lambda s: f"{int(s) / 3600:.2f} h"
pod = {k: int(env[k]) for k in kcl}
gpus = [l[5:] for l in pre if l.startswith("gpu: ")]
kit = [l[5:] for l in pre if l.startswith("kit: ")]
md = f"""# Fidelity record: {rec['name']}

Launched {rec['launched_utc']} on pod {rec['pod']} by `experiments/E055_kaggle_replica/toolkit/replica_run.sh`{' (DRY RUN: nothing was started)' if rec['dry_run'] else ''}.

## Corrections applied
- **Workload:** `{rec['workload']}` ({rec['challenges_local']}, sha256 {rec['challenges_sha256'][:12]}; solutions sha256 {rec['solutions_sha256'][:12]}).
- **Deadlines shifted {rec['shift_s']} s earlier** (pod clock): stop-new {h(pod['ARC_STOP_NEW_S'])}, force {h(pod['ARC_FORCE_AT_S'])}, hard stop {h(pod['ARC_HARD_STOP_S'])}, kill switch {h(pod['ARC_KILL_AT_S'])}; on Kaggle's clock {h(kcl['ARC_STOP_NEW_S'])} / {h(kcl['ARC_FORCE_AT_S'])} / {h(kcl['ARC_HARD_STOP_S'])} / {h(kcl['ARC_KILL_AT_S'])}. The solver's clock starts at the notebook's first cell (ARC_T0 in local_run.py cell 0); on Kaggle the offline install (256 s), the toolchain check (17 s) and the FUSE model load (590 s vs ~285 s on a pod) all come after it, so Kaggle's first attempt starts ~600 s later than a pod's (Kaggle bench V1: server up at +0.24 h; pods +0.06-0.10 h).
- **KV pool:** ARC_GPU_UTIL={env.get('ARC_GPU_UTIL')} (pods: 1,556,850 tokens at 0.938 vs Kaggle's 1,554,183 at 0.945).
- **Memory log** with the cgroup's memory.current (Kaggle's limit: 173 GiB) and per-GPU power / SM clock every 60 s (`mem.log`).
- **Power cap:** {rec['power_cap']}.

## Settings
{chr(10).join(f'- `{k}={v}`' for k, v in sorted(env.items()))}

Pod pre-flight: GPUs {'; '.join(gpus) or 'n/a'}; kit {'; '.join(kit) or 'n/a'}; {'; '.join(l for l in pre if l.startswith(('cgroup_max', 'nproc', 'mem_total')))}.

## Remaining known gaps (corrected or bounded by replica_report.py)
1. **Throughput: pods are faster than Kaggle's 50 W L4s.** Measured on the twin pair Kaggle bench V1 / pod `mt36_b05_gate60` (V6 settings): effective prefill 821 vs 1,187 tok/s (k = 1.45; notes section 7: 1.26-1.45 from saturated rates), pure-decode per-stream rate at 34-36 running 9.60 vs 10.20 tok/s (d = 0.94), generated tokens 1 / 1.5 / 2 h after server up 0.89 / 0.89 / 0.91 x the pod (the stretch below predicts 0.88 / 0.88 / 0.89; `calibrate_twin.py`, in-sample). The report's Kaggle-clock section stretches the pod timeline by (1 + (k - 1) s) / d per 10-s interval (s = prefill share of server time from server.log) and drops attempts that would end after Kaggle's force deadline: central (1.45, 0.94), optimistic (1.3, 1.0), pessimistic (1.6, 0.94). It assumes the same sequence of server states, only slower; in thrash (s up to ~0.75) the gap widens to ~x0.7.
2. **Workload:** eval tasks (public, ~2,607 cells per task, 41% multi-input) stand in for the private 240 tasks / ~335 pairs; a heavier or harder private set shifts KV pressure and accuracy together. Transformed copies are new prompts for the model but the same puzzles: outcomes of a task and its copy are correlated, so the score's spread is that of ~120 independent tasks (the report's cluster bootstrap).
3. **Draw noise:** the stack is not run-to-run deterministic; one replica is one draw (sd ~4 task-weighted points on 120 clusters). Compare candidates on the same workload and seed, paired by task (`replica_report.py --baseline`).
4. **Restarts:** a pod reloads the model from local disk (~110-175 s); Kaggle reloads from the FUSE page cache (untested). The Kaggle-clock stretch does not lengthen outages.
5. **Host:** CPU count and speed differ (tool execution is ~0.5% of attempt time on pods); the pod's `free -g` shows the whole host, so use cgroup / RSS for the 173 GiB limit.
6. **Before T0:** Kaggle's container start and input mounting before the first cell (~6-9 s visible in the kernel log; the rest unknown) count against 12 h but not against the solver's deadlines.
"""
(d / "fidelity.md").write_text(md)
print(f"wrote {d / 'replica.json'} and {d / 'fidelity.md'}")
PY
if [ "$DRY" = 1 ]; then echo "[dry-run] scp replica.json fidelity.md -> $R/" >&2; else
  "${SCP[@]}" "$LOCAL/replica.json" "$LOCAL/fidelity.md" "$USER_@$IP:$R/"; fi

cat <<EOF

started $NAME on $IP:$PORT (workload $WLN, deadline shift ${SHIFT}s).
Mirror the run (read-only) and write the report:
  rsync -a --exclude scratch -e "ssh -p $PORT" $USER_@$IP:$R/ $LOCAL/
  .venv/bin/python experiments/E055_kaggle_replica/toolkit/replica_report.py $LOCAL --out $LOCAL/report.md
EOF
