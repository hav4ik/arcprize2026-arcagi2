#!/bin/bash
# serve_dummy.sh <ckpt dir> <tag> [dcp=4] [cuda_visible=0,1,2,3] [lmo=0] : B32-M's exact server command (E064b replica's
# server_cmd.json, minus --language-model-only unless lmo=1) on a --load-format dummy checkpoint, port 8100.
set -u
CK=$1; TAG=$2; DCP=${3:-4}; CVD=${4:-0,1,2,3}; LMO=${5:-0}
V=/root/b32m/venv; OUT=/root/e062pl/runs/$TAG; mkdir -p $OUT /root/e062pl/scratch/cache
eval "$($V/bin/python -I -c '
import json, shlex, sys
env = json.load(open("/root/e062pl/server_env.json"))
for k, v in env.items():
    print("export %s=%s" % (k, shlex.quote(v.replace("__MODEL__", sys.argv[1]))))' "$CK")"
export CUDA_VISIBLE_DEVICES=$CVD
mapfile -t ARGS < <(sed -e "s#__MODEL__#$CK#" -e "s#__DCP__#$DCP#" /root/e062pl/server_args.txt)
[ "$LMO" = 1 ] && ARGS+=("--language-model-only")
[ -n "${EXTRA_SERVE_ARGS:-}" ] && ARGS+=(${EXTRA_SERVE_ARGS})
{ echo "$(date -u +%FT%TZ) serve $TAG dcp=$DCP cvd=$CVD lmo=$LMO"; printf '%s\n' "${ARGS[@]}"; env | grep -E '^(VLLM|DSV41|E043|CUDA_VISIBLE)' | sort; } > $OUT/meta.txt
setsid nohup $V/bin/vllm serve "${ARGS[@]}" > $OUT/server.log 2>&1 < /dev/null &
echo $! > $OUT/server.pid
echo "started $TAG pid $(cat $OUT/server.pid)"
