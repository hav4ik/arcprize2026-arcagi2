#!/bin/bash
# Stop (not terminate) Runpod pods by id: GPUs released, billing for compute ends; the user decides on terminate.
# Usage: runpod_stop.sh <pod_id> [...]
cd /home/vu/Workspace/ARC-AGI-2
for id in "$@"; do
  RUNPOD_KEY="$(sed -n 's/^runpod_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")" POD="$id" .venv/bin/python - <<'PY'
import os, httpx
q = 'mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }'
r = httpx.post("https://api.runpod.io/graphql", json={"query": q, "variables": {"id": os.environ["POD"]}},
               headers={"Authorization": f"Bearer {os.environ['RUNPOD_KEY']}"}, timeout=60).json()
print(os.environ["POD"], r.get("data") or r.get("errors"))
PY
done
