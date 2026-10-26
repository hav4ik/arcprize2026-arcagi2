#!/bin/bash
# Poll the Runpod API until a pod not in KNOWN has a public ssh port; print "id ip port" and exit.
cd /home/vu/Workspace/ARC-AGI-2
KNOWN="$*"
while true; do
  R=$(RUNPOD_KEY="$(sed -n 's/^runpod_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")" KNOWN="$KNOWN" .venv/bin/python - 2>/dev/null <<'PY'
import os, httpx
q = "query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort } } } } }"
d = httpx.post("https://api.runpod.io/graphql", json={"query": q}, headers={"Authorization": f"Bearer {os.environ['RUNPOD_KEY']}"}, timeout=30).json()
for p in d["data"]["myself"]["pods"]:
    if p["id"] in os.environ["KNOWN"].split():
        continue
    for x in ((p.get("runtime") or {}).get("ports") or []):
        if x.get("isIpPublic") and x["privatePort"] == 22:
            print(p["id"], x["ip"], x["publicPort"])
            break
PY
)
  [ -n "$R" ] && { echo "$R" | head -1; exit 0; }
  sleep 45
done
