#!/usr/bin/env bash
# Start the code-execution sandbox for LLM tool use (NeMo-Skills IPython sandbox server, the image used in
# AIMO-proof-pilot). Prints the URL to pass to evals/harness code.
#
# Isolation:
# - network: Docker network created with --internal (no route to the internet; the host can still reach the
#   container on the bridge). Verified: raw-IP connect -> ENETUNREACH, DNS fails.
#   NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1 additionally blocks sockets inside the Python sessions.
# - filesystem: no host directories mounted except the patched server file (read-only), so the ARC data on
#   this machine (including solutions) is not visible to executed code.
# - main.py: the image's /app/main.py with one fix: SIGINT handler SIG_DFL -> default_int_handler, so a
#   timeout raises KeyboardInterrupt in the cell instead of killing the shell (session state survives).
set -euo pipefail
IMAGE=locally-built-dockerfiles-dockerfile-sandbox:c87907bf1ce9
NAME=${NAME:-arc-sandbox}
NET=arc-sandbox-net
HERE=$(cd "$(dirname "$0")" && pwd)
# Sessions idle longer than this are stopped by the server (its built-in reaper, off by default).
# The longest recorded attempt took 15.4 min, so 30 min only removes finished sessions.
MEMORY=${SANDBOX_MEMORY:-40g}  # container memory cap (RAM = swap)
SESSION_TIMEOUT=${SANDBOX_SESSION_TIMEOUT:-1800}

docker network inspect "$NET" >/dev/null 2>&1 || docker network create --internal "$NET" >/dev/null
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" --network "$NET" \
  --memory "$MEMORY" --memory-swap "$MEMORY" \
  -e NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1 \
  -e NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT="$SESSION_TIMEOUT" \
  -v "$HERE/main.py:/app/main.py:ro" \
  "$IMAGE" >/dev/null
IP=$(docker inspect "$NAME" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
for _ in $(seq 60); do curl -s -m 2 "http://$IP:6000/health" | grep -q healthy && break; sleep 2; done
curl -s -m 2 "http://$IP:6000/health" | grep -q healthy || { echo "sandbox failed to start" >&2; exit 1; }
echo "http://$IP:6000"
