#!/bin/bash
set -euo pipefail
exec /workspace/dsv4/venv/bin/python -u /workspace/dsv4/repo/experiments/E076_active14_13/scripts/batch.py --hours 48 --tools-first
