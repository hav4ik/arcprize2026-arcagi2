#!/bin/bash
set -euo pipefail
/workspace/dsv4/venv/bin/python -u /workspace/dsv4/repo/experiments/E074_active_params/scripts/batch.py --hours 14 2>&1 | tee -a /workspace/dsv4/logs/batch.log
