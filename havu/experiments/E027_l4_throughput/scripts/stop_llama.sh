#!/bin/bash
for p in $(pgrep -f 'bin/llama-server'); do kill $p; done
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
