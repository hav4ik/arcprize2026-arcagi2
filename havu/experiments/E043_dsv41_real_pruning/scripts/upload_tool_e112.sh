#!/bin/bash
# E043: upload the tool E112 2-bit export (built by export_tool_e112.sh, files <= 5 GB) to a private HF model repo.
# Adds the model card (README_tool_e112.md, copied next to this script) and serving/ (the patches this layout needs),
# uploads with upload_folder (as tool E128; tvenv huggingface_hub 2.3 has no upload_large_folder), then compares the repo with the folder by name and size.
# Token: $HF_TOKEN, else the first line of stdin (never on the command line).
set -uo pipefail
[ -n "${HF_TOKEN:-}" ] || read -r HF_TOKEN; export HF_TOKEN
cd /workspace/dsv4
X=/dev/shm/e043/export/tool_e112_w10_eng64_gptq
REPO=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit
cp code/scripts/README_tool_e112.md $X/README.md
mkdir -p $X/serving
cp code/patches/apply_compact_engram.py code/patches/apply_e043_humming.py code/patches/e043_humming.py $X/serving/
rm -rf $X/.cache
sed -i "s/{SIZE}/$(du -sb $X | cut -f1 | python3 -c 'print(f"{int(input()):,}")')/" $X/README.md
grep -n "Size:" $X/README.md
date -u; echo "uploading $X -> $REPO"
for i in 1 2 3; do
  tvenv/bin/python code/scripts/hf_upload.py --repo $REPO --type model --path $X --message "E112 2-bit GPTQ tool model (E043), safetensors <= 5 GB" && break
  echo "upload attempt $i failed; retrying in 60 s"; sleep 60
done
date -u
tvenv/bin/python code/scripts/hf_verify_upload.py --repo $REPO --path $X
echo UPLOAD-TOOL-E112-DONE
