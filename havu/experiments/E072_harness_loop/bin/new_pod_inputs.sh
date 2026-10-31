#!/bin/bash
# New-pod inputs for the V12 replica layout (token on stdin only): pod4_scripts/dl_inputs.sh (vllm-wheels v11, tools-kit v5, arc2-2026-src v23,
# cu13-install v2) + unsloth-main v15, then symlinks to the layout make_tree.sh expects, then fix_cu13_deps.sh. Log /root/p4/logs/new_pod_inputs.log
read -r tok; L=/root/p4/logs; exec >> $L/new_pod_inputs.log 2>&1; echo "start $(date -u +%T)"
printf '%s\n' "$tok" | bash /root/p4/scripts/dl_inputs.sh
for i in 1 2 3; do KAGGLE_API_TOKEN="$tok" KAGGLEHUB_CACHE=/root/kin/kh /root/kgl/bin/python -I -c "import kagglehub; print('GOT', kagglehub.dataset_download('nguyennguyen599/unsloth-main/versions/15'))" && break; sleep 30; done
unset tok
ln -sfn /root/kin/kh/datasets /root/kin/datasets; ln -sfn /root/kin/vllm-wheels-v11 /root/kin/vllm_wheels_v11
bash /root/p4/scripts/fix_cu13_deps.sh
echo "NEW_POD_INPUTS_DONE $(date -u +%T)"
