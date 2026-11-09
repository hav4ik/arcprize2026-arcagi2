#!/bin/bash
/root/kgl/bin/python -I dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/8 manh_wheels_files_novllm.tsv /root/b32m/dl > dl_wheels2.log 2>&1
/root/kgl/bin/python -I dl_kaggle_files.py model chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1 kaggle_todo.tsv /root/dsv41/model-e128k > dl_model.log 2>&1
echo CHAIN_DL_DONE >> dl_model.log
