# filler: replicate of the incumbent (Manh V12 as submitted) on eval120_kaggle; feeds the noise baseline when the queue is empty
JOB_LABEL=@LABEL@
NB_PATH=/root/queue/nb/base/manh_v12.ipynb
NB_SHA256=c47be5a8
WORKLOAD=eval120_kaggle
KT_WHEELS_DIR=$WHEELS_V11
WHEELS_EXPECT_COUNT=294
WHEELS_EXPECT_VLLM=g8cf0c0664b
MODEL_KIND=e128
EXTRA_ENV=""
MAX_KAGGLE_H=7
