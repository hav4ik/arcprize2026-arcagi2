#!/bin/bash
# Phase E: sequential pool checks (B32-M flags, util 0.938, DCP4, order 0,1,2,3)
D=/root/e062pl
bash $D/measure.sh $D/ckpt/c1_128 c1_128_lmo 4 0,1,2,3 1
bash $D/measure.sh $D/ckpt/c1_128 c1_128_vis 4 0,1,2,3 0
bash $D/measure.sh $D/ckpt/f72_120 f72_120_vis 4 0,1,2,3 0
bash $D/measure.sh $D/ckpt/f72_124 f72_124_vis 4 0,1,2,3 0
echo CHAIN_POOL_DONE
