#!/bin/bash
# E059 restart-under-load (B32 on L4, 90 heaviest tasks, 3 h, TP1 worker killed at +1.5 h); was: E055 final candidate B replica: next-candidate code + auto-repair + S15/S16/S17 (solver 77fe80a3), multi-test,
# interval 512, E054 indexer skip (#156), E051 KV offload 32 GiB private, x56 + KV gate 0.85 + hold on server queue;
# transformed eval x2 workload (evalx2t_s2026), toolkit deadlines (-600 s), util 0.938 (Kaggle's KV pool).
cd "$(dirname "$0")/../E055_kaggle_replica"
export REPLICA_SHIFT_S=0
exec toolkit/replica_run.sh 43647 157.157.221.29 b32kill_heavy evalx2t_s2026 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/dsv41/venv_offload \
  PATH=/root/dsv41/venv_offload/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=56 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  ARC_TASK_IDS=981571dc,f081d783,d8e07eb2,c69ce4ab,9aaea919,64048db9,b99e7126,d188e255,7c66cb00,d6cafd57,0934a4d8,1c1904af,16b78196,1672eaf6,e3721c99,a0865888,5961cc34,9911a8c1,195c6913,5db67b1d,a32d8b75,97147f7d,c4d067a0,61f6573f,e87109e9,db5924cc,aa4ec2a5,86600f13,3e6067c3,c6c2afc7,8f215267,8188a224,20a9e565,9c0a7b40,b10624e5,1ccf5f3d,19adfaff,a25697e4,62593bfd,469596c6,6bf5d7a9,64efde09,a47bf94d,bfe8f473,221dfab4,c7d12dd5,a395ee82,738021fc,b5ca7ac4,67898199,a1b2c775,2c181942,e12f9a14,60799774,4c416de3,93c7021e,8b9c3697,5aac65b7,8b7bacbf,cda7f206,a251c730,66ff7e89,67e490f4,83a256c6,cc390903,36a08778,d5e00a86,6e4f6532,cb2d8a2c,9c516d89,88e364bc,51020d13,7b0280bc,4fd2c393,129a2b37,2ba387bc,7b80bb43,1e8ba212,a5b54137,13e47133,4c7dc4dd,e6ea2d2d,269e22fb,d108c01d,142ca369,712dec6e,719f884d,a6f40cea,cbebaa4b,aed6f368 ARC_STOP_NEW_S=9600 ARC_FORCE_AT_S=10200 ARC_HARD_STOP_S=10500 ARC_KILL_AT_S=10800 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}"]'
