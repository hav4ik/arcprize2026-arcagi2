#!/bin/bash
# enqueue.sh <pod2|pod3> <job file> [target name]: copy a job file to the pod's pending queue (oldest first)
declare -A P=([pod2]="18070 213.173.105.27" [pod3]="29668 157.157.221.29"); set -- "$@"; read -r PORT IP <<< "${P[$1]:?pod2|pod3}"
J=$2; N=${3:-$(basename $J)}; scp -q -P $PORT "$J" root@$IP:/root/queue/pending/$N && ssh -p $PORT root@$IP 'ls -1tr /root/queue/pending'
