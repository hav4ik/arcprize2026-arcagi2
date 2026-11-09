#!/bin/bash
while true; do echo "$(date -u +%T) used_gib=$(free -g | awk 'NR==2{print $3}') rss_gib=$(ps -eo rss= | awk '{s+=$1} END {printf "%.1f", s/1048576}') mlocked_kb=$(grep Mlocked /proc/meminfo | awk '{print $2}')" >> $1; sleep 60; done
