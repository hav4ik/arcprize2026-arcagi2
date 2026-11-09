#!/bin/bash
# Make Manh's cell-1 `dpkg -i cu13-install/*.deb` succeed on this pod as on Kaggle's image (pod 2's fix): install the
# debs once, then `apt-get install -f --no-install-recommends` from the Ubuntu 24.04 sources only (a private source dir,
# because Manh's cuda-keyring adds a cuda list whose Signed-By conflicts with the image's own cuda.list). Then dpkg -i
# again must exit 0. Log: /root/e066/c1text/cu13_deps.log
set -u
L=/root/e066/c1text/cu13_deps.log; exec > >(tee -a $L) 2>&1
echo "fix_cu13_deps $(date -u +%T): $(df -h / | tail -1)"
cd /root/kin/cu13  # the cu13-install v2 copy that /kaggle/input/datasets/nguyennguyen599/cu13-install maps to
dpkg -i cuda-keyring_1.1-1_all.deb > /dev/null 2>&1; dpkg -i ./*.deb > /root/e066/c1text/dpkg1.log 2>&1; echo "dpkg pass 1 rc=$? ($(grep -c 'dependency problems' /root/e066/c1text/dpkg1.log) dependency-problem lines)"
A=/root/apt_ubuntu; mkdir -p $A/sources.list.d; cp /etc/apt/sources.list.d/ubuntu.sources $A/sources.list.d/ 2>/dev/null || grep -h "^deb .*ubuntu" /etc/apt/sources.list > $A/sources.list.d/ubuntu.list
O=(-o Dir::Etc::SourceParts=$A/sources.list.d -o Dir::Etc::SourceList=/dev/null)
apt-get "${O[@]}" update -qq > /dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get "${O[@]}" install -f -y --no-install-recommends 2>&1 | grep -E "^(Setting up|Removing|[0-9]+ upgraded)" | tail -30
dpkg -i ./*.deb > /root/e066/c1text/dpkg2.log 2>&1; echo "dpkg pass 2 rc=$?"; dpkg --audit | head -5; echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | tail -2 | head -1)"; echo "after: $(df -h / | tail -1)"
