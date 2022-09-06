#! /bin/bash
#
# Tencent is pleased to support the open source community by making embedx
# available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the BSD 3-Clause License and other third-party components,
# please refer to LICENSE for details.
#
# Author: Chunchen Su (chunchen.scut@gmail.com)
#

cd $(dirname $0)

source <(awk '/GS_MACHINES=/,/\)/' run_dist.sh)
source <(awk '/DIST_MACHINES=/,/\)/' run_dist.sh)

MACHINES=(${GS_MACHINES[@]} ${DIST_MACHINES[@]})
MACHINES=($(echo "${MACHINES[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

user=$(whoami)
echo user=${user}
for machine in "${MACHINES[@]}"; do
    echo machine=${machine}
    remote_dir=/tmp
    # If 'ssh' is unavailable, you need to copy 'run_dist.sh'
    # to corresponding machine and run it manually.
    scp -pq run_dist.sh ${user}@${machine}:${remote_dir}
    ssh ${user}@${machine} 'cd '"${remote_dir}"' && bash run_dist.sh'
done
