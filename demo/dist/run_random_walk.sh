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
# Author: Shuting Guo (shutingnjupt@gmail.com)
#

set -e
cd "$(dirname "$0")"
source runtime.sh

readonly DATASET="cora"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"

readonly FLAGS_dist=1
readonly FLAGS_gs_shard_num=2
readonly FLAGS_gs_thread_num=8
readonly FLAGS_gs_worker_num=4
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_out="sequence"

################################################################
# Random walk
################################################################
run_graph_server ${DATASET}
run_dist_random_walk ${DATASET}
wait_task_finish "wk" ${FLAGS_gs_worker_num}
close_graph_server
