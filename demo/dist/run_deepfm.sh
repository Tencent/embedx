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

set -e
cd "$(dirname "$0")"
source runtime.sh

readonly DATASET="ctr"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly GROUP_CONFIG="${DATASET_DIR}/feature_group_config.txt"

readonly PS_NUM=2
readonly WK_NUM=8

# trainer & predictor flags
readonly FLAGS_cs_addr="127.0.0.1:61000"
readonly FLAGS_ps_addrs="127.0.0.1:60000;127.0.0.1:60001"
readonly FLAGS_gnn_model=false
readonly FLAGS_deep_model=true
readonly FLAGS_model="deepfm"
readonly FLAGS_model_config="config=${GROUP_CONFIG};deep_dims=64,32;sparse=1"
readonly FLAGS_instance_reader="libsvm"
readonly FLAGS_instance_reader_config="w=0"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_epoch=10
readonly FLAGS_in="${DATASET_DIR}/libsvm.txt"
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="probs"

################################################################
# Train
################################################################
FLAGS_sub_command="train"
FLAGS_target_type=0
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
stop_unfinished_dist_worker

################################################################
# Predict
################################################################
FLAGS_sub_command="predict"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_target_type=1
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
stop_unfinished_dist_worker
