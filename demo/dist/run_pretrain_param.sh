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

readonly DATASET="dssm"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly USER_GROUP_CONFIG="${DATASET_DIR}/user_group_config.txt"
readonly ITEM_GROUP_CONFIG="${DATASET_DIR}/item_group_config.txt"

readonly PS_NUM=2
readonly WK_NUM=8

# graph flags
readonly FLAGS_node_config="${DATASET_DIR}/freq_file_ns_config"

# trainer & predictor flags
readonly FLAGS_cs_addr="127.0.0.1:61000"
readonly FLAGS_ps_addrs="127.0.0.1:60000;127.0.0.1:60001"
readonly FLAGS_gnn_model=false
readonly FLAGS_deep_model=true
readonly FLAGS_model="dssm"
readonly FLAGS_model_config="user_config=${USER_GROUP_CONFIG};item_config=${ITEM_GROUP_CONFIG};dim=64,32;alpha=0.1;num_neg=20;sparse=1"
readonly FLAGS_instance_reader="dssm"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_in="${DATASET_DIR}/training_data"
readonly FLAGS_pretrain_path="${DATASET_DIR}/pretrain_param.txt"
readonly FLAGS_freq_file="${DATASET_DIR}/freq_file"
readonly FLAGS_item_feature="${DATASET_DIR}/item_feature"
readonly FLAGS_target_type=0
readonly FLAGS_out_model="model"

################################################################
# Train
################################################################
FLAGS_sub_command="train"
FLAGS_instance_reader_config="num_neg=20;add_node=0"

run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
stop_unfinished_dist_worker
