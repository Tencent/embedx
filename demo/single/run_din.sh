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

readonly DATASET="uch"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly USER_GROUP_CONFIG="${DATASET_DIR}/user_group_config.txt"
readonly ITEM_GROUP_CONFIG="${DATASET_DIR}/item_group_config.txt"

# trainer & predictor flags
readonly FLAGS_gnn_model=false
readonly FLAGS_thread_num=8
readonly FLAGS_model="din"
readonly FLAGS_model_config="user_config=${USER_GROUP_CONFIG};item_config=${ITEM_GROUP_CONFIG};deep_dims=64,32;att_hidden_dim=36;hist_size=50;sparse=1"
readonly FLAGS_instance_reader="uch"
readonly FLAGS_instance_reader_config="hist_item_size=50;w=0"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_epoch=10
readonly FLAGS_in="${DATASET_DIR}/uch.txt"
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="probs"

################################################################
# Train
################################################################
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_target_type=1
FLAGS_in_model="${FLAGS_out_model}"
run_predictor ${DATASET}
