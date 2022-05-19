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

readonly DATASET="youtube_dnn"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly GROUP_CONFIG="${DATASET_DIR}/group_config.txt"
readonly TRAINING_DATA="${DATASET_DIR}/training_data"
readonly PREDICT_ITEM_IN="${DATASET_DIR}/predict_item_embed_in"
readonly USER_EMBEDDING="user_embedding"
readonly ITEM_EMBEDDING="item_embedding"

# graph flags
readonly FLAGS_node_config="${DATASET_DIR}/freq_file_ns_config"

# trainer & predictor flags
readonly FLAGS_thread_num=8
readonly FLAGS_gnn_model=false
readonly FLAGS_deep_model=true
readonly FLAGS_model="youtube_dnn"
readonly FLAGS_model_config="config=${GROUP_CONFIG};dim=64,32;alpha=0.1;num_neg=20;label_group_id=58;sparse=1"
readonly FLAGS_instance_reader="youtube_dnn"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_freq_file="${DATASET_DIR}/freq_file"
readonly FLAGS_epoch=10
readonly FLAGS_out_model="model"

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neg=20"
FLAGS_in="${TRAINING_DATA}"
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict user embedding
################################################################
FLAGS_instance_reader_config="is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${TRAINING_DATA}"
FLAGS_target_type=2
FLAGS_out_predict="${USER_EMBEDDING}"
run_predictor ${DATASET}

################################################################
# Predict item embedding
################################################################
FLAGS_instance_reader_config="is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${PREDICT_ITEM_IN}"
FLAGS_target_type=3
FLAGS_out_predict="${ITEM_EMBEDDING}"
run_predictor ${DATASET}

################################################################
# Evaluate
################################################################
evaluate_hit_rate "${USER_EMBEDDING}" "${ITEM_EMBEDDING}" "50,100,200,500"
