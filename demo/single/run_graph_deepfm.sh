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

# graph flags
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_node_config="${DATASET_DIR}/namespace_config.txt"

# trainer & predictor flags
readonly FLAGS_thread_num=10
readonly FLAGS_model="graph_deepfm2"
readonly FLAGS_model_config="config=${GROUP_CONFIG};dfm_dims=64,32;sparse=1;depth=1;user_group_id=1;sage_dim=16;alpha=0.1;sage_encoder_type=0"
readonly FLAGS_instance_reader="graph_deepfm2"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="probs"

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neg=10;num_neighbors=20;walk_length=2;window_size=2;user_group_id=1;item_group_id=18"
FLAGS_in="${DATASET_DIR}/libsvm.txt"
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_instance_reader_config="is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_target_type=1
run_predictor ${DATASET}
