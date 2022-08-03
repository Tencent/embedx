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

readonly DATASET="semisup_graphsage"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly TRAINING_DATA="${DATASET_DIR}/training_data"
readonly TRAIN_LABELS="${DATASET_DIR}/train_labels"
readonly TEST_LABELS="${DATASET_DIR}/test_labels"

# graph flags
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_gs_thread_num=8
readonly FLAGS_out="average_feature"
readonly FLAGS_neighbor_feature="${FLAGS_out}"

# trainer & predictor flags
readonly FLAGS_thread_num=8
readonly FLAGS_model="semisup_graphsage"
readonly FLAGS_model_config="config=0:1500:128;sparse=1;depth=1;dim=128;alpha=0;max_label=1;multi_label=0;num_label=1;use_neigh_feat=1"
readonly FLAGS_instance_reader="semisup_graphsage"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=0.001;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_epoch=1
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

################################################################
# Average feature
################################################################
run_average_feature ${DATASET}

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neighbors=10;max_label=1;multi_label=0;num_label=1;use_neigh_feat=1"
FLAGS_in="${TRAINING_DATA}"
FLAGS_batch=512
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_instance_reader_config="num_neighbors=10;use_neigh_feat=1;is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_batch=256
FLAGS_target_type=2
run_predictor ${DATASET}

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "SGD"
