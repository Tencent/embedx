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
readonly TRAIN_LABELS="${DATASET_DIR}/train_labels"
readonly TEST_LABELS="${DATASET_DIR}/test_labels"

# graph flags
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_gs_thread_num=8

# trainer & predictor flags
readonly FLAGS_thread_num=8
readonly FLAGS_deep_model=true
readonly FLAGS_model="node_infograph"
readonly FLAGS_model_config="config=0:54:64;depth=2;dim=128;alpha=0.1;sparse=1;use_neigh_feat=0;max_label=6;multi_label=0"
readonly FLAGS_instance_reader="node_infograph_inst_reader"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_in="${FLAGS_node_graph}"
readonly FLAGS_inst_file="${TRAIN_LABELS}"
readonly FLAGS_epoch=200
readonly FLAGS_batch=512
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neg=10;num_neighbors=50,10;use_neigh_feat=0;shuffle_type=0;instance_sample_prob=0.2;max_label=6;multi_label=0;train_data_type=1"
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_instance_reader_config="num_neighbors=50,10;use_neigh_feat=0;is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_target_type=2
run_predictor ${DATASET}

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "SGD"
