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
# Author: Zhitao Wang (wztzenk@gmail.com)
#

set -e
cd "$(dirname "$0")"
source runtime.sh

readonly DATASET="ppi"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly TRAIN_LABELS="${DATASET_DIR}/train_labels"
readonly TEST_LABELS="${DATASET_DIR}/test_labels"
readonly GROUP_CONFIG="${DATASET_DIR}/group_config.txt"
readonly EDGE="edge"
readonly AVERAGE_FEATURE="average_feature"

# graph flags
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_negative_sampler_type=1
readonly FLAGS_gs_thread_num=8

# trainer & predictor flags
readonly FLAGS_thread_num=8
readonly FLAGS_model="bipartite_plnlp"
readonly FLAGS_model_config="config=${GROUP_CONFIG};depth=1;dim=128;alpha=0.1;sparse=1;use_neigh_feat=1;num_neg=10;decoder_name=DOT;fc_dims=128"
readonly FLAGS_instance_reader="bipartite_plnlp"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_model_shard=10
readonly FLAGS_out_model="bipartite_plnlp_model"
readonly FLAGS_out_predict="bipartite_plnlp_embedding"

# random walker flags
readonly FLAGS_walk_length=5
readonly FLAGS_dump_type=1

################################################################
# Random walk
################################################################
FLAGS_epoch=50
FLAGS_out="${EDGE}"
run_random_walker ${DATASET}

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neg=10;num_neighbors=10;use_neigh_feat=1;user_ns_id=0;item_ns_id=1"
FLAGS_in="${EDGE}"
FLAGS_epoch=1
FLAGS_batch=512
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_neighbor_feature="${AVERAGE_FEATURE}"
FLAGS_instance_reader_config="num_neighbors=10;use_neigh_feat=1;is_train=0;user_ns_id=0;item_ns_id=1"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_batch=256
FLAGS_target_type=2
run_predictor ${DATASET}

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "SGD"
