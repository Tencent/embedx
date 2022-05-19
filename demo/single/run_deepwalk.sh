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

readonly DATASET="blogcatalog"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly TRAIN_LABELS="${DATASET_DIR}/train_labels"
readonly TEST_LABELS="${DATASET_DIR}/test_labels"

# graph flags
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_gs_thread_num=8
readonly FLAGS_out="sequence"

# trainer & predictor flags
readonly FLAGS_thread_num=8
readonly FLAGS_model="deepwalk"
readonly FLAGS_model_config="config=0:15000:128;sparse=1"
readonly FLAGS_instance_reader="deepwalk"
readonly FLAGS_optimizer="adagrad"
readonly FLAGS_optimizer_config="alpha=0.1;beta=1e-6"
readonly FLAGS_model_shard=10
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

# random walker flags
readonly FLAGS_walk_length=80

################################################################
# Random walk
################################################################
FLAGS_epoch=50
run_random_walker ${DATASET}

################################################################
# Train
################################################################
FLAGS_instance_reader_config="num_neg=5;window_size=5;is_train=1"
FLAGS_in="${FLAGS_out}"
FLAGS_epoch=1
FLAGS_batch=1
FLAGS_target_type=0
run_trainer ${DATASET}

################################################################
# Predict
################################################################
FLAGS_instance_reader_config="is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_epoch=1
FLAGS_batch=256
FLAGS_target_type=2
run_predictor ${DATASET}

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "OneVsRest"
