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

readonly DATASET="blogcatalog"
readonly DATASET_DIR="${DEMO_DIR}/data/${DATASET}"
readonly TRAIN_LABELS="${DATASET_DIR}/train_labels"
readonly TEST_LABELS="${DATASET_DIR}/test_labels"

readonly PS_NUM=2
readonly WK_NUM=8

# graph flags
readonly FLAGS_dist=1
readonly FLAGS_gs_addrs="127.0.0.1:8888;127.0.0.1:8889"
readonly FLAGS_gs_shard_num=2
readonly FLAGS_gs_worker_num=4
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_gs_thread_num=8
readonly FLAGS_out="sequence"

# trainer & predictor flags
readonly FLAGS_cs_addr="127.0.0.1:61000"
readonly FLAGS_ps_addrs="127.0.0.1:60000;127.0.0.1:60001"
readonly FLAGS_ps_thread_num=10
readonly FLAGS_model="deepwalk"
readonly FLAGS_model_config="config=0:15000:128;sparse=1"
readonly FLAGS_instance_reader="deepwalk"
readonly FLAGS_optimizer="adagrad"
readonly FLAGS_optimizer_config="alpha=0.1;beta=1e-6"
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

# random walker flags
readonly FLAGS_walk_length=80

################################################################
# Random walk
################################################################
FLAGS_epoch=50
run_graph_server ${DATASET}
run_dist_random_walk ${DATASET}
wait_task_finish "wk" ${FLAGS_gs_worker_num}
close_graph_server

################################################################
# Train
################################################################
FLAGS_sub_command="train"
FLAGS_instance_reader_config="num_neg=5;window_size=5;is_train=1"
FLAGS_in="${FLAGS_out}"
FLAGS_epoch=1
FLAGS_batch=1
FLAGS_target_type=0
run_graph_server ${DATASET}
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
stop_unfinished_dist_worker
close_graph_server

################################################################
# Predict
################################################################
FLAGS_sub_command="predict"
FLAGS_instance_reader_config="is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_target_type=2
run_graph_server ${DATASET}
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
stop_unfinished_dist_worker
close_graph_server

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "OneVsRest"
