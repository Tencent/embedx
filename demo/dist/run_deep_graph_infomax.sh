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

readonly PS_NUM=2
readonly WK_NUM=8

# graph flags
readonly FLAGS_dist=1
readonly FLAGS_gs_addrs="127.0.0.1:8888;127.0.0.1:8889"
readonly FLAGS_gs_shard_num=2
readonly FLAGS_gs_worker_num=4
readonly FLAGS_node_graph="${DATASET_DIR}/context"
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_gs_thread_num=8
readonly FLAGS_out="sequence"

# trainer & predictor flags
readonly FLAGS_cs_addr="127.0.0.1:61000"
readonly FLAGS_ps_addrs="127.0.0.1:60000;127.0.0.1:60001"
readonly FLAGS_ps_thread_num=10
readonly FLAGS_model="deep_graph_infomax"
readonly FLAGS_model_config="config=0:54:64;depth=2;dim=128;alpha=0.1;sparse=1;use_neigh_feat=0"
readonly FLAGS_instance_reader="deep_graph_infomax_inst_reader"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8"
readonly FLAGS_batch=512
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

# random walker flags
readonly FLAGS_walk_length=20

################################################################
# Random walk
################################################################
FLAGS_epoch=1
run_graph_server ${DATASET}
run_dist_random_walk ${DATASET}
wait_task_finish "wk" ${FLAGS_gs_worker_num}
close_graph_server

################################################################
# Train
################################################################
FLAGS_sub_command="train"
FLAGS_instance_reader_config="num_neg=10;num_neighbors=10,10;use_neigh_feat=0;shuffle_type=0"
FLAGS_epoch=200
FLAGS_in="${FLAGS_out}"
FLAGS_target_type=0
run_graph_server ${DATASET}
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
close_graph_server

################################################################
# Predict
################################################################
FLAGS_sub_command="predict"
FLAGS_instance_reader_config="num_neighbors=10,10;use_neigh_feat=0;is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_in="${FLAGS_node_graph}"
FLAGS_target_type=2
run_graph_server ${DATASET}
run_dist_server ${DATASET} ${PS_NUM}
run_dist_worker ${DATASET} ${WK_NUM}
wait_task_finish "ps" ${PS_NUM}
close_graph_server

################################################################
# Evaluate
################################################################
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "SGD"
