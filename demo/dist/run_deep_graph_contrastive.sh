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
readonly FLAGS_node_config=""
readonly FLAGS_node_feature="${DATASET_DIR}/node_feature"
readonly FLAGS_negative_sampler_type=0
readonly FLAGS_neighbor_sampler_type=0
readonly FLAGS_gs_thread_num=8

# trainer & predictor flags
readonly FLAGS_cs_addr="127.0.0.1:61000"
readonly FLAGS_ps_addrs="127.0.0.1:60000;127.0.0.1:60001"
readonly FLAGS_model="deep_graph_contrastive"
readonly FLAGS_model_config="config=0:54:64;depth=2;dim=128;alpha=0.1;sparse=1;use_neigh_feat=0;tau=0.4"
readonly FLAGS_instance_reader="deep_graph_contrastive_inst_reader"
readonly FLAGS_optimizer="adam"
readonly FLAGS_optimizer_config="rho1=0.9;rho2=0.999;alpha=5e-4;beta=1e-8"
readonly FLAGS_epoch=200
readonly FLAGS_in="${FLAGS_node_graph}"
readonly FLAGS_out_model="model"
readonly FLAGS_out_predict="embedding"

################################################################
# Train
################################################################
FLAGS_sub_command="train"
FLAGS_instance_reader_config="num_neighbors=50,50;use_neigh_feat=0;left_edge_drop_prob=0.1;right_edge_drop_prob=0.3;left_feat_mask_prob=0.2;right_feat_mask_prob=0.4"
FLAGS_batch=2708
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
FLAGS_instance_reader_config="num_neighbors=50,50;use_neigh_feat=0;is_train=0"
FLAGS_in_model="${FLAGS_out_model}"
FLAGS_batch=1
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
evaluate_embedding "${TEST_LABELS}" "${TRAIN_LABELS}" "${FLAGS_out_predict}" "SGD"
