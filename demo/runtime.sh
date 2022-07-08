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

cd "$(dirname "$0")"

# graph flags
FLAGS_dist=0
FLAGS_gs_addrs=""
FLAGS_gs_shard_num=1
FLAGS_gs_shard_id=0
FLAGS_gs_worker_num=-1
FLAGS_gs_worker_id=-1
FLAGS_node_graph=""
FLAGS_node_config=""
FLAGS_node_feature=""
FLAGS_neighbor_feature=""
FLAGS_negative_sampler_type=0
FLAGS_neighbor_sampler_type=0
FLAGS_random_walker_type=0
FLAGS_batch_node=128
FLAGS_gs_thread_num=1
FLAGS_cache_thld=0.0
FLAGS_cache_type=1
FLAGS_max_node_per_rpc=2000
FLAGS_out=""
FLAGS_success_out=""
get_graph_flags() {
    local flags=""
    local flags="${flags} --dist=${FLAGS_dist}"
    local flags="${flags} --gs_addrs=${FLAGS_gs_addrs}"
    local flags="${flags} --gs_shard_num=${FLAGS_gs_shard_num}"
    local flags="${flags} --gs_shard_id=${FLAGS_gs_shard_id}"
    local flags="${flags} --gs_worker_num=${FLAGS_gs_worker_num}"
    local flags="${flags} --gs_worker_id=${FLAGS_gs_worker_id}"
    local flags="${flags} --node_graph=${FLAGS_node_graph}"
    local flags="${flags} --node_config=${FLAGS_node_config}"
    local flags="${flags} --node_feature=${FLAGS_node_feature}"
    local flags="${flags} --neighbor_feature=${FLAGS_neighbor_feature}"
    local flags="${flags} --negative_sampler_type=${FLAGS_negative_sampler_type}"
    local flags="${flags} --neighbor_sampler_type=${FLAGS_neighbor_sampler_type}"
    local flags="${flags} --random_walker_type=${FLAGS_random_walker_type}"
    local flags="${flags} --batch_node=${FLAGS_batch_node}"
    local flags="${flags} --gs_thread_num=${FLAGS_gs_thread_num}"
    local flags="${flags} --cache_thld=${FLAGS_cache_thld}"
    local flags="${flags} --cache_type=${FLAGS_cache_type}"
    local flags="${flags} --max_node_per_rpc=${FLAGS_max_node_per_rpc}"
    local flags="${flags} --out=${FLAGS_out}"
    local flags="${flags} --success_out=${FLAGS_success_out}"
    echo "${flags}"
}

# average feature flags
FLAGS_sample_num=100000000
get_average_feature_flags() {
    local flags=""
    local flags="$(get_graph_flags)"
    local flags="${flags} --sample_num=${FLAGS_sample_num}"
    echo "${flags}"
}

# random walker flags
FLAGS_walk_length=10
FLAGS_dump_type=0
FLAGS_epoch=1
FLAGS_meta_path_config=""
get_random_walker_flags() {
    local flags=""
    local flags="$(get_graph_flags)"
    local flags="${flags} --walk_length=${FLAGS_walk_length}"
    local flags="${flags} --dump_type=${FLAGS_dump_type}"
    local flags="${flags} --epoch=${FLAGS_epoch}"
    local flags="${flags} --meta_path_config=${FLAGS_meta_path_config}"
    echo "${flags}"
}

# extern DEMO_DIR
prepare_data() {
    local dataset=$1
    local dataset_dir="${DEMO_DIR}"/data/${dataset}

    if [[ ! -d "${dataset_dir}" ]]; then
        echo "Preparing ${dataset} dataset ..."
        bash "${DEMO_DIR}"/data/prepare_"${dataset}".sh
    fi
}

# extern DEMO_DIR
evaluate_hit_rate() {
    local query_embed="$1"
    local item_embed="$2"
    local topk="$3"

    log "Evaluate hit rate ..."
    python "${DEMO_DIR}"/hit_rate.py \
        --query_embed="${query_embed}" \
        --item_embed="${item_embed}" \
        --topk="${topk}"
    local k=$(echo "${topk}" | awk -F "," '{print NF}')
    awk '/^top/{print $0}' hit_rate.log | tail -n $k
    log "Done."
}

# extern DEMO_DIR
evaluate_embedding() {
    local test_label_file="$1"
    local train_label_file="$2"
    local embedding_file="$3"
    local classifier_name="$4"

    log "Evaluate embedding ..."
    python "${DEMO_DIR}"/evaluate.py \
        --test_label_file="${test_label_file}" \
        --train_label_file="${train_label_file}" \
        --embedding_file="${embedding_file}" \
        --classifier_name="${classifier_name}"
    log $(awk -F ": " '/micro/{f1score=$NF} END {print f1score}' f1score.log)
    log "Done."
}
