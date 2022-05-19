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
readonly DEMO_DIR=$(dirname "${PWD}")
source "${DEMO_DIR}"/env.sh
source "${DEMO_DIR}"/common.sh

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
    local flags=
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
get_random_walker_flags() {
    local flags=""
    local flags="$(get_graph_flags)"
    local flags="${flags} --walk_length=${FLAGS_walk_length}"
    local flags="${flags} --dump_type=${FLAGS_dump_type}"
    local flags="${flags} --epoch=${FLAGS_epoch}"
    echo "${flags}"
}

# trainer & predictor flags
FLAGS_thread_num=1
FLAGS_gnn_model=true
FLAGS_deep_model=false
FLAGS_model="unsup_graphsage"
FLAGS_model_config=""
FLAGS_instance_reader="unsup_graphsage"
FLAGS_instance_reader_config=""
FLAGS_optimizer="adam"
FLAGS_optimizer_config=""
FLAGS_model_shard=0
FLAGS_in_model=""
FLAGS_in=""
FLAGS_pretrain_path=""
FLAGS_item_feature=""
FLAGS_inst_file=""
FLAGS_freq_file=""
FLAGS_shuffle=true
FLAGS_epoch=1
FLAGS_batch=32
FLAGS_ts_enable=false
FLAGS_ts_now=0
FLAGS_ts_expire_threshold=0
FLAGS_verbose=1
FLAGS_seed=9527
FLAGS_target_type=2
FLAGS_out_model_remove_zeros=false
FLAGS_out_model=""
FLAGS_out_model_text=""
FLAGS_out_predict=""
get_trainer_flags() {
    local flags=""
    local flags="$(get_graph_flags)"
    local flags="${flags} --thread_num=${FLAGS_thread_num}"
    local flags="${flags} --gnn_model=${FLAGS_gnn_model}"
    local flags="${flags} --deep_model=${FLAGS_deep_model}"
    local flags="${flags} --model=${FLAGS_model}"
    local flags="${flags} --model_config=${FLAGS_model_config}"
    local flags="${flags} --instance_reader=${FLAGS_instance_reader}"
    local flags="${flags} --instance_reader_config=${FLAGS_instance_reader_config}"
    local flags="${flags} --optimizer=${FLAGS_optimizer}"
    local flags="${flags} --optimizer_config=${FLAGS_optimizer_config}"
    local flags="${flags} --model_shard=${FLAGS_model_shard}"
    local flags="${flags} --in_model=${FLAGS_in_model}"
    local flags="${flags} --in=${FLAGS_in}"
    local flags="${flags} --pretrain_path=${FLAGS_pretrain_path}"
    local flags="${flags} --item_feature=${FLAGS_item_feature}"
    local flags="${flags} --inst_file=${FLAGS_inst_file}"
    local flags="${flags} --freq_file=${FLAGS_freq_file}"
    local flags="${flags} --shuffle=${FLAGS_shuffle}"
    local flags="${flags} --epoch=${FLAGS_epoch}"
    local flags="${flags} --batch=${FLAGS_batch}"
    local flags="${flags} --ts_enable=${FLAGS_ts_enable}"
    local flags="${flags} --ts_now=${FLAGS_ts_now}"
    local flags="${flags} --ts_expire_threshold=${FLAGS_ts_expire_threshold}"
    local flags="${flags} --verbose=${FLAGS_verbose}"
    local flags="${flags} --seed=${FLAGS_seed}"
    local flags="${flags} --target_type=${FLAGS_target_type}"
    local flags="${flags} --out_model_remove_zeros=${FLAGS_out_model_remove_zeros}"
    local flags="${flags} --out_model=${FLAGS_out_model}"
    local flags="${flags} --out_model_text=${FLAGS_out_model_text}"
    echo "${flags}"
}

get_predictor_flags() {
    local flags=""
    local flags="$(get_graph_flags)"
    local flags="${flags} --thread_num=${FLAGS_thread_num}"
    local flags="${flags} --gnn_model=${FLAGS_gnn_model}"
    local flags="${flags} --instance_reader=${FLAGS_instance_reader}"
    local flags="${flags} --instance_reader_config=${FLAGS_instance_reader_config}"
    local flags="${flags} --batch=${FLAGS_batch}"
    local flags="${flags} --in=${FLAGS_in}"
    local flags="${flags} --in_model=${FLAGS_in_model}"
    local flags="${flags} --target_type=${FLAGS_target_type}"
    local flags="${flags} --verbose=${FLAGS_verbose}"
    local flags="${flags} --out_predict=${FLAGS_out_predict}"
    echo "${flags}"
}

prepare_data() {
    local dataset=$1
    local dataset_dir="${DEMO_DIR}"/data/${dataset}

    if [[ ! -d "${dataset_dir}" ]]; then
        echo "Preparing ${dataset} dataset ..."
        bash "${DEMO_DIR}"/data/prepare_"${dataset}".sh
    fi
}

run_average_feature() {
    local dataset=$1
    prepare_data "${dataset}"

    rm -rf "${FLAGS_out}" "${FLAGS_success_out}"
    log "Run average feature ..."
    ${AVERAGE_FEATURE_MAIN} $(get_average_feature_flags) >$(get_average_feature_log) 2>&1
    log "Done."
}

run_random_walker() {
    local dataset=$1
    prepare_data "${dataset}"

    rm -rf "${FLAGS_out}" "${FLAGS_success_out}"
    log "Run random walker ..."
    ${RANDOM_WALKER_MAIN} $(get_random_walker_flags) >$(get_random_walker_log) 2>&1
    log "Done."
}

run_trainer() {
    local dataset=$1
    prepare_data "${dataset}"

    rm -rf "${FLAGS_out_model}" "${FLAGS_out_model_text}" "${FLAGS_out_model_fkv}"
    log "Run trainer ..."
    ${TRAINER} $(get_trainer_flags) >$(get_trainer_log) 2>&1
    log "Done."
}

run_predictor() {
    local dataset=$1
    prepare_data "${dataset}"

    rm -rf "${FLAGS_out_predict}"
    log "Run predictor ..."
    ${PREDICTOR} $(get_predictor_flags) >$(get_predictor_log) 2>&1
    log "Done."
}

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
