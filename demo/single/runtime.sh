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
source "${DEMO_DIR}"/runtime.sh

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
