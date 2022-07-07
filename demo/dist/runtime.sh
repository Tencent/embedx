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

# dist flags
FLAGS_sub_command="train"
FLAGS_role="ps"
FLAGS_cs_addr="127.0.0.1:61000"
FLAGS_ps_addrs="127.0.0.1:60000"
FLAGS_ps_id=0
FLAGS_ps_thread_num=1
FLAGS_gnn_model=true
FLAGS_deep_model=false
FLAGS_model="unsup_graphsage"
FLAGS_model_config=""
FLAGS_instance_reader="unsup_graphsage"
FLAGS_instance_reader_config=""
FLAGS_optimizer="adam"
FLAGS_optimizer_config=""
FLAGS_epoch=1
FLAGS_batch=32
FLAGS_in_model=""
FLAGS_in=""
FLAGS_pretrain_path=""
FLAGS_item_feature=""
FLAGS_inst_file=""
FLAGS_freq_file=""
FLAGS_shuffle=true
FLAGS_ts_enable=false
FLAGS_ts_now=0
FLAGS_ts_expire_threshold=0
FLAGS_verbose=1
FLAGS_seed=9527
FLAGS_target_type=2
FLAGS_out_model_remove_zeros=false
FLAGS_out_model=""
FLAGS_out_model_text=""
FLAGS_out_model_fkv=""
FLAGS_out_model_fkv_pb_version=2
FLAGS_out_predict=""

get_dist_flags() {
  local flags=""
  local flags="${flags} --sub_command=${FLAGS_sub_command}"
  local flags="${flags} --role=${FLAGS_role}"
  local flags="${flags} --cs_addr=${FLAGS_cs_addr}"
  local flags="${flags} --ps_addrs=${FLAGS_ps_addrs}"
  local flags="${flags} --ps_id=${FLAGS_ps_id}"
  local flags="${flags} --ps_thread_num=${FLAGS_ps_thread_num}"
  local flags="${flags} --gnn_model=${FLAGS_gnn_model}"
  local flags="${flags} --deep_model=${FLAGS_deep_model}"
  local flags="${flags} --model=${FLAGS_model}"
  local flags="${flags} --model_config=${FLAGS_model_config}"
  local flags="${flags} --instance_reader=${FLAGS_instance_reader}"
  local flags="${flags} --instance_reader_config=${FLAGS_instance_reader_config}"
  local flags="${flags} --optimizer=${FLAGS_optimizer}"
  local flags="${flags} --optimizer_config=${FLAGS_optimizer_config}"
  local flags="${flags} --epoch=${FLAGS_epoch}"
  local flags="${flags} --batch=${FLAGS_batch}"
  local flags="${flags} --in_model=${FLAGS_in_model}"
  local flags="${flags} --in=${FLAGS_in}"
  local flags="${flags} --pretrain_path=${FLAGS_pretrain_path}"
  local flags="${flags} --item_feature=${FLAGS_item_feature}"
  local flags="${flags} --inst_file=${FLAGS_inst_file}"
  local flags="${flags} --freq_file=${FLAGS_freq_file}"
  local flags="${flags} --shuffle=${FLAGS_shuffle}"
  local flags="${flags} --ts_enable=${FLAGS_ts_enable}"
  local flags="${flags} --ts_now=${FLAGS_ts_now}"
  local flags="${flags} --ts_expire_threshold=${FLAGS_ts_expire_threshold}"
  local flags="${flags} --verbose=${FLAGS_verbose}"
  local flags="${flags} --seed=${FLAGS_seed}"
  local flags="${flags} --target_type=${FLAGS_target_type}"
  local flags="${flags} --out_model_remove_zeros=${FLAGS_out_model_remove_zeros}"
  local flags="${flags} --out_model=${FLAGS_out_model}"
  local flags="${flags} --out_model_text=${FLAGS_out_model_text}"
  local flags="${flags} --out_model_fkv=${FLAGS_out_model_fkv}"
  local flags="${flags} --out_model_fkv_pb_version=${FLAGS_out_model_fkv_pb_version}"
  local flags="${flags} --out_predict=${FLAGS_out_predict}"
  echo "${flags}"
}

# average feature flags
FLAGS_sample_num=100000000
get_average_feature_flags() {
  local flags=""
  local flags="${flags} $(get_graph_flags)"
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
  local flags="${flags} $(get_graph_flags)"
  local flags="${flags} --walk_length=${FLAGS_walk_length}"
  local flags="${flags} --dump_type=${FLAGS_dump_type}"
  local flags="${flags} --epoch=${FLAGS_epoch}"
  local flags="${flags} --meta_path_config=${FLAGS_meta_path_config}"
  echo "${flags}"
}

get_dist_trainer_flags() {
  local flags=""
  local flags="${flags} $(get_graph_flags)"
  local flags="${flags} $(get_dist_flags)"
  echo "${flags}"
}

prepare_data() {
  local dataset=$1
  local dataset_dir=${DEMO_DIR}/data/${dataset}
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Preparing ${dataset} dataset ..."
    bash "${DEMO_DIR}"/data/prepare_"${dataset}".sh
  fi
}

run_dist_average_feature() {
  local dataset=$1
  prepare_data ${dataset}

  rm -rf ${FLAGS_out} ${FLAGS_success_out}
  rm -rf *.finish
  log "Run dist average feature ..."
  for ((i = 0; i < ${FLAGS_gs_worker_num}; ++i)); do
    ${AVERAGE_FEATURE_MAIN} $(get_average_feature_flags) "--gs_worker_id=$i" \
      >$(get_average_feature_log $i) 2>&1 &&
      touch $(get_wk_finished $i) &
  done
  log "Done."
}

run_dist_random_walk() {
  local dataset=$1
  prepare_data ${dataset}

  rm -rf ${FLAGS_out} ${FLAGS_success_out}
  rm -rf *.finish
  log "Run dist random walk ..."
  for ((i = 0; i < ${FLAGS_gs_worker_num}; ++i)); do
    ${RANDOM_WALKER_MAIN} $(get_random_walker_flags) "--gs_worker_id=$i" \
      >$(get_random_walker_log $i) 2>&1 &&
      touch $(get_wk_finished $i) &
  done
  log "Done."
}

run_graph_server() {
  local dataset=$1
  prepare_data ${dataset}

  log "Run graph server ..."
  for ((i = 0; i < ${FLAGS_gs_shard_num}; ++i)); do
    ${GRAPH_SERVER_MAIN} $(get_graph_flags) "--gs_shard_id=$i" >$(get_gs_log $i) 2>&1 &
  done
  log "Done."
}

close_graph_server() {
  log "Close graph server ..."
  ${CLOSE_SERVER_MAIN} "--gs_addrs=${FLAGS_gs_addrs}" >close.log 2>&1
  log "Done."
}

run_dist_server() {
  local dataset=$1
  local ps_num=$2
  prepare_data "${dataset}"

  if [[ "${FLAGS_sub_command}" == "train" ]]; then
    rm -rf ${FLAGS_model}
  fi

  rm -rf *.finish
  log "Run dist server ..."
  for ((i = 0; i < ${ps_num}; ++i)); do
    flags="$(get_dist_trainer_flags) --role=ps --ps_id=$i"
    ${DIST_TRAINER} ${flags} >${FLAGS_sub_command}_$(get_ps_log $i) 2>&1 && touch $(get_ps_finished $i) &
  done
  log "Done."
}

run_dist_worker() {
  local dataset=$1
  local wk_num=$2
  prepare_data "${dataset}"

  if [[ "${FLAGS_sub_command}" == "predict" ]]; then
    rm -rf ${FLAGS_out_predict}
  fi

  rm -rf *.finish
  log "Run dist worker ..."
  for ((i = 0; i < ${wk_num}; ++i)); do
    flags="$(get_dist_trainer_flags) --role=wk"
    ${DIST_TRAINER} ${flags} >${FLAGS_sub_command}_$(get_wk_log $i) 2>&1 && touch $(get_wk_finished $i) &
  done
  log "Done."
}

wait_task_finish() {
  local role=$1 # ps or wk
  local task_num=$2
  while [ 0 ]; do
    for ((i = 0; i < ${task_num}; ++i)); do
      if test ! -f $(get_${role}_finished $i); then
        break
      fi
      if test $i == $((${task_num} - 1)); then
        return
      fi
    done
    sleep 1 # magic number
  done
}

stop_unfinished_dist_worker() {
  sleep 5 # magic number
  ps aux | grep dist_trainer | grep role=wk | grep -v grep | awk '{print $2}' | xargs kill -STOP >/dev/null 2>&1
  sleep 1 # magic number
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
