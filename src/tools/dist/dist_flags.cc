// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "src/tools/dist/dist_flags.h"

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/ts_store.h>

#include "src/tools/graph/graph_flags.h"
#include "src/tools/shard_func_name.h"

// dist train & predict
DEFINE_string(sub_command, "train", "train or predict.");
DEFINE_string(role, "ps", "ps or wk.");
DEFINE_string(cs_addr, "127.0.0.1:61000", "Coord server address.");
DEFINE_string(ps_addrs, "127.0.0.1:60000", "Param server addresses.");
DEFINE_int32(ps_id, 0, "Param server id(role is ps).");
DEFINE_int32(ps_thread_num, 1, "Number of threads(role is ps).");

DEFINE_bool(gnn_model, true, "true for GNN models, false for NonGNN models.");
DEFINE_bool(deep_model, false,
            "true for Deep models, false for NonDeep models.");
DEFINE_string(model, "unsup_graphsage", "Model name.");
DEFINE_string(model_config, "", "Model config.");
DEFINE_string(instance_reader, "unsup_graphsage",
              "Instance reader name(role is wk).");
DEFINE_string(instance_reader_config, "",
              "Instance reader config(role is wk).");
DEFINE_string(optimizer, "adam", "Optimizer name.");
DEFINE_string(optimizer_config, "", "Optimizer config.");
DEFINE_int32(epoch, 1, "Number of epochs.");
DEFINE_int32(batch, 32, "Batch size(sub_command is train, role is wk).");
DEFINE_string(in_model, "", "Input dir of model.");
DEFINE_string(warmup_model, "", "Warmup dir of model.");
DEFINE_string(in, "", "Input dir/file of training/testing data(role is ps).");
DEFINE_string(pretrain_path, "", "Input dir/file of pretrain param.");
DEFINE_string(item_feature, "", "Input dir/file of item feature.");
DEFINE_string(inst_file, "", "Input dir/file of instance file.");
DEFINE_string(freq_file, "", "Input dir/file of item frequency.");
DEFINE_bool(
    shuffle, true,
    "Shuffle input files for each epoch(sub_command is train, role is ps).");
DEFINE_bool(ts_enable, false, "Enable timestamp.");
DEFINE_uint64(ts_now, 0, "Timestamp of now.");
DEFINE_uint64(ts_expire_threshold, 0, "Timestamp expiration threshold.");
DEFINE_uint64(freq_filter_threshold, 0, "Frequency filter threshold.");
DEFINE_int32(verbose, 1, "Verbose level: 0-10(role is wk).");
DEFINE_int32(seed, 9527, "Seed of random engine(role is ps).");
DEFINE_int32(target_type, 2, "0 for loss, 1 for prob, 2 for emb.");
DEFINE_bool(out_model_remove_zeros, false, "Remove zeros from output model.");
DEFINE_string(out_model, "",
              "Output model dir(sub_command is train, role is ps).");
DEFINE_string(out_model_text, "",
              "Output model text dir(sub_command is train, role is ps).");
DEFINE_string(out_model_fkv, "",
              "Output model to dir in feature kv format(sub_command is train, "
              "role is ps).");
DEFINE_int32(out_model_fkv_pb_version, 2,
             "Set feature kv outputs's protocol version.");
DEFINE_string(
    out_predict, "",
    "Output predict dir(optional)(sub_command is predict, role is wk).");

namespace embedx {

int FLAGS_is_train = 0;
deepx_core::TcpEndpoint FLAGS_cs_endpoint;
std::vector<deepx_core::TcpEndpoint> FLAGS_ps_endpoints;
int FLAGS_ps_size = 0;
deepx_core::Shard FLAGS_shard;

void CheckGNNFlags() {
  if (FLAGS_is_train) {
    DXCHECK_THROW(FLAGS_target_type == 0);
  } else {
    // for GNN models
    // 1 : dump classification prob
    // 2 : dump node embedding
    DXCHECK_THROW(FLAGS_target_type == 1 || FLAGS_target_type == 2);
  }

  if (FLAGS_dist) {
    DXCHECK_THROW(!FLAGS_gs_addrs.empty());
  } else {
    DXCHECK_THROW(!FLAGS_node_graph.empty());
  }
  DXCHECK_THROW(
      FLAGS_negative_sampler_type == 0 || FLAGS_negative_sampler_type == 1 ||
      FLAGS_negative_sampler_type == 2 || FLAGS_negative_sampler_type == 3);
  DXCHECK_THROW(
      FLAGS_neighbor_sampler_type == 0 || FLAGS_neighbor_sampler_type == 1 ||
      FLAGS_neighbor_sampler_type == 2 || FLAGS_neighbor_sampler_type == 3);
  DXCHECK_THROW(FLAGS_gs_thread_num > 0);
}

void CheckNonGNNFlags() {
  if (FLAGS_is_train) {
    DXCHECK_THROW(FLAGS_target_type == 0);
  } else {
    // for NonGNN models
    // 1 : dump classification prob
    // 2 : dump user embedding
    // 3 : dump item embedding
    DXCHECK_THROW(FLAGS_target_type == 1 || FLAGS_target_type == 2 ||
                  FLAGS_target_type == 3);
  }
}

void CheckFlags() {
  deepx_core::AutoFileSystem fs;

  DXCHECK_THROW(FLAGS_sub_command == "train" || FLAGS_sub_command == "predict");
  DXCHECK_THROW(FLAGS_role == "ps" || FLAGS_role == "wk");
  DXCHECK_THROW(!FLAGS_cs_addr.empty());
  DXCHECK_THROW(!FLAGS_ps_addrs.empty());
  FLAGS_is_train = FLAGS_sub_command == "train" ? 1 : 0;
  FLAGS_cs_endpoint = deepx_core::MakeTcpEndpoint(FLAGS_cs_addr);
  FLAGS_ps_endpoints = deepx_core::MakeTcpEndpoints(FLAGS_ps_addrs);
  FLAGS_ps_size = (int)FLAGS_ps_endpoints.size();
  if (FLAGS_role == "ps") {
    DXCHECK_THROW(0 <= FLAGS_ps_id && FLAGS_ps_id < FLAGS_ps_size);

    deepx_core::CanonicalizePath(&FLAGS_in);
    DXCHECK_THROW(!FLAGS_in.empty());
    DXCHECK_THROW(fs.Open(FLAGS_in));
  }

  if (FLAGS_role == "wk") {
    if (FLAGS_gnn_model) {
      CheckGNNFlags();
    } else {
      CheckNonGNNFlags();
    }

    DXCHECK_THROW(!FLAGS_instance_reader.empty());
    DXCHECK_THROW(FLAGS_batch > 0);
  }

  DXCHECK_THROW(FLAGS_epoch > 0);
  if (FLAGS_sub_command == "predict") {
    FLAGS_epoch = 1;
    FLAGS_shuffle = false;
  }

  deepx_core::CanonicalizePath(&FLAGS_in_model);
  if (FLAGS_in_model.empty()) {
    DXCHECK_THROW(!FLAGS_model.empty());
    DXCHECK_THROW(!FLAGS_optimizer.empty());
  } else {
    DXINFO("--model will be ignored.");
    DXINFO("--model_config will be ignored.");
    DXINFO("--optimizer will be ignored.");
    DXINFO("--optimizer_config will be ignored.");
    DXCHECK_THROW(fs.Open(FLAGS_in_model));
    DXCHECK_THROW(!deepx_core::IsStdinStdoutPath(FLAGS_in_model));
  }

  deepx_core::CanonicalizePath(&FLAGS_warmup_model);
  if (!FLAGS_warmup_model.empty()) {
    DXCHECK_THROW(fs.Open(FLAGS_warmup_model));
    DXCHECK_THROW(!deepx_core::IsStdinStdoutPath(FLAGS_warmup_model));
  }

  if (FLAGS_sub_command == "train" && FLAGS_role == "ps") {
    deepx_core::CanonicalizePath(&FLAGS_out_model);
    DXCHECK_THROW(!FLAGS_out_model.empty());
    DXCHECK_THROW(fs.Open(FLAGS_out_model));
    DXCHECK_THROW(!deepx_core::IsStdinStdoutPath(FLAGS_out_model));
    (void)deepx_core::AutoFileSystem::MakeDir(FLAGS_out_model);

    deepx_core::CanonicalizePath(&FLAGS_out_model_text);
    if (!FLAGS_out_model_text.empty()) {
      DXCHECK_THROW(fs.Open(FLAGS_out_model_text));
      DXCHECK_THROW(!deepx_core::IsStdinStdoutPath(FLAGS_out_model_text));
      (void)deepx_core::AutoFileSystem::MakeDir(FLAGS_out_model_text);
    }
  }

  if (FLAGS_sub_command == "predict" && FLAGS_role == "wk") {
    deepx_core::CanonicalizePath(&FLAGS_out_predict);
    if (FLAGS_out_predict.empty()) {
      FLAGS_out_predict = FLAGS_in + ".predict";
      DXINFO("Didn't specify --out_predict, output to: %s.",
             FLAGS_out_predict.c_str());
    }
    (void)deepx_core::AutoFileSystem::MakeDir(FLAGS_out_predict);
  }

  DXCHECK_THROW(FLAGS_verbose >= 0);

  FLAGS_shard.InitShard(FLAGS_ps_size, MOD9973_NAME);

  if (FLAGS_is_train) {
    if (FLAGS_ts_enable) {
      DXCHECK_THROW(FLAGS_ts_now <=
                    (google::uint64)
                        std::numeric_limits<deepx_core::DataType::ts_t>::max());
      DXCHECK_THROW(FLAGS_ts_expire_threshold <=
                    (google::uint64)
                        std::numeric_limits<deepx_core::DataType::ts_t>::max());
    }

    DXCHECK_THROW(FLAGS_freq_filter_threshold <=
                  (google::uint64)
                      std::numeric_limits<deepx_core::DataType::freq_t>::max());
  }
}

}  // namespace embedx
