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

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/array_view.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/ps/coord_server.h>
#include <deepx_core/ps/param_server.h>

#include <memory>  // std::unque_ptr
#include <string>

#include "src/model/model_zoo.h"
#include "src/tools/dist/dist_flags.h"
#include "src/tools/model_util.h"

namespace embedx {
namespace {

/************************************************************************/
/* RankCoordServer */
/************************************************************************/
class RankCoordServer : public deepx_core::CoordServer {};

/************************************************************************/
/* RankParamServer */
/************************************************************************/
class RankParamServer : public deepx_core::ParamServer {
 private:
  struct SessionData {
    deepx_core::PullRequest pull_request;
    deepx_core::TensorMap param;
    deepx_core::TensorMap grad;
    deepx_core::TensorMap overwritten_param;
  };

 private:
  deepx_core::Graph graph_;
  deepx_core::ModelShard model_shard_;
  std::unique_ptr<ModelUtil> model_util_;

 public:
  bool Init();

 protected:
  void OnAccept(conn_t conn) override;
  void OnPullRequest(conn_t conn) override;
  void OnPushNotify(conn_t conn) override;
  void OnModelSaveRequest(conn_t conn) override;
  void OnTerminationNotify(conn_t conn) override;
};

bool RankParamServer::Init() {
  if (FLAGS_is_train) {
    if (FLAGS_in_model.empty()) {
      std::unique_ptr<ModelZoo> model_zoo(NewModelZoo(FLAGS_model));
      DXCHECK_THROW(model_zoo);
      deepx_core::StringMap config;
      DXCHECK_THROW(deepx_core::ParseConfig(FLAGS_model_config, &config));
      DXCHECK_THROW(model_zoo->InitConfig(config));
      DXCHECK_THROW(model_zoo->InitGraph(&graph_));
    } else {
      DXCHECK_THROW(deepx_core::LoadGraph(FLAGS_in_model, &graph_));
    }

    model_util_.reset(new ModelUtil(&graph_));

    model_shard_.seed(FLAGS_seed + FLAGS_ps_id * 10099);  // magic number
    model_shard_.InitShard(&FLAGS_shard, FLAGS_ps_id);
    model_shard_.InitGraph(&graph_);
    if (FLAGS_in_model.empty()) {
      DXCHECK_THROW(model_shard_.InitModel());
      if (!FLAGS_pretrain_path.empty()) {
        DXCHECK_THROW(model_util_->LoadPretrainParam(
            FLAGS_pretrain_path, FLAGS_shard, &model_shard_));
      }
      DXCHECK_THROW(
          model_shard_.InitOptimizer(FLAGS_optimizer, FLAGS_optimizer_config));

      if (FLAGS_ts_enable) {
        DXCHECK_THROW(model_shard_.InitTSStore(
            (deepx_core::DataType::ts_t)FLAGS_ts_now,
            (deepx_core::DataType::ts_t)FLAGS_ts_expire_threshold));
      }

      if (FLAGS_freq_filter_threshold > 0) {
        DXCHECK_THROW(model_shard_.InitFreqStore(
            (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold));
      }
    } else {
      DXCHECK_THROW(model_shard_.LoadModel(FLAGS_in_model));
      if (FLAGS_is_train) {
        DXCHECK_THROW(
            model_shard_.LoadOptimizer(FLAGS_in_model, FLAGS_optimizer_config));

        if (FLAGS_ts_enable) {
          if (!model_shard_.LoadTSStore(FLAGS_in_model, FLAGS_ts_now,
                                        FLAGS_ts_expire_threshold)) {
            DXCHECK_THROW(model_shard_.InitTSStore(
                (deepx_core::DataType::ts_t)FLAGS_ts_now,
                (deepx_core::DataType::ts_t)FLAGS_ts_expire_threshold));
          }
        }

        if (FLAGS_freq_filter_threshold > 0) {
          if (!model_shard_.LoadFreqStore(
                  FLAGS_in_model,
                  (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold)) {
            DXCHECK_THROW(model_shard_.InitFreqStore(
                (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold));
          }
        }
      }
    }

  } else {
    DXCHECK_THROW(deepx_core::LoadGraph(FLAGS_in_model, &graph_));
    model_shard_.InitShard(&FLAGS_shard, FLAGS_ps_id);
    model_shard_.InitGraph(&graph_);
    DXCHECK_THROW(model_shard_.LoadModel(FLAGS_in_model));
  }

  DXCHECK_THROW(model_shard_.model().HasSRM());

  if (FLAGS_is_train && FLAGS_ps_thread_num > 1) {
    DXCHECK_THROW(model_shard_.InitLock());
  }
  return true;
}

void RankParamServer::OnAccept(conn_t conn) {
  conn->mutable_user_data()->emplace(SessionData());
  ParamServer::OnAccept(conn);
}

void RankParamServer::OnPullRequest(conn_t conn) {
  auto& session_data = conn->mutable_user_data()->unsafe_to_ref<SessionData>();

  {
    const deepx_core::const_string_view& buf =
        conn->in_message().pull_request().buf;
    deepx_core::InputStringStream is;
    is.SetView(buf.data(), buf.size());
    is >> session_data.pull_request;
    DXCHECK_THROW(is);
  }

  model_shard_.Pull(&session_data.pull_request, &session_data.param);

  {
    std::string& buf =
        conn->mutable_out_message()->mutable_pull_response()->buf;
    deepx_core::OutputStringStream os;
    buf.clear();
    os.SetView(&buf);
    os << session_data.param;
    DXCHECK_THROW(os);
  }
}

void RankParamServer::OnPushNotify(conn_t conn) {
  auto& session_data = conn->mutable_user_data()->unsafe_to_ref<SessionData>();

  const deepx_core::const_string_view& buf =
      conn->in_message().push_notify().buf;
  deepx_core::InputStringStream is;
  is.SetView(buf.data(), buf.size());
  // view, zero-copy
  ReadView(is, session_data.grad);
  ReadView(is, session_data.overwritten_param);
  DXCHECK_THROW(is);

  model_shard_.Push(&session_data.grad, &session_data.overwritten_param);
}

void RankParamServer::OnModelSaveRequest(conn_t /*conn*/) {
  if (FLAGS_ps_id == 0) {
    DXCHECK_THROW(deepx_core::SaveGraph(FLAGS_out_model, graph_));
    DXCHECK_THROW(deepx_core::SaveShard(FLAGS_out_model, FLAGS_shard));
  }

  if (FLAGS_out_model_remove_zeros) {
    model_shard_.mutable_model()->RemoveZerosSRM();
  }

  if (FLAGS_ts_enable && FLAGS_ts_expire_threshold > 0) {
    model_shard_.ExpireTSStore();
  }

  DXCHECK_THROW(model_shard_.SaveModel(FLAGS_out_model));
  DXCHECK_THROW(model_shard_.SaveOptimizer(FLAGS_out_model));

  if (FLAGS_ts_enable) {
    DXCHECK_THROW(model_shard_.SaveTSStore(FLAGS_out_model));
  }

  if (FLAGS_freq_filter_threshold > 0) {
    DXCHECK_THROW(model_shard_.SaveFreqStore(FLAGS_out_model));
  }

  if (!FLAGS_out_model_text.empty()) {
    DXCHECK_THROW(model_shard_.SaveTextModel(FLAGS_out_model_text));
  }

  if (!FLAGS_out_model_fkv.empty()) {
    DXCHECK_THROW(model_shard_.SaveFeatureKVModel(
        FLAGS_out_model_fkv, FLAGS_out_model_fkv_pb_version));
    DXCHECK_THROW(model_shard_.SaveSuccess(FLAGS_out_model_fkv));
  }
}

void RankParamServer::OnTerminationNotify(conn_t /*conn*/) {}

}  // namespace

void RunCoordServer() {
  deepx_core::CoordServerConfig config;
  config.listen_endpoint = FLAGS_cs_endpoint;
  config.ps_endpoints = FLAGS_ps_endpoints;
  config.epoch = FLAGS_epoch;
  DXCHECK_THROW(deepx_core::AutoFileSystem::ListRecursive(
      FLAGS_in, true, &config.file_dispatcher_files));
  DXCHECK_THROW(!config.file_dispatcher_files.empty());
  DXINFO("Got %d files.", (int)config.file_dispatcher_files.size());
  for (const std::string& file : config.file_dispatcher_files) {
    DXINFO("  %s", file.c_str());
  }
  config.file_dispatcher_shuffle = FLAGS_shuffle ? 1 : 0;
  config.file_dispatcher_timeout = 0;
  if (FLAGS_is_train) {
    config.dump_model = 1;
  } else {
    config.dump_model = 0;
  }

  RankCoordServer server;
  server.set_config(config);
  server.Run();
}

void RunParamServer() {
  deepx_core::TcpServerConfig config;
  config.listen_endpoint = FLAGS_ps_endpoints[FLAGS_ps_id];
  config.thread = FLAGS_ps_thread_num;

  RankParamServer server;
  server.set_config(config);
  DXCHECK(server.Init());
  server.Run();
}

}  // namespace embedx
