// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/array_view.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/common/profile_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/ps/tcp_connection.h>

#include <chrono>
#include <memory>  // std::unique_ptr
#include <string>
#include <thread>
#include <vector>

#include "src/deep/client/deep_client.h"
#include "src/deep/deep_config.h"
#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/model_zoo.h"
#include "src/tools/dist/dist_flags.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/trainer_context.h"

namespace embedx {
namespace {

/************************************************************************/
/* TrainerContextDist */
/************************************************************************/
class TrainerContextDist : public TrainerContext {
 private:
  deepx_core::IoContext io_;
  deepx_core::TcpConnections ps_conns_;
  deepx_core::InputStringStream is_;
  deepx_core::OutputStringStream os_;
  int shard_size_ = 0;
  deepx_core::PullRequest pull_request_;
  std::vector<deepx_core::PullRequest> pull_requests_;
  std::vector<int> pull_request_masks_;
  std::vector<std::unique_ptr<deepx_core::TensorMap>> params_;
  std::vector<std::unique_ptr<deepx_core::TensorMap>> grads_;
  std::vector<std::unique_ptr<deepx_core::TensorMap>> overwritten_params_;
  std::vector<deepx_core::PullRequest::id_set_t*> aux1_;
  std::vector<srm_t*> aux2_;

 public:
  TrainerContextDist();
  bool Init(deepx_core::ModelShard* local_model_shard);
  void TrainBatch() override;
  void PredictBatch() override;

 private:
  void Pull();
  void Push();
};

TrainerContextDist::TrainerContextDist() : io_(), ps_conns_(&io_) {}

bool TrainerContextDist::Init(deepx_core::ModelShard* local_model_shard) {
  DoInit(local_model_shard);

  DXCHECK_THROW(ps_conns_.ConnectRetry(FLAGS_ps_endpoints) == 0);

  shard_size_ = FLAGS_shard.shard_size();
  pull_requests_.resize(shard_size_);
  pull_request_masks_.resize(shard_size_);
  params_.resize(shard_size_);
  grads_.resize(shard_size_);
  overwritten_params_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    params_[i].reset(new deepx_core::TensorMap);
    grads_[i].reset(new deepx_core::TensorMap);
    overwritten_params_[i].reset(new deepx_core::TensorMap);
  }
  aux1_.resize(shard_size_);
  aux2_.resize(shard_size_);
  return true;
}

void TrainerContextDist::TrainBatch() {
  if (!enable_profile_) {
    op_context_->InitForward();
    op_context_->InitBackward();

    Pull();
    op_context_->Forward();
    op_context_->Backward();
    Push();
  } else {
    {
      deepx_core::NanosecondTimerGuard _(profile_.init_op_context);
      op_context_->InitForward();
      op_context_->InitBackward();
    }
    {
      deepx_core::NanosecondTimerGuard _(profile_.pull);
      Pull();
    }
    {
      deepx_core::NanosecondTimerGuard _(profile_.forward);
      op_context_->Forward();
    }
    {
      deepx_core::NanosecondTimerGuard _(profile_.backward);
      op_context_->Backward();
    }
    {
      deepx_core::NanosecondTimerGuard _(profile_.push);
      Push();
    }
  }
  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextDist::PredictBatch() {
  op_context_->InitPredict();

  Pull();
  op_context_->Predict();
}

void TrainerContextDist::Pull() {
  op_context_->GetPullRequest(&pull_request_);
  pull_request_.is_train = FLAGS_is_train;
  local_model_shard_->SplitPullRequest(pull_request_, &pull_requests_, &aux1_);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_requests_[i].empty()) {
      pull_request_masks_[i] = 0;
    } else {
      pull_request_masks_[i] = 1;
    }
  }

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      std::string& buf =
          ps_conns_[i]->mutable_out_message()->mutable_pull_request()->buf;
      buf.clear();
      os_.SetView(&buf);
      os_ << pull_requests_[i];
      DXCHECK_THROW(os_);
    }
  }

  DXCHECK_THROW(ps_conns_.RpcPullRequest(&pull_request_masks_) == 0);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      const deepx_core::const_string_view& buf =
          ps_conns_[i]->in_message().pull_response().buf;
      is_.SetView(buf.data(), buf.size());
      // view, zero-copy
      ReadView(is_, *params_[i]);
      DXCHECK_THROW(is_);
    } else {
      params_[i]->clear();
    }
  }

  local_model_shard_->mutable_model()->SetParam(&params_);
}

void TrainerContextDist::Push() {
  local_model_shard_->SplitGrad(local_model_shard_->param(),
                                op_context_->mutable_grad(), &grads_, &aux2_);
  local_model_shard_->SplitParam(op_context_->overwritten_param(),
                                 &overwritten_params_, &aux2_);

  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      std::string& buf =
          ps_conns_[i]->mutable_out_message()->mutable_push_notify()->buf;
      buf.clear();
      os_.SetView(&buf);
      os_ << *grads_[i] << *overwritten_params_[i];
      DXCHECK_THROW(os_);
    }
  }

  DXCHECK_THROW(ps_conns_.RpcPushNotify(&pull_request_masks_) == 0);
}

/************************************************************************/
/* TrainerDist */
/************************************************************************/
class TrainerDist : public deepx_core::DataType {
 private:
  deepx_core::IoContext io_;
  deepx_core::TcpConnection cs_conn_;
  deepx_core::Graph graph_;
  deepx_core::ModelShard local_model_shard_;
  TrainerContextDist context_;

 private:
  std::unique_ptr<GraphClient> graph_client_;
  std::unique_ptr<DeepClient> deep_client_;

 public:
  TrainerDist();
  bool Init();
  void Train();
  void Predict();
};

TrainerDist::TrainerDist() : io_(), cs_conn_(&io_) {}

bool TrainerDist::Init() {
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

  local_model_shard_.seed(0);
  local_model_shard_.InitShard(&FLAGS_shard, 0);
  local_model_shard_.InitGraph(&graph_);
  DXCHECK_THROW(local_model_shard_.InitModelPlaceholder());

  std::string target_name = graph_.target(FLAGS_target_type).name();

  if (FLAGS_gnn_model) {
    GraphConfig graph_config;
    graph_config.set_ip_ports(FLAGS_gs_addrs);

    graph_client_ = NewGraphClient(graph_config, GraphClientEnum::DIST);
    if (!graph_client_) {
      return false;
    }
  }

  if (FLAGS_deep_model) {
    if (FLAGS_is_train &&
        (!FLAGS_item_feature.empty() || !FLAGS_inst_file.empty() ||
         !FLAGS_freq_file.empty())) {
      DeepConfig deep_config;

      if (!FLAGS_item_feature.empty()) {
        deep_config.set_item_feature(FLAGS_item_feature);
      }
      if (!FLAGS_inst_file.empty()) {
        deep_config.set_inst_file(FLAGS_inst_file);
      }
      if (!FLAGS_freq_file.empty()) {
        deep_config.set_freq_file(FLAGS_freq_file);
      }

      if (!FLAGS_node_config.empty()) {
        deep_config.set_node_config(FLAGS_node_config);
      }

      if (FLAGS_negative_sampler_type != 0 &&
          FLAGS_negative_sampler_type != 1 &&
          FLAGS_negative_sampler_type != 2 &&
          FLAGS_negative_sampler_type != 3) {
        DXERROR(
            "Negative sampler type only support : '0(UNIFORM) || 1(ALIAS) || "
            "(2)WORD2VEC || 3(PARTIAL_SUM)'.");
        return false;
      }
      deep_config.set_negative_sampler_type(FLAGS_negative_sampler_type);
      deep_client_ = NewDeepClient(deep_config, DeepClientEnum::LOCAL);
      DXCHECK(deep_client_ != nullptr);
    }
  }

  auto instance_reader_creator = [this]() {
    std::unique_ptr<EmbedInstanceReader> instance_reader(
        NewEmbedInstanceReader(FLAGS_instance_reader));
    DXCHECK(instance_reader);
    deepx_core::StringMap config;
    DXCHECK(deepx_core::ParseConfig(FLAGS_instance_reader_config, &config));
    config["batch"] = std::to_string(FLAGS_batch);

    DXCHECK(instance_reader->InitConfig(config));

    if (graph_client_) {
      DXCHECK(instance_reader->InitGraphClient(graph_client_.get()));
    }

    if (deep_client_) {
      DXCHECK(instance_reader->InitDeepClient(deep_client_.get()));
    }

    return instance_reader;
  };

  context_.set_verbose(FLAGS_verbose);
  context_.set_target_name(target_name);
  context_.set_target_type(FLAGS_target_type);
  context_.set_instance_reader_creator(instance_reader_creator);
  return context_.Init(&local_model_shard_);
}

void TrainerDist::Train() {
  int epoch = 0;
  std::string file;
  for (;;) {
    DXINFO("Epoch %d begins.", epoch + 1);
    DXCHECK_THROW(cs_conn_.ConnectRetry(FLAGS_cs_endpoint) == 0);
    for (;;) {
      if (cs_conn_.RpcFileRequest() == 0) {
        epoch = cs_conn_.in_message().file_response().epoch;
        file = cs_conn_.in_message().file_response().file;
        if (file.empty()) {
          DXINFO("Worker got no new file.");
          std::this_thread::sleep_for(std::chrono::seconds(5));  // magic number
          continue;
        } else {
          DXINFO("Worker has got file: %s.", file.c_str());
          context_.TrainFile(0, file);
          auto* file_finished_notify =
              cs_conn_.mutable_out_message()->mutable_file_finish_notify();
          file_finished_notify->file = file;
          file_finished_notify->loss = context_.file_loss();
          file_finished_notify->loss_weight = context_.file_loss_weight();
          DXCHECK_THROW(cs_conn_.RpcFileFinishNotify() == 0);
        }
      } else {
        DXINFO("Failed to RpcFileRequest.");
        break;
      }
    }

    cs_conn_.Close();
    DXINFO("Epoch %d completed.", epoch + 1);
    if (epoch == FLAGS_epoch - 1) {
      break;
    }
  }
}

void TrainerDist::Predict() {
  std::string file;
  DXCHECK_THROW(cs_conn_.ConnectRetry(FLAGS_cs_endpoint) == 0);
  for (;;) {
    if (cs_conn_.RpcFileRequest() == 0) {
      file = cs_conn_.in_message().file_response().file;
      if (file.empty()) {
        DXINFO("Worker got no new file.");
        std::this_thread::sleep_for(std::chrono::seconds(5));  // magic number
        continue;
      } else {
        DXINFO("Worker has got file: %s.", file.c_str());
        context_.PredictFile(
            0, file, deepx_core::GetOutputPredictFile(FLAGS_out_predict, file));
        auto* file_finished_notify =
            cs_conn_.mutable_out_message()->mutable_file_finish_notify();
        file_finished_notify->file = file;
        file_finished_notify->loss = 0;
        file_finished_notify->loss_weight = 0;
        DXCHECK_THROW(cs_conn_.RpcFileFinishNotify() == 0);
      }
    } else {
      DXINFO("Failed to RpcFileRequest.");
      break;
    }
  }

  cs_conn_.Close();
}

}  // namespace

void RunWorker() {
  TrainerDist trainer;
  DXCHECK(trainer.Init());
  if (FLAGS_is_train) {
    trainer.Train();
  } else {
    trainer.Predict();
  }
}

}  // namespace embedx
