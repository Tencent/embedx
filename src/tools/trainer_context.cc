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

#include "src/tools/trainer_context.h"

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/common/profile_util.h>
#include <deepx_core/graph/freq_store.h>

#include <chrono>
#include <cstdlib>
#include <sstream>

#include "src/model/instance_node_name.h"

namespace embedx {

/************************************************************************/
/* TrainerContext */
/************************************************************************/
void TrainerContext::DoInit(deepx_core::ModelShard* local_model_shard) {
  local_model_shard_ = local_model_shard;
  op_context_.reset(new deepx_core::OpContext);
  op_context_->Init(&local_model_shard_->graph(),
                    local_model_shard_->mutable_param());
  instance_reader_ = instance_reader_creator_();
  op_context_batch_ = -1;
  file_loss_ = 0;
  file_loss_weight_ = 0;

  char* enable_profile = std::getenv("EMBEDX_ENABLE_PROFILE");
  if (enable_profile) {
    enable_profile_ = std::stoi(enable_profile);
  }
}

void TrainerContext::TrainFile(int thread_id, const std::string& file) {
  DXCHECK_THROW(op_context_->InitOp({target_name_}, 0));
  op_context_->mutable_inst()->clear();
  op_context_batch_ = -1;
  file_loss_ = 0;
  file_loss_weight_ = 0;

  DXCHECK_THROW(instance_reader_->Open(file));

  size_t processed_batch = 0;
  size_t processed_inst = 0;
  size_t verbose_batch = deepx_core::GetVerboseBatch(verbose_);
  auto begin = std::chrono::steady_clock::now();
  if (enable_profile_) {
    profile_.clear();
  }

  auto dump_speed = [this, thread_id, &processed_inst, &begin]() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    DXINFO("[%d] %f instances/s, file_loss=%f", thread_id,
           processed_inst * 1000.0 / ms.count(),
           file_loss_ / file_loss_weight_);
    if (enable_profile_) {
      DumpProfile();
    }
  };

  Instance* inst = op_context_->mutable_inst();
  for (;;) {
    bool success;
    if (!enable_profile_) {
      success = instance_reader_->GetBatch(inst);
    } else {
      deepx_core::NanosecondTimerGuard _(profile_.get_batch);
      success = instance_reader_->GetBatch(inst);
    }
    if (!success) {
      break;
    }

    TrainBatch();
    processed_batch += 1;
    processed_inst += inst->batch();
    profile_.processed_inst += inst->batch();
    if (verbose_ && processed_batch % verbose_batch == 0) {
      dump_speed();
    }
  }

  if (inst->batch() > 0) {
    TrainBatch();
    processed_inst += inst->batch();
  }

  if (verbose_) {
    dump_speed();
  }
}

void TrainerContext::DumpBatch(deepx_core::OutputStream& os) const {
  // output format
  // FLAGS_target_type=1
  //     ctr: label prob
  //     classification: node prob0 prob1
  // FLAGS_target_type=2 or FLAGS_target_type=3
  //     embedding: node val0 val1 val2 val3

  const vec_int_t* nodes = nullptr;
  const tsr_t* Y = nullptr;
  const auto* Z = op_context_->ptr().get<tsr_t*>(target_name_);
  const Instance& inst = op_context_->inst();

  int inst_batch = Z->dim(0);
  DXCHECK_THROW(Z->is_rank(2));

  if (target_type_ == 1) {
    auto it = inst.find(deepx_core::Y_NAME);
    if (it != inst.end()) {
      Y = &it->second.to_ref<tsr_t>();
      DXCHECK_THROW(Y->is_rank(2));
      DXCHECK_THROW(Y->dim(0) == inst_batch);
    }

    it = inst.find(instance_name::X_PREDICT_NODE_NAME);
    if (it != inst.end()) {
      nodes = &it->second.to_ref<vec_int_t>();
      DXCHECK_THROW((int)nodes->size() == inst_batch);
    }
  } else if (target_type_ == 2 or target_type_ == 3) {
    auto it = inst.find(instance_name::X_PREDICT_NODE_NAME);
    DXCHECK_THROW(it != inst.end());

    nodes = &it->second.to_ref<vec_int_t>();
    DXCHECK_THROW((int)nodes->size() == inst_batch);
  }

  std::ostringstream oss;
  for (int i = 0; i < inst_batch; ++i) {
    oss.clear();
    oss.str("");
    if (nodes) {
      oss << (*nodes)[i];
    }
    if (Y) {
      for (int j = 0; j < Y->dim(1); ++j) {
        oss << ' ' << Y->data(i * Y->dim(1) + j);
      }
    }
    for (int j = 0; j < Z->dim(1); ++j) {
      oss << ' ' << Z->data(i * Z->dim(1) + j);
    }
    oss << "\n";
    std::string s = oss.str();
    os.Write(s.data(), s.size());
  }
}

void TrainerContext::PredictFile(int thread_id, const std::string& in_file,
                                 const std::string& out_file) {
  DXINFO("target_name: %s", target_name_.c_str());
  DXCHECK_THROW(op_context_->InitOp({target_name_}, -1));
  op_context_->mutable_inst()->clear();
  op_context_batch_ = -1;

  DXCHECK_THROW(instance_reader_->Open(in_file));

  deepx_core::AutoOutputFileStream os;
  DXINFO("out file: %s", out_file.c_str());
  DXCHECK_THROW(os.Open(out_file));

  size_t processed_batch = 0;
  size_t processed_inst = 0;
  size_t verbose_batch = deepx_core::GetVerboseBatch(verbose_);
  auto begin = std::chrono::steady_clock::now();

  auto dump_speed = [thread_id, &processed_inst, &begin]() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    DXINFO("[%d] %f instances/s", thread_id,
           processed_inst * 1000.0 / ms.count());
  };

  Instance* inst = op_context_->mutable_inst();
  while (instance_reader_->GetBatch(inst)) {
    PredictBatch();
    DumpBatch(os);
    processed_batch += 1;
    processed_inst += inst->batch();
    if (verbose_ && processed_batch % verbose_batch == 0) {
      dump_speed();
    }
  }

  if (inst->batch() > 0) {
    PredictBatch();
    DumpBatch(os);
    processed_inst += inst->batch();
  }

  if (verbose_) {
    dump_speed();
  }
}

void TrainerContext::DumpProfile() const noexcept {
  std::ostringstream os;
  auto dump_member = [&os](const std::string& name, double avg_duration) {
    char buf[256];
    double avg_in_us = avg_duration / 1000;  // 1000: ns -> us
    std::snprintf(buf, sizeof(buf), "%s=%.4fus ", name.c_str(), avg_in_us);
    os << buf;
  };
  double processed_inst = profile_.processed_inst;
  dump_member("GetBatch", profile_.get_batch / processed_inst);
  dump_member("InitOpContext", profile_.init_op_context / processed_inst);
  dump_member("Forward", profile_.forward / processed_inst);
  dump_member("Backward", profile_.backward / processed_inst);
  dump_member("Pull", profile_.pull / processed_inst);
  dump_member("Push", profile_.push / processed_inst);

  DXINFO("%s", os.str().c_str());
}

/************************************************************************/
/* TrainerContextNonShard */
/************************************************************************/
bool TrainerContextNonShard::Init(deepx_core::ModelShard* model_shard) {
  DoInit(model_shard);
  optimizer_ = local_model_shard_->mutable_optimizer();
  return true;
}

void TrainerContextNonShard::TrainBatch() {
  op_context_->InitForward();
  op_context_->InitBackward();

  op_context_->Forward();
  op_context_->Backward();

  optimizer_->Update(op_context_->mutable_grad());
  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextNonShard::PredictBatch() {
  op_context_->InitPredict();
  op_context_->Predict();
}

/************************************************************************/
/* TrainerContextShard */
/************************************************************************/
bool TrainerContextShard::Init(
    std::vector<deepx_core::ModelShard>* model_shards,
    deepx_core::ModelShard* local_model_shard) {
  DoInit(local_model_shard);

  shard_size_ = (int)model_shards->size();
  model_shards_.resize(shard_size_);
  models_.resize(shard_size_);
  optimizers_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i] = &(*model_shards)[i];
    models_[i] = model_shards_[i]->mutable_model();
    optimizers_[i] = model_shards_[i]->mutable_optimizer();
  }

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

void TrainerContextShard::TrainBatch() {
  op_context_->InitForward();
  op_context_->InitBackward();

  Pull(1);
  op_context_->Forward();
  op_context_->Backward();
  Push();
  file_loss_ += op_context_->loss();
  file_loss_weight_ += 1;
}

void TrainerContextShard::PredictBatch() {
  op_context_->InitPredict();
  Pull(0);
  op_context_->Predict();
}

void TrainerContextShard::CompletionHandler() {
  std::unique_lock<std::mutex> guard(wait_token_.mutex);
  if (--wait_token_.remain == 0) {
    wait_token_.cond.notify_all();
  }
}

void TrainerContextShard::WaitForCompletion() {
  std::unique_lock<std::mutex> guard(wait_token_.mutex);
  while (wait_token_.remain > 0) {
    wait_token_.cond.wait(guard);
  }
}

void TrainerContextShard::Pull(int is_train) {
  op_context_->GetPullRequest(&pull_request_);
  if (freq_filter_threshold_ > 0 && is_train) {
    deepx_core::FreqStore::GetIdFreqMap(op_context_->inst(),
                                        &pull_request_.id_freq_map);
  }
  pull_request_.is_train = is_train;
  local_model_shard_->SplitPullRequest(pull_request_, &pull_requests_, &aux1_);

  pull_request_active_ = 0;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_requests_[i].empty()) {
      pull_request_masks_[i] = 0;
    } else {
      pull_request_masks_[i] = 1;
      ++pull_request_active_;
    }
  }

  wait_token_.remain = pull_request_active_;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      model_shards_[i]->AsyncPull(&pull_requests_[i], params_[i].get(),
                                  [this]() { CompletionHandler(); });
    } else {
      params_[i]->clear();
    }
  }
  WaitForCompletion();

  local_model_shard_->mutable_model()->SetParam(&params_);
}

void TrainerContextShard::Push() {
  local_model_shard_->SplitGrad(local_model_shard_->param(),
                                op_context_->mutable_grad(), &grads_, &aux2_);
  local_model_shard_->SplitParam(op_context_->overwritten_param(),
                                 &overwritten_params_, &aux2_);

  wait_token_.remain = pull_request_active_;
  for (int i = 0; i < shard_size_; ++i) {
    if (pull_request_masks_[i]) {
      model_shards_[i]->AsyncPush(grads_[i].get(), overwritten_params_[i].get(),
                                  [this]() { CompletionHandler(); });
    }
  }
  WaitForCompletion();
}

}  // namespace embedx
