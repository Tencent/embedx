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
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/graph/optimizer.h>
#include <deepx_core/tensor/data_type.h>

#include <atomic>
#include <memory>  // std::unique_ptr

#include "src/model/embed_instance_reader.h"

namespace embedx {

/************************************************************************/
/* TrainerContext */
/************************************************************************/
class TrainerContext : public deepx_core::DataType {
 protected:
  freq_t freq_filter_threshold_ = 0;
  int verbose_ = 0;
  std::string target_name_;
  int target_type_ = 0;
  std::function<std::unique_ptr<EmbedInstanceReader>()>
      instance_reader_creator_;

  deepx_core::ModelShard* local_model_shard_ = nullptr;
  std::unique_ptr<deepx_core::OpContext> op_context_;
  std::unique_ptr<EmbedInstanceReader> instance_reader_;
  int op_context_batch_ = -1;
  double file_loss_ = 0;
  double file_loss_weight_ = 0;
  double thread_loss_ = 0;
  double tread_loss_weight_ = 0;

 public:
  void set_freq_filter_threshold(freq_t freq_filter_threshold) noexcept {
    freq_filter_threshold_ = freq_filter_threshold;
  }
  void set_verbose(int verbose) noexcept { verbose_ = verbose; }
  void set_target_name(const std::string& target_name) noexcept {
    target_name_ = target_name;
  }
  void set_target_type(int target_type) noexcept { target_type_ = target_type; }

  void set_instance_reader_creator(
      const std::function<std::unique_ptr<EmbedInstanceReader>()>&
          instance_reader_creator) noexcept {
    instance_reader_creator_ = instance_reader_creator;
  }

  double file_loss() const noexcept { return file_loss_; }
  double file_loss_weight() const noexcept { return file_loss_weight_; }

 protected:
  void DoInit(deepx_core::ModelShard* local_model_shard);

 public:
  virtual ~TrainerContext() = default;
  virtual void TrainBatch() = 0;
  virtual void TrainFile(int thread_id, const std::string& file);
  virtual void PredictBatch() = 0;
  virtual void DumpBatch(deepx_core::OutputStream& os) const;  // NOLINT
  virtual void PredictFile(int thread_id, const std::string& in_file,
                           const std::string& out_file);

 protected:
  int enable_profile_ = 0;
  struct Profile {
    int processed_inst = 0;
    double get_batch = 0;
    double init_op_context = 0;
    double forward = 0;
    double backward = 0;
    double pull = 0;
    double push = 0;

    void clear() {
      processed_inst = 0;
      get_batch = 0;
      init_op_context = 0;
      forward = 0;
      backward = 0;
      pull = 0;
      push = 0;
    }
  };
  Profile profile_;
  virtual void DumpProfile() const noexcept;
};

/************************************************************************/
/* TrainerContextNonShard */
/************************************************************************/
class TrainerContextNonShard : public TrainerContext {
 protected:
  deepx_core::Optimizer* optimizer_ = nullptr;

 public:
  bool Init(deepx_core::ModelShard* model_shard);
  void TrainBatch() override;
  void PredictBatch() override;
};

/************************************************************************/
/* TrainerContextShard */
/************************************************************************/
class TrainerContextShard : public TrainerContext {
 protected:
  int shard_size_ = 0;
  std::vector<deepx_core::ModelShard*> model_shards_;
  std::vector<deepx_core::Model*> models_;
  std::vector<deepx_core::Optimizer*> optimizers_;

  deepx_core::PullRequest pull_request_;
  std::vector<deepx_core::PullRequest> pull_requests_;
  int pull_request_active_ = 0;
  std::vector<int> pull_request_masks_;

  std::vector<std::unique_ptr<deepx_core::TensorMap>> params_;
  std::vector<std::unique_ptr<deepx_core::TensorMap>> grads_;
  std::vector<std::unique_ptr<deepx_core::TensorMap>> overwritten_params_;
  std::vector<deepx_core::PullRequest::id_set_t*> aux1_;
  std::vector<srm_t*> aux2_;

  deepx_core::ThreadPool::wait_token_t wait_token_;

 public:
  bool Init(std::vector<deepx_core::ModelShard>* model_shards,
            deepx_core::ModelShard* local_model_shard);
  void TrainBatch() override;
  void PredictBatch() override;

 protected:
  void CompletionHandler();
  void WaitForCompletion();
  void Pull(int is_train);
  void Push();
};

}  // namespace embedx
