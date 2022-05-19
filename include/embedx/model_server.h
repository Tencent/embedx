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

#pragma once
#include <memory>  // std::unique_ptr
#include <string>
#include <utility>  // std::pair
#include <vector>

namespace deepx_core {

class Graph;
class Model;
class OpContext;

}  // namespace deepx_core

namespace embedx {

using deepx_core::Graph;
using deepx_core::Model;
using deepx_core::OpContext;

using feature_t = std::pair<uint64_t, float>;
using features_t = std::vector<feature_t>;
using embedding_t = std::vector<float>;

class ModelServer {
 private:
  std::unique_ptr<deepx_core::Graph> graph_;
  std::string target_name_;
  std::unique_ptr<deepx_core::Model> model_;
  // 1 for classification prob
  // 2 for user embedding
  // 3 for item embedding
  int target_type_ = 2;

 public:
  ModelServer();
  ~ModelServer();
  ModelServer(const ModelServer&) = delete;
  ModelServer& operator=(const ModelServer&) = delete;

 public:
  void set_target_type(int target_type) noexcept { target_type_ = target_type; }

 public:
  bool Load(const std::string& file);
  bool LoadGraph(const std::string& file);
  bool LoadModel(const std::string& file);

 public:
  bool Predict(const features_t& features, float* prob) const;
  bool Predict(const features_t& features, std::vector<float>* probs) const;
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  bool PredictUserEmbedding(const features_t& user_features,
                            embedding_t* embedding) const;
  bool BatchPredictUserEmbedding(
      const std::vector<features_t>& batch_user_features,
      std::vector<embedding_t>* embeddings) const;
  bool BatchGraphDeepFMPredict(const std::vector<features_t>& batch_features,
                               const std::vector<features_t>& batch_users,
                               std::vector<float>* batch_prob) const;

 public:
  std::unique_ptr<OpContext, void (*)(OpContext*)> NewOpContext() const;
  bool Predict(OpContext* op_context, const features_t& features,
               float* prob) const;
  bool Predict(OpContext* op_context, const features_t& features,
               std::vector<float>* probs) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  bool PredictUserEmbedding(OpContext* op_context,
                            const features_t& user_features,
                            embedding_t* embedding) const;
  bool BatchPredictUserEmbedding(
      OpContext* op_context, const std::vector<features_t>& batch_user_features,
      std::vector<embedding_t>* embeddings) const;
  bool BatchGraphDeepFMPredict(OpContext* op_context,
                               const std::vector<features_t>& batch_features,
                               const std::vector<features_t>& batch_users,
                               std::vector<float>* batch_prob) const;
};

}  // namespace embedx
