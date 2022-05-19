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

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/model.h>
#include <deepx_core/graph/op_context.h>
#include <deepx_core/instance/base.h>
#include <embedx/model_server.h>

#include "src/model/instance_node_name.h"

namespace embedx {

using namespace deepx_core;  // NOLINT

using float_t = InstanceReader::float_t;
using int_t = InstanceReader::int_t;
using tsr_t = InstanceReader::tsr_t;
using csr_t = InstanceReader::csr_t;

static void EmplaceRow(const features_t& features, csr_t* X) {
  static constexpr float MAX_FEATURE_VALUE =
      InstanceReaderHelper<float, uint64_t>::MAX_FEATURE_VALUE;
  for (const auto& entry : features) {
    if (-MAX_FEATURE_VALUE <= entry.second &&
        entry.second <= MAX_FEATURE_VALUE) {
      X->emplace((int_t)entry.first, (float_t)entry.second);
    }
  }
  X->add_row();
}

ModelServer::ModelServer() {}

ModelServer::~ModelServer() {}

bool ModelServer::Load(const std::string& file) {
  AutoInputFileStream is;
  if (!is.Open(file)) {
    DXERROR("Failed to open: %s.", file.c_str());
    return false;
  }

  DXINFO("Loading graph from: %s...", file.c_str());
  graph_.reset(new Graph);
  if (!graph_->Read(is)) {
    return false;
  }
  DXCHECK(graph_->target_size() > target_type_);
  target_name_ = graph_->target(target_type_).name();
  DXINFO("Done.");

  DXINFO("Loading model from: %s...", file.c_str());
  model_.reset(new Model);
  model_->Init(graph_.get());
  if (!model_->Read(is)) {
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool ModelServer::LoadGraph(const std::string& file) {
  graph_.reset(new Graph);
  if (!graph_->Load(file)) {
    return false;
  }
  DXCHECK(graph_->target_size() > target_type_);
  target_name_ = graph_->target(target_type_).name();
  return true;
}

bool ModelServer::LoadModel(const std::string& file) {
  model_.reset(new Model);
  model_->Init(graph_.get());
  return model_->Load(file);
}

bool ModelServer::Predict(const features_t& features, float* prob) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  *prob = (float)P.data(0);
  return true;
}

bool ModelServer::Predict(const features_t& features,
                          std::vector<float>* probs) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  const float_t* _P = P.data();
  probs->resize(col);
  for (int j = 0; j < col; ++j) {
    (*probs)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(const std::vector<features_t>& batch_features,
                               std::vector<float>* batch_prob) const {
  if (batch_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(
    const std::vector<features_t>& batch_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  batch_probs->resize(X.row());
  const float_t* _P = P.data();
  batch_probs->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::PredictUserEmbedding(const features_t& user_features,
                                       embedding_t* embedding) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(instance_name::X_USER_FEATURE_NAME);
  EmplaceRow(user_features, &X);
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& hidden = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(hidden.is_rank(2));
  int col = hidden.dim(1);
  DXASSERT(hidden.same_shape(X.row(), col));
  const float_t* _P = hidden.data();
  embedding->resize(col);
  for (int j = 0; j < col; ++j) {
    (*embedding)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredictUserEmbedding(
    const std::vector<features_t>& batch_user_features,
    std::vector<embedding_t>* embeddings) const {
  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(instance_name::X_USER_FEATURE_NAME);
  for (const auto& features : batch_user_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& hidden = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(hidden.is_rank(2));
  int col = hidden.dim(1);
  DXASSERT(hidden.same_shape(X.row(), col));
  embeddings->resize(X.row());
  const float_t* _P = hidden.data();
  embeddings->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& embedding = (*embeddings)[i];
    embedding.resize(col);
    for (int j = 0; j < col; ++j) {
      embedding[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::BatchGraphDeepFMPredict(
    const std::vector<features_t>& batch_features,
    const std::vector<features_t>& batch_users,
    std::vector<float>* batch_prob) const {
  DXCHECK(target_type_ == 1);
  if (batch_features.empty()) {
    return false;
  }

  if (!graph_ || !model_) {
    return false;
  }

  OpContext op_context;
  op_context.Init(graph_.get(), model_->mutable_param());
  if (!op_context.InitOp({target_name_}, -1)) {
    return false;
  }

  Instance* inst = op_context.mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }

  auto& User = inst->insert<csr_t>(instance_name::X_USER_NODE_NAME);
  for (const auto& users : batch_users) {
    EmplaceRow(users, &User);
  }
  DXASSERT(X.row() == User.row());

  inst->set_batch(X.row());

  op_context.InitPredict();
  op_context.Predict();
  const auto& P = op_context.hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

static void DeleteOpContext(OpContext* op_context) noexcept {
  delete op_context;
}

std::unique_ptr<OpContext, void (*)(OpContext*)> ModelServer::NewOpContext()
    const {
  std::unique_ptr<OpContext, void (*)(OpContext*)> op_context(new OpContext,
                                                              DeleteOpContext);

  if (!graph_ || !model_) {
    op_context.reset();
    return op_context;
  }

  op_context->Init(graph_.get(), model_->mutable_param());
  if (!op_context->InitOp({target_name_}, -1)) {
    op_context.reset();
    return op_context;
  }

  return op_context;
}

bool ModelServer::Predict(OpContext* op_context, const features_t& features,
                          float* prob) const {
  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(deepx_core::X_NAME);
  X.clear();
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  *prob = (float)P.data(0);
  return true;
}

bool ModelServer::Predict(OpContext* op_context, const features_t& features,
                          std::vector<float>* probs) const {
  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(deepx_core::X_NAME);
  X.clear();
  EmplaceRow(features, &X);
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  const float_t* _P = P.data();
  probs->resize(col);
  for (int j = 0; j < col; ++j) {
    (*probs)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(OpContext* op_context,
                               const std::vector<features_t>& batch_features,
                               std::vector<float>* batch_prob) const {
  if (batch_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(deepx_core::X_NAME);
  X.clear();
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredict(
    OpContext* op_context, const std::vector<features_t>& batch_features,
    std::vector<std::vector<float>>* batch_probs) const {
  if (batch_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  int prev_batch = inst->batch();

  auto& X = inst->get_or_insert<csr_t>(deepx_core::X_NAME);
  X.clear();
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  if (prev_batch != inst->batch()) {
    op_context->InitPredict();
  }

  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  int col = P.dim(1);
  DXASSERT(P.same_shape(X.row(), col));
  batch_probs->resize(X.row());
  const float_t* _P = P.data();
  batch_probs->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& batch_prob = (*batch_probs)[i];
    batch_prob.resize(col);
    for (int j = 0; j < col; ++j) {
      batch_prob[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::PredictUserEmbedding(OpContext* op_context,
                                       const features_t& user_features,
                                       embedding_t* embedding) const {
  Instance* inst = op_context->mutable_inst();
  auto& X = inst->insert<csr_t>(instance_name::X_USER_FEATURE_NAME);
  EmplaceRow(user_features, &X);
  inst->set_batch(X.row());

  op_context->InitPredict();
  op_context->Predict();
  const auto& hidden = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(hidden.is_rank(2));
  int col = hidden.dim(1);
  DXASSERT(hidden.same_shape(X.row(), col));
  const float_t* _P = hidden.data();
  embedding->resize(col);
  for (int j = 0; j < col; ++j) {
    (*embedding)[j] = (float)*_P;
    ++_P;
  }
  return true;
}

bool ModelServer::BatchPredictUserEmbedding(
    OpContext* op_context, const std::vector<features_t>& batch_user_features,
    std::vector<embedding_t>* embeddings) const {
  Instance* inst = op_context->mutable_inst();
  auto& X = inst->insert<csr_t>(instance_name::X_USER_FEATURE_NAME);
  for (const auto& features : batch_user_features) {
    EmplaceRow(features, &X);
  }
  inst->set_batch(X.row());

  op_context->InitPredict();
  op_context->Predict();
  const auto& hidden = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(hidden.is_rank(2));
  int col = hidden.dim(1);
  DXASSERT(hidden.same_shape(X.row(), col));
  embeddings->resize(X.row());
  const float_t* _P = hidden.data();
  embeddings->resize(X.row());
  for (int i = 0; i < X.row(); ++i) {
    auto& embedding = (*embeddings)[i];
    embedding.resize(col);
    for (int j = 0; j < col; ++j) {
      embedding[j] = (float)*_P;
      ++_P;
    }
  }
  return true;
}

bool ModelServer::BatchGraphDeepFMPredict(
    OpContext* op_context, const std::vector<features_t>& batch_features,
    const std::vector<features_t>& batch_users,
    std::vector<float>* batch_prob) const {
  DXCHECK(target_type_ == 1);
  if (batch_features.empty()) {
    return false;
  }

  Instance* inst = op_context->mutable_inst();
  auto& X = inst->insert<csr_t>(deepx_core::X_NAME);
  for (const auto& features : batch_features) {
    EmplaceRow(features, &X);
  }

  auto& User = inst->insert<csr_t>(instance_name::X_USER_NODE_NAME);
  for (const auto& users : batch_users) {
    EmplaceRow(users, &User);
  }
  DXASSERT(X.row() == User.row());

  inst->set_batch(X.row());

  op_context->InitPredict();
  op_context->Predict();
  const auto& P = op_context->hidden().get<tsr_t>(target_name_);
  DXASSERT(P.is_rank(2));
  DXASSERT(P.same_shape(X.row(), 1));
  batch_prob->resize(X.row());
  const float_t* _P = P.data();
  for (int i = 0; i < X.row(); ++i) {
    (*batch_prob)[i] = (float)*_P;
    ++_P;
  }
  return true;
}

}  // namespace embedx
