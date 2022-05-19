// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"

namespace embedx {

std::vector<GraphNode*> BinaryClassificationTarget(const std::string& prefix,
                                                   GraphNode* X, GraphNode* Y,
                                                   int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(X->shape()[1] == 1);
  DXCHECK_THROW(Y->shape()[1] == 1);
  auto* L = deepx_core::SigmoidBCELoss(prefix + "L", X, Y);
  auto* P = deepx_core::Sigmoid(prefix + "P", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul(prefix + "WL", L, W);
    auto* WM = deepx_core::ReduceMean(prefix + "WM", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean(prefix + "M", L);
    return {M, P};
  }
}

std::vector<GraphNode*> BinaryClassificationTarget(GraphNode* X, GraphNode* Y,
                                                   int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(X->shape()[1] == 1);
  DXCHECK_THROW(Y->shape()[1] == 1);
  auto* L = deepx_core::SigmoidBCELoss("", X, Y);
  auto* P = deepx_core::Sigmoid("", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul("", L, W);
    auto* WM = deepx_core::ReduceMean("", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean("", L);
    return {M, P};
  }
}

std::vector<GraphNode*> MultiClassificationTarget(const std::string& prefix,
                                                  GraphNode* X, int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* Y = deepx_core::GetY(1);
  auto* L = deepx_core::BatchSoftmaxCELoss(prefix + "L", X, Y);
  auto* P = deepx_core::Softmax(prefix + "P", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul(prefix + "WL", L, W);
    auto* WM = deepx_core::ReduceMean(prefix + "WM", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean(prefix + "M", L);
    return {M, P};
  }
}

std::vector<GraphNode*> MultiClassificationTarget(GraphNode* X, int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* Y = deepx_core::GetY(1);
  auto* L = deepx_core::BatchSoftmaxCELoss("", X, Y);
  auto* P = deepx_core::Softmax("", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul("", L, W);
    auto* WM = deepx_core::ReduceMean("", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean("", L);
    return {M, P};
  }
}

std::vector<GraphNode*> MultiLabelClassificationTarget(
    const std::string& prefix, GraphNode* X, int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* X1 = deepx_core::Reshape2("X1", X, Shape(-1, 1));
  auto* Y = deepx_core::GetY(1);
  auto* Y1 = deepx_core::Reshape2("Y1", Y, Shape(-1, 1));
  auto* L = deepx_core::SigmoidBCELoss(prefix + "L", X1, Y1);
  auto* P = deepx_core::Sigmoid(prefix + "P", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul(prefix + "WL", L, W);
    auto* WM = deepx_core::ReduceMean(prefix + "WM", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean(prefix + "M", L);
    return {M, P};
  }
}

std::vector<GraphNode*> MultiLabelClassificationTarget(GraphNode* X,
                                                       int has_w) {
  DXCHECK_THROW(X->shape().is_rank(2));
  auto* X1 = deepx_core::Reshape2("", X, Shape(-1, 1));
  auto* Y = deepx_core::GetY(1);
  auto* Y1 = deepx_core::Reshape2("", Y, Shape(-1, 1));
  auto* L = deepx_core::SigmoidBCELoss("", X1, Y1);
  auto* P = deepx_core::Sigmoid("", X);
  if (has_w) {
    auto* W = deepx_core::GetW(1);
    auto* WL = deepx_core::Mul("", L, W);
    auto* WM = deepx_core::ReduceMean("", WL);
    return {WM, P};
  } else {
    auto* M = deepx_core::ReduceMean("", L);
    return {M, P};
  }
}

}  // namespace embedx
