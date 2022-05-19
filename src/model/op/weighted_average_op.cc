// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include <cmath>  // std::exp

#include "src/model/op/gnn_graph_node.h"

namespace embedx {

bool WeightedAverageInferShape(const Shape& X, const Shape& W,
                               Shape* Z) noexcept {
  if (!X.is_rank(2)) {
    DXERROR("Invalid X, rank of X: %d must be 2.", X.rank());
    return false;
  }

  if (!W.is_rank(2)) {
    DXERROR("Invalid W, rank of W: %d must be 2.", W.rank());
    return false;
  }
  if (W.dim(1) == 0) {
    return false;
  }

  Z->resize(X[0], X[1] / W[1]);
  return true;
}

template <typename T>
void WeightedAverage(const Tensor<T>& X, const Tensor<T>& W, Tensor<T>* Z,
                     Tensor<T>* aux) noexcept {
  DXASSERT_RANK2(X);
  DXASSERT_RANK2(W);
  DXASSERT_RANK2(*Z);
  DXASSERT(Z->same_shape(X.dim(0), X.dim(1) / W.dim(1)));
  DXASSERT(X.dim(0) == W.dim(0));
  int row = X.dim(0);

  // exp
  aux->resize(row, 1);
  aux->zeros();
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < W.dim(1); ++j) {
      aux->data(i) += std::exp(W.data(i * W.dim(1) + j));
    }
  }

  Z->zeros();
  for (int i = 0; i < row; ++i) {
    auto* Zi = Z->data() + i * Z->dim(1);
    for (int j = 0; j < W.dim(1); ++j) {
      DXASSERT(aux->data(i) != 0);
      auto Wij = std::exp(W.data(i * W.dim(1) + j)) / aux->data(i);
      const auto* Xij = X.data() + i * X.dim(1) + j * Z->dim(1);
      deepx_core::LLMath<T>::axpy(Z->dim(1), Wij, Xij, Zi);
    }
  }
}

template <typename T>
void WeightedAverageBackward(const Tensor<T>& X, const Tensor<T>& W,
                             const Tensor<T>& Z, const Tensor<T>& gZ,
                             Tensor<T>* gX, Tensor<T>* gW, Tensor<T>* aux1,
                             Tensor<T>* aux2, Tensor<T>* aux3,
                             Tensor<T>* aux4) noexcept {
  DXASSERT_RANK2(X);
  DXASSERT_RANK2(W);
  DXASSERT_RANK2(Z);
  DXASSERT_RANK2(gZ);
  DXASSERT_RANK2(*gX);
  DXASSERT_RANK2(*gW);
  int row = X.dim(0);

  // exp
  aux1->resize(row, 1);
  aux1->zeros();
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < W.dim(1); ++j) {
      aux1->data(i) += std::exp(W.data(i * W.dim(1) + j));
    }
  }

  aux2->resize(1, Z.dim(1));
  aux3->resize(1, Z.dim(1));
  aux4->resize(1, Z.dim(1));

  for (int i = 0; i < row; ++i) {
    const auto* Zi = Z.data() + i * Z.dim(1);
    const auto* gZi = gZ.data() + i * Z.dim(1);
    for (int j = 0; j < W.dim(1); ++j) {
      DXASSERT(aux1->data(i) != 0);
      auto Wij = std::exp(W.data(i * W.dim(1) + j)) / aux1->data(i);
      // gX
      auto* gXij = gX->data() + i * X.dim(1) + j * Z.dim(1);
      deepx_core::LLMath<T>::axpy(Z.dim(1), Wij, gZi, gXij);
      // gW
      const auto* Xij = X.data() + i * X.dim(1) + j * Z.dim(1);
      deepx_core::LLMath<T>::mul_scalar(Z.dim(1), Xij, Wij, aux2->data());
      deepx_core::LLMath<T>::mul_scalar(Z.dim(1), Zi, Wij, aux3->data());
      deepx_core::LLMath<T>::sub(Z.dim(1), aux2->data(), aux3->data(),
                                 aux4->data());
      auto* gWij = gW->data() + i * W.dim(1) + j;
      *gWij += deepx_core::LLMath<T>::dot(Z.dim(1), aux4->data(), gZi);
    }
  }
}

WeightedAverageNode::WeightedAverageNode(std::string name, GraphNode* X,
                                         GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == deepx_core::TENSOR_TYPE_TSR);
  DXCHECK_THROW(W->tensor_type() == deepx_core::TENSOR_TYPE_TSR);
  input_ = {X, W};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;
  if (X->shape().is_rank(2) && W->shape().is_rank(2)) {
    (void)WeightedAverageInferShape(X->shape(), W->shape(), &shape_);
  }
}

class WeightedAverageOp : public deepx_core::OpImpl {
 private:
  const tsr_t* X_ = nullptr;
  const tsr_t* W_ = nullptr;
  Shape ZShape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gX_ = nullptr;
  tsr_t* gW_ = nullptr;

  tsr_t aux1_;
  tsr_t aux2_;
  tsr_t aux3_;
  tsr_t aux4_;

 public:
  DEFINE_OP_LIKE(WeightedAverageOp);

  void InitForward() override {
    X_ = GetPtrTSR(node_->input(0));
    W_ = GetPtrTSR(node_->input(1));
    DXCHECK_THROW(
        WeightedAverageInferShape(X_->shape(), W_->shape(), &ZShape_));
    Z_ = InitHiddenTSR(node_, ZShape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gX_ = InitGradTSR(node_->input(0), X_->shape());
    gW_ = InitGradTSR(node_->input(1), W_->shape());
  }

  void Forward() override { WeightedAverage(*X_, *W_, Z_, &aux1_); }

  void Backward() override {
    WeightedAverageBackward(*X_, *W_, *Z_, *gZ_, gX_, gW_, &aux1_, &aux2_,
                            &aux3_, &aux4_);
  }
};

GRAPH_NODE_REGISTER(WeightedAverageNode);
OP_REGISTER(WeightedAverageOp, "WeightedAverageNode");

}  // namespace embedx
