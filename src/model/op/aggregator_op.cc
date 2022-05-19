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

#include "src/common/data_types.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

bool AggregatorInferShape(int Xrow, const Shape& W, Shape* Z) noexcept {
  if (!W.is_rank(2)) {
    DXERROR("Invalid W, rank of W: %d must be 2.", W.rank());
    return false;
  }
  Z->resize(Xrow, W[1]);
  return true;
}

template <typename T, typename I>
void MeanAggregate(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                   Tensor<T>* Z) noexcept {
  DXASSERT_RANK2(W);
  int col = W.dim(1);
  DXASSERT(Z->same_shape(X.row(), col));
  const auto* _W = W.data();
  auto* _Z = Z->data();

  Z->zeros();

  T weight_sum = 0;
  CSR_FOR_EACH_ROW(X, i) {
    weight_sum = 0;
    CSR_FOR_EACH_COL(X, i) { weight_sum += CSR_VALUE(X); }
    CSR_FOR_EACH_COL(X, i) {
      DXASSERT(CSR_COL(X) < (int_t)W.dim(0));
      const auto* Wj = _W + CSR_COL(X) * col;
      deepx_core::LLMath<T>::axpy(col, CSR_VALUE(X) / weight_sum, Wj, _Z);
    }
    _Z += col;
  }
}

template <typename T, typename I>
void MeanAggregateBackward(const CSRMatrix<T, I>& X, const Tensor<T>& /*W*/,
                           const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                           Tensor<T>* gW) noexcept {
  int col = gW->dim(1);
  DXASSERT(gZ.same_shape(X.row(), col));
  const auto* _gZ = gZ.data();
  auto* _gW = gW->data();

  T weight_sum = 0;
  CSR_FOR_EACH_ROW(X, i) {
    weight_sum = 0;
    CSR_FOR_EACH_COL(X, i) { weight_sum += CSR_VALUE(X); }
    CSR_FOR_EACH_COL(X, i) {
      DXASSERT(CSR_COL(X) < (int_t)gW->dim(0));
      auto* gWj = _gW + CSR_COL(X) * col;
      deepx_core::LLMath<T>::axpy(col, CSR_VALUE(X) / weight_sum, _gZ, gWj);
    }
    _gZ += col;
  }
}

template <typename T, typename I>
void SumAggregate(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                  Tensor<T>* Z) noexcept {
  DXASSERT_RANK2(W);
  int col = W.dim(1);
  DXASSERT(Z->same_shape(X.row(), col));
  const auto* _W = W.data();
  auto* _Z = Z->data();

  Z->zeros();

  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      DXASSERT(CSR_COL(X) < (int_t)W.dim(0));
      const auto* Wj = _W + CSR_COL(X) * col;
      deepx_core::LLMath<T>::axpy(col, CSR_VALUE(X), Wj, _Z);
    }
    _Z += col;
  }
}

template <typename T, typename I>
void SumAggregateBackward(const CSRMatrix<T, I>& X, const Tensor<T>& /*W*/,
                          const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                          Tensor<T>* gW) noexcept {
  int col = gW->dim(1);
  DXASSERT(gZ.same_shape(X.row(), col));
  const auto* _gZ = gZ.data();
  auto* _gW = gW->data();

  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      DXASSERT(CSR_COL(X) < (int_t)gW->dim(0));
      auto* gWj = _gW + CSR_COL(X) * col;
      deepx_core::LLMath<T>::axpy(col, CSR_VALUE(X), _gZ, gWj);
    }
    _gZ += col;
  }
}

AggregatorNodeBase::AggregatorNodeBase(std::string name, GraphNode* X,
                                       GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == deepx_core::GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(W->tensor_type() == deepx_core::TENSOR_TYPE_TSR);
  input_ = {X, W};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && !W->shape().empty()) {
    (void)AggregatorInferShape(X->shape()[0], W->shape(), &shape_);
  }
}

class AggregatorOpBase : public deepx_core::OpImpl {
 protected:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  int W_node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  int W_tensor_type_ = deepx_core::TENSOR_TYPE_TSR;
  const csr_t* X_ = nullptr;
  const tsr_t* Wtsr_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gW_ = nullptr;

 public:
  void InitForward() override {
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    W_node_type_ = Wnode_->node_type();
    W_tensor_type_ = Wnode_->tensor_type();
    X_ = GetPtrCSR(Xnode_);
    Wtsr_ = GetPtrTSR(Wnode_);
    DXCHECK_THROW(AggregatorInferShape(X_->row(), Wtsr_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gW_ = InitGradTSR(Wnode_, Wtsr_->shape());
  }
};

MeanAggregatorNode::MeanAggregatorNode(std::string name, GraphNode* X,
                                       GraphNode* W)
    : AggregatorNodeBase(std::move(name), X, W) {}

class MeanAggregatorOp : public AggregatorOpBase {
 public:
  DEFINE_OP_LIKE(MeanAggregatorOp);
  void Forward() override { MeanAggregate(*X_, *Wtsr_, Z_); }

  void Backward() override {
    MeanAggregateBackward(*X_, *Wtsr_, *Z_, *gZ_, gW_);
  }
};

GRAPH_NODE_REGISTER(MeanAggregatorNode);
OP_REGISTER(MeanAggregatorOp, "MeanAggregatorNode");

SumAggregatorNode::SumAggregatorNode(std::string name, GraphNode* X,
                                     GraphNode* W)
    : AggregatorNodeBase(std::move(name), X, W) {}

class SumAggregatorOp : public AggregatorOpBase {
 public:
  DEFINE_OP_LIKE(SumAggregatorOp);
  void Forward() override { SumAggregate(*X_, *Wtsr_, Z_); }

  void Backward() override {
    SumAggregateBackward(*X_, *Wtsr_, *Z_, *gZ_, gW_);
  }
};

GRAPH_NODE_REGISTER(SumAggregatorNode);
OP_REGISTER(SumAggregatorOp, "SumAggregatorNode");

}  // namespace embedx
