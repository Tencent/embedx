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

#include "src/model/op/gnn_graph_node.h"

namespace embedx {

bool HiddenLookupInferShape(int Xrow, const Shape& W, Shape* Z) noexcept {
  if (!W.is_rank(2)) {
    DXERROR("Invalid W, rank of W: %d must be 2.", W.rank());
    return false;
  }
  Z->resize(Xrow, W[1]);
  return true;
}

template <typename T, typename I>
void HiddenLookup(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                  Tensor<T>* Z) noexcept {
  deepx_core::LLSparseTensor<T, I>::gesmm_mod(X, W, 0, Z);
}

template <typename T, typename I>
void HiddenLookupBackward(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                          const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                          Tensor<T>* gW) noexcept {
  int k = W.dim(0);
  int n = gZ.dim(1);
  DXASSERT(gZ.same_shape(X.row(), n));
  const auto* _gZ = gZ.data();
  auto* _gW = gW->data();

  CSR_FOR_EACH_ROW(X, i) {
    CSR_FOR_EACH_COL(X, i) {
      auto* gWj = _gW + (CSR_COL(X) % k) * n;
      deepx_core::LLMath<T>::axpy(n, CSR_VALUE(X), _gZ, gWj);
    }
    _gZ += n;
  }
}

HiddenLookupNode::HiddenLookupNode(std::string name, GraphNode* X, GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == deepx_core::GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(W->tensor_type() == deepx_core::TENSOR_TYPE_TSR);
  input_ = {X, W};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;

  if (X->shape().is_rank(2) && !W->shape().empty()) {
    (void)HiddenLookupInferShape(X->shape()[0], W->shape(), &shape_);
  }
}

class HiddenLookupOp : public deepx_core::OpImpl {
 private:
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
  DEFINE_OP_LIKE(HiddenLookupOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    W_node_type_ = Wnode_->node_type();
    W_tensor_type_ = Wnode_->tensor_type();
    X_ = GetPtrCSR(Xnode_);
    Wtsr_ = GetPtrTSR(Wnode_);
    DXCHECK_THROW(HiddenLookupInferShape(X_->row(), Wtsr_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gW_ = InitGradTSR(Wnode_, Wtsr_->shape());
  }

  void Forward() override { HiddenLookup(*X_, *Wtsr_, Z_); }

  void Backward() override {
    HiddenLookupBackward(*X_, *Wtsr_, *Z_, *gZ_, gW_);
  }
};

GRAPH_NODE_REGISTER(HiddenLookupNode);
OP_REGISTER(HiddenLookupOp, "HiddenLookupNode");

}  // namespace embedx
