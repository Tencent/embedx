// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/op/gnn_graph_node.h"

namespace embedx {

bool AssembleInferShape(int Xrow, const Shape& Y, const Shape& W,
                        Shape* Z) noexcept {
  if (!Y.is_rank(2)) {
    DXERROR("Invalid Y, rank of Y: %d must be 2.", Y.rank());
    return false;
  }

  if (!W.is_rank(2)) {
    DXERROR("Invalid W, rank of W: %d must be 2.", W.rank());
    return false;
  }

  if (Xrow != Y[0]) {
    DXERROR("Invalid X, X row: %d must be %d.", Xrow, Y[0]);
    return false;
  }

  if (Y[1] != W[1]) {
    DXERROR("Invalid Y, col of Y: %d must be col of W: %d.", Y[1], W[1]);
    return false;
  }

  *Z = Y;
  return true;
}

AssembleNode::AssembleNode(std::string name, GraphNode* X, GraphNode* Y,
                           GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(Y->tensor_type() == deepx_core::TENSOR_TYPE_TSR);
  DXCHECK_THROW(W->tensor_type() == deepx_core::TENSOR_TYPE_SRM);
  input_ = {X, Y, W};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;

  if (!Y->shape().empty()) {
    (void)AssembleInferShape(X->shape()[0], Y->shape(), W->shape(), &shape_);
  }
}

template <typename T, typename I>
void AssembleBackward(const CSRMatrix<T, I>& X, const Tensor<T>& Y,
                      SparseRowMatrix<T, I>* W) noexcept {
  DXASSERT_RANK2(Y);
  DXASSERT(X.row() == (int)X.value_size());

  int row = Y.dim(0);
  int col = Y.dim(1);

  DXASSERT(W->col() == col);

  W->zeros();
  const auto* _Y = Y.data();

  for (int i = 0; i < row; ++i) {
    W->assign(X.col(i), _Y);
    _Y += col;
  }
}

class AssembleOp : public deepx_core::OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Ynode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  const csr_t* X_ = nullptr;
  const tsr_t* Y_ = nullptr;

  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gY_ = nullptr;

  const srm_t* Wsrm_ = nullptr;
  srm_t* Overwritten_Wsrm_ = nullptr;

 public:
  DEFINE_OP_LIKE(AssembleOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    Ynode_ = node_->input(1);
    Wnode_ = node_->input(2);
    X_ = GetPtrCSR(Xnode_);
    Y_ = GetPtrTSR(Ynode_);
    Wsrm_ = GetPtrSRM(Wnode_);
    DXCHECK_THROW(
        AssembleInferShape(X_->row(), Y_->shape(), Wsrm_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gY_ = InitGradTSR(Ynode_, Y_->shape());
    Overwritten_Wsrm_ = InitOverwrittenParamSRM(Wnode_, Wsrm_->col());
  }

  void Forward() override { Z_->set_data(*Y_); }

  void Backward() override {
    if (gY_) {
      gY_->set_data(*gZ_);
    }
    AssembleBackward(*X_, *Y_, Overwritten_Wsrm_);
  }

  void GetPullRequest(
      deepx_core::PullRequest* /*pull_request*/) const override {}
};

GRAPH_NODE_REGISTER(AssembleNode);
OP_REGISTER(AssembleOp, "AssembleNode");

}  // namespace embedx
