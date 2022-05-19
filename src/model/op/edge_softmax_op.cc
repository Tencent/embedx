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

bool EdgeSoftmaxInferShape(const Shape& W, Shape* Z) {
  if (!W.is_rank(2)) {
    DXERROR("Invalid W, rank of W: %d must be 2.", W.rank());
    return false;
  }

  Z->resize(W);
  return true;
}

// Edge softmax is an operation that computes softmax on node's neighbors.
// Support multi-head input logits.
// For node i and one head edge softmax is computed as below:
//     a_ij = exp(z_ij) / sum(exp(z_ik))
// k is neighobr of node i; z_ij is the logit of edge j -> i
//
// inputs:
//      X(CSR): SubGrph, a batch of nodes and its neighbors
//      W(Tensor): Shape(num_head, num_edge), unnormalized logits
//      eg.
//         X:
//            node0, neighbor1 neighbor2 neighbor3
//            node1, neighbor1 neighbor3 neighbor4
//            node2, neighbor1 neighbor5
//         W:     wi_jk indicate the importance of the k-th neighbor to the
//                j-th node in the i-th head
//            head0,   w0_01, w0_02, w0_03, w0_11, w0_13, w0_14, w0_21, w0_25
//            head1,   w1_01, w1_02, w1_03, w1_11, w1_13, w1_14, w1_21, w1_25
//
// output:
//      Z(Tensor): Shape(num_head, num_edge), normalized weights
template <typename T, typename I>
void EdgeSoftmax(const CSRMatrix<T, I>& X, const Tensor<T>& W, Tensor<T>* Z) {
  Z->zeros();
  int num_head = W.shape()[0];
  int num_edge = W.shape()[1];

  const auto* _W = W.data();
  auto* _Z = Z->data();

  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < X.row(); ++j) {
      int row_start = X.row_offset(j);
      int row_end = X.row_offset(j + 1);
      // num neighbors
      int n = row_end - row_start;

      const auto* _Wj = _W + row_start;
      auto* _Zj = _Z + row_start;

      deepx_core::LLMath<T>::softmax(n, _Wj, _Zj);
    }
    _W += num_edge;
    _Z += num_edge;
  }
}

template <typename T, typename I>
void EdgeSoftmaxBackward(const CSRMatrix<T, I>& X, const Tensor<T>& W,
                         const Tensor<T>& Z, const Tensor<T>& gZ,
                         Tensor<T>* gW) {
  int num_head = W.shape()[0];
  int num_edge = W.shape()[1];

  const auto* _gZ = gZ.data();
  const auto* _Z = Z.data();
  auto* _gW = gW->data();

  for (int i = 0; i < num_head; ++i) {
    for (int j = 0; j < X.row(); ++j) {
      int row_start = X.row_offset(j);
      int row_end = X.row_offset(j + 1);
      // num neighbors
      int n = row_end - row_start;

      const auto* _gZj = _gZ + row_start;
      const auto* _Zj = _Z + row_start;
      auto* _gWj = _gW + row_start;

      deepx_core::LLMath<T>::xypz(n, _gZj, _Zj, _gWj);
      deepx_core::LLMath<T>::axpy(n, -deepx_core::LLMath<T>::dot(n, _gZj, _Zj),
                                  _Zj, _gWj);
    }
    _gZ += num_edge;
    _Z += num_edge;
    _gW += num_edge;
  }
}

EdgeSoftmaxNode::EdgeSoftmaxNode(std::string name, GraphNode* X, GraphNode* W)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(X->node_type() == deepx_core::GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(X->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(W->tensor_type() == deepx_core::TENSOR_TYPE_TSR);

  input_ = {X, W};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;
}

class EdgeSoftmaxOp : public deepx_core::OpImpl {
 private:
  const GraphNode* Xnode_ = nullptr;
  const GraphNode* Wnode_ = nullptr;
  const csr_t* X_ = nullptr;
  const tsr_t* W_ = nullptr;
  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gW_ = nullptr;

 public:
  DEFINE_OP_LIKE(EdgeSoftmaxOp);

  void InitForward() override {
    Xnode_ = node_->input(0);
    DXCHECK_THROW(!Xnode_->need_grad());
    Wnode_ = node_->input(1);
    X_ = GetPtrCSR(Xnode_);
    W_ = GetPtrTSR(Wnode_);
    DXCHECK_THROW((int)X_->col_size() == W_->shape()[1]);
    DXCHECK_THROW(EdgeSoftmaxInferShape(W_->shape(), &Zshape_));
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);
    gW_ = InitGradTSR(Wnode_, W_->shape());
  }

  void Forward() override { EdgeSoftmax(*X_, *W_, Z_); }

  void Backward() override { EdgeSoftmaxBackward(*X_, *W_, *Z_, *gZ_, gW_); }
};

GRAPH_NODE_REGISTER(EdgeSoftmaxNode);
OP_REGISTER(EdgeSoftmaxOp, "EdgeSoftmaxNode");

}  // namespace embedx
