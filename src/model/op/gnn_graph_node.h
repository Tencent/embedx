// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#pragma once
#include <deepx_core/common/class_factory.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/op_impl.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/csr_matrix.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor.h>
#include <deepx_core/tensor/tensor_type.h>

#include <string>

namespace embedx {

using ::deepx_core::CSRMatrix;
using ::deepx_core::GraphNode;
using ::deepx_core::InstanceNode;
using ::deepx_core::Op;
using ::deepx_core::Shape;
using ::deepx_core::SparseRowMatrix;
using ::deepx_core::Tensor;
using ::deepx_core::VariableNode;

class HiddenLookupNode : public GraphNode {
 public:
  HiddenLookupNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(HiddenLookupNode);
};

class AggregatorNodeBase : public GraphNode {
 public:
  AggregatorNodeBase(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(AggregatorNodeBase);
};

class MeanAggregatorNode : public AggregatorNodeBase {
 public:
  MeanAggregatorNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(MeanAggregatorNode);
};

class SumAggregatorNode : public AggregatorNodeBase {
 public:
  SumAggregatorNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(SumAggregatorNode);
};

class BatchLookupDotNode : public GraphNode {
 public:
  BatchLookupDotNode(std::string name, GraphNode* Xin, GraphNode* Xout,
                     GraphNode* Win, GraphNode* Wout);
  DEFINE_GRAPH_NODE_LIKE(BatchLookupDotNode);
};

class WeightedAverageNode : public GraphNode {
 public:
  WeightedAverageNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(WeightedAverageNode);
};

// EdgeSoftmax is an operation that computes softmax on node's neighbors.
class EdgeSoftmaxNode : public GraphNode {
 public:
  EdgeSoftmaxNode(std::string name, GraphNode* X, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(EdgeSoftmaxNode);
};

// Assemble is an operation that assemble X(as key) and Y (as value) to
// update W.
// if key not in W:
//      W.emplace(key, value)
// else:
//      W[key] = value
//
// inputs:
//      X(CSR): Shape(row,    ), node ids
//      Y(TSR): Shape(row, dim), hidden embeddings
//      W(SRM): Shape(   , dim), cache embeddings
// output:
//      Z(TSR): Shape(row, dim), hidden embeddings copied from Y.
//
// eg.
//      inputs:
//         X:
//            node0,
//            node1,
//         Y:
//            y_01, y_02, y_03, ... , y_0dim
//            y_11, y_12, y_13, ... , y_1dim
//         W(not updated):
//            node0 : w_01, w_02, w_03, ... , w_0dim
//            node2 : w_22, w_22, w_23, ... , w_2dim
//      outputs:
//         Z:
//            y_01, y_02, y_03, ... , y_0dim
//            y_11, y_12, y_13, ... , y_1dim
//         W(updated):
//            node0 : y_01, y_02, y_03, ... , y_0dim
//            node1 : y_11, y_12, y_13, ... , y_1dim
//            node2 : w_22, w_22, w_23, ... , w_2dim

class AssembleNode : public GraphNode {
 public:
  AssembleNode(std::string name, GraphNode* X, GraphNode* Y, GraphNode* W);
  DEFINE_GRAPH_NODE_LIKE(AssembleNode);
};

DEFINE_GRAPH_NODE_CREATOR(HiddenLookup)
DEFINE_GRAPH_NODE_CREATOR(WeightedAverage)
DEFINE_GRAPH_NODE_CREATOR(BatchLookupDot)
DEFINE_GRAPH_NODE_CREATOR(MeanAggregator)
DEFINE_GRAPH_NODE_CREATOR(SumAggregator)
DEFINE_GRAPH_NODE_CREATOR(EdgeSoftmax)
DEFINE_GRAPH_NODE_CREATOR(Assemble)

}  // namespace embedx
