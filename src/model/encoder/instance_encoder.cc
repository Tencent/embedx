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

#include "src/model/encoder/gnn_encoder.h"

namespace embedx {

GraphNode* GetXInput(const std::string& name) {
  // Shape (BATCH_PLACEHOLDER = 100, ) cannot reshape to (n,3) when compiling
  // the graph. Change X shape from (BATCH_PLACEHOLDER, ) to (-1, ).
  return new InstanceNode(name, Shape(-1, 0), TENSOR_TYPE_CSR);
}

std::vector<GraphNode*> GetXBlockInputs(const std::string& name, int depth) {
  std::vector<GraphNode*> inst_nodes;
  for (int i = 0; i < depth; ++i) {
    auto* inst_node = new InstanceNode(
        name + std::to_string(i), Shape(BATCH_PLACEHOLDER, 0), TENSOR_TYPE_CSR);
    inst_nodes.emplace_back(inst_node);
  }
  return inst_nodes;
}

// Y_UNSUPVISED_NAME
GraphNode* GetYUnsup(const std::string& name, int label_size) {
  return new InstanceNode(name, Shape(BATCH_PLACEHOLDER, label_size),
                          TENSOR_TYPE_TSR);
}

}  // namespace embedx
