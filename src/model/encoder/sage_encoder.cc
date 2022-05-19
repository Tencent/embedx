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

#include "src/model/encoder/gnn_encoder.h"

namespace embedx {

GraphNode* SageEncoder(const std::string& prefix, GraphNode* hidden,
                       std::vector<GraphNode*> self_blocks,
                       std::vector<GraphNode*> neigh_blocks,
                       int sage_encoder_type, int depth, int sage_dim,
                       double relu_alpha) {
  GraphNode* next_hidden = hidden;
  for (int i = 0; i < depth; ++i) {
    auto* self_block = self_blocks[i];
    auto* neigh_block = neigh_blocks[i];
    if (sage_encoder_type == 0) {
      next_hidden = PinsageEncoder(
          "PinsageEncoder_" + prefix + std::to_string(i), next_hidden,
          self_block, neigh_block, sage_dim, relu_alpha);
    } else {
      next_hidden = DenseSageEncoder(
          "DenseSageEncoder_" + prefix + std::to_string(i), next_hidden,
          self_block, neigh_block, sage_dim, true, relu_alpha);
    }
  }
  return next_hidden;
}

}  // namespace embedx
