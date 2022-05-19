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
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

GraphNode* PinsageEncoder(const std::string& prefix, GraphNode* hidden,
                          GraphNode* self_block, GraphNode* neigh_block,
                          int dim, double alpha) {
  // self
  auto* self_embed = HiddenLookup("", self_block, hidden);

  // neigh
  auto* hidden_fc =
      deepx_core::FullyConnect(prefix + "_hidden_fc", hidden, dim);
  auto* hidden_fc_act = deepx_core::LeakyRelu("", hidden_fc, alpha);
  auto* neigh_embed = MeanAggregator("", neigh_block, hidden_fc_act);

  auto* concat = deepx_core::Concat("", {self_embed, neigh_embed});
  auto* concat_fc =
      deepx_core::FullyConnect(prefix + "_concat_fc", concat, dim);
  auto* sage_embed = deepx_core::LeakyRelu("", concat_fc, alpha);
  return sage_embed;
}

GraphNode* PinsageRootEncoder(const std::string& prefix, GraphNode* hidden,
                              int dim, double alpha) {
  auto* hidden_fc =
      deepx_core::FullyConnect(prefix + "_hidden_fc", hidden, dim);
  auto* hidden_fc_act = deepx_core::LeakyRelu("", hidden_fc, alpha);
  auto* sage_embed =
      deepx_core::FullyConnect(prefix + "_sage_embed", hidden_fc_act, dim);
  return sage_embed;
}

}  // namespace embedx
