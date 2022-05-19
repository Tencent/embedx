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
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

GraphNode* EgesEncoder(const std::string& prefix, GraphNode* Xsrc_feat,
                       GraphNode* Xsrc_node,
                       const std::vector<GroupConfigItem3>& items,
                       bool sparse) {
  auto* feat_embed = XInputGroupEmbeddingLookup(prefix + "node_feat", Xsrc_feat,
                                                items, sparse);
  auto* group_weight = XNodeGroupWeightLookup(prefix + "group_weights",
                                              Xsrc_node, items, sparse);
  auto* src_embed = WeightedAverage("", feat_embed, group_weight);
  return src_embed;
}

}  // namespace embedx
