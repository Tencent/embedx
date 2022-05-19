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
#include "src/model/instance_node_name.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

GraphNode* SparseSageEncoder(const std::string& prefix, GraphNode* self_feat,
                             GraphNode* neigh_feat,
                             const std::vector<GroupConfigItem3>& items,
                             bool sparse, bool is_act, double alpha) {
  auto* self_embed =
      XInputGroupEmbeddingLookup(prefix + "_self", self_feat, items, sparse);
  auto* neigh_embed =
      XInputGroupEmbeddingLookup(prefix + "_neigh", neigh_feat, items, sparse);
  auto* sage_embed = deepx_core::Concat("", {self_embed, neigh_embed});
  if (is_act) {
    sage_embed = deepx_core::LeakyRelu("", sage_embed, alpha);
  }
  return sage_embed;
}

GraphNode* DenseSageEncoder(const std::string& prefix, GraphNode* hidden,
                            GraphNode* self_block, GraphNode* neigh_block,
                            int dim, bool is_act, double alpha) {
  // self emb
  auto* self_embed = HiddenLookup("", self_block, hidden);
  // neighbor mean emb
  auto* neigh_embed = MeanAggregator("", neigh_block, hidden);

  auto* self_fc =
      deepx_core::FullyConnect(prefix + "_self_fc", self_embed, dim);
  auto* neigh_fc =
      deepx_core::FullyConnect(prefix + "_neigh_fc", neigh_embed, dim);

  auto* sage_embed = deepx_core::Concat("", {self_fc, neigh_fc});
  if (is_act) {
    sage_embed = deepx_core::LeakyRelu("", sage_embed, alpha);
  }
  return sage_embed;
}

GraphNode* GraphSageEncoder(const std::string& encoder_name,
                            const std::vector<GroupConfigItem3>& items,
                            int depth, bool use_neigh_feat, bool sparse,
                            double relu_alpha, int dim) {
  auto* Xnode_feat =
      GetXInput(instance_name::X_NODE_FEATURE_NAME + encoder_name);

  GraphNode* next_hidden = nullptr;
  if (use_neigh_feat) {
    auto* Xneigh_feat =
        GetXInput(instance_name::X_NEIGH_FEATURE_NAME + encoder_name);
    bool is_act = depth > 0 ? true : false;
    next_hidden =
        SparseSageEncoder("SparseSageEncoder" + encoder_name, Xnode_feat,
                          Xneigh_feat, items, sparse, is_act, relu_alpha);
  } else {
    next_hidden = XInputGroupEmbeddingLookup("node_feature" + encoder_name,
                                             Xnode_feat, items, sparse);
  }

  const auto& self_blocks =
      GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME + encoder_name, depth);
  const auto& neigh_blocks =
      GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME + encoder_name, depth);
  for (int i = 0; i < depth; ++i) {
    bool is_act = (i + 1) < depth ? true : false;
    next_hidden = DenseSageEncoder(
        encoder_name + "DenseSageEncoder" + std::to_string(i), next_hidden,
        self_blocks[i], neigh_blocks[i], dim, is_act, relu_alpha);
  }
  return next_hidden;
}

GraphNode* GraphSageEncoder(const std::string& encoder_name,
                            const std::vector<GroupConfigItem3>& items,
                            GraphNode* Xnode_feat, GraphNode* Xneigh_feat,
                            const std::vector<GraphNode*> self_blocks,
                            const std::vector<GraphNode*> neigh_blocks,
                            bool sparse, double relu_alpha, int dim) {
  auto depth = self_blocks.size();

  GraphNode* next_hidden = nullptr;
  if (Xneigh_feat != nullptr) {
    bool is_act = depth > 0;
    next_hidden =
        SparseSageEncoder("SparseSageEncoder" + encoder_name, Xnode_feat,
                          Xneigh_feat, items, sparse, is_act, relu_alpha);
  } else {
    next_hidden = XInputGroupEmbeddingLookup("node_feature" + encoder_name,
                                             Xnode_feat, items, sparse);
  }

  for (size_t i = 0; i < depth; ++i) {
    bool is_act = (i + 1) < depth;
    next_hidden = DenseSageEncoder(
        encoder_name + "DenseSageEncoder" + std::to_string(i), next_hidden,
        self_blocks[i], neigh_blocks[i], dim, is_act, relu_alpha);
  }
  return next_hidden;
}

// Different namespaces use different encoders, and then concat the output of
// different encoders as the final representation
GraphNode* HeterGraphSageEncoder(const id_name_t& id_2_name,
                                 const std::string& prefix,
                                 const std::vector<GroupConfigItem3>& items,
                                 int depth, bool use_neigh_feat, bool sparse,
                                 double relu_alpha, int dim) {
  std::vector<GraphNode*> hiddens;
  GraphNode* hidden = nullptr;
  for (auto& entry : id_2_name) {
    auto ns_id = entry.first;
    auto ns_name = entry.second;
    hidden = GraphSageEncoder(prefix + std::to_string(ns_id) + ns_name, items,
                              depth, use_neigh_feat, sparse, relu_alpha, dim);
    hiddens.emplace_back(hidden);
  }
  auto* next_hidden = Concat("", hiddens, 0);
  return next_hidden;
}

}  // namespace embedx
