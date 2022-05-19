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

#include "src/graph/in_memory_graph.h"

#include <deepx_core/dx_log.h>

namespace embedx {

/************************************************************************/
/* Build graph */
/************************************************************************/
bool InMemoryGraph::Build(const GraphConfig& config) {
  DXINFO("Build in memory graph...");

  graph_builder_ = GraphBuilder::Create(config);
  if (!graph_builder_) {
    return false;
  }

  post_builder_ =
      PostBuilder::Create(graph_builder_->context_storage(), config);
  if (!post_builder_) {
    return false;
  }

  if (!CheckSizeValid()) {
    return false;
  }

  PrintGraphTopo();

  DXINFO("Done.");
  return true;
}

bool InMemoryGraph::CheckSizeValid() const {
  // len(node_feat_list) <= len(context_list)
  if (!node_feature_empty()) {
    if (node_feature_size() > node_size()) {
      DXERROR(
          "Need node_feature.size() <= node_graph.size(), got "
          "node_feature.size(): %zu vs node_graph.size(): %zu.",
          node_feature_size(), node_size());
      return false;
    }
  }

  // len(neigh_feat_list) <= len(context_list)
  if (!neigh_feature_empty()) {
    if (neigh_feature_size() > node_size()) {
      DXERROR(
          "Need neighbor_feature.size() <= node_graph.size(), got "
          "neighbor_feature.size(): %zu vs node_graph.size(): %zu.",
          neigh_feature_size(), node_size());
      return false;
    }
  }

  return true;
}

void InMemoryGraph::PrintGraphTopo() const {
  for (const auto& entry : id_name_map()) {
    auto ns_id = entry.first;
    auto ns_name = entry.second;
    auto& uniq_nodes = uniq_nodes_list()[ns_id];
    DXINFO("Number of namespace: %s nodes are: %zu.", ns_name.c_str(),
           uniq_nodes.size());
    auto& freq = total_freqs()[ns_id];
    DXINFO("Frequency of namespace: %s nodes are: %d.", ns_name.c_str(),
           (int)freq);
  }
}

std::unique_ptr<InMemoryGraph> InMemoryGraph::Create(
    const GraphConfig& config) {
  std::unique_ptr<InMemoryGraph> graph;
  graph.reset(new InMemoryGraph());

  if (!graph->Build(config)) {
    DXERROR("Failed to create in memory graph.");
    graph.reset();
  }

  return graph;
}

}  // namespace embedx
