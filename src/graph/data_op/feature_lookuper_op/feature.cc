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

#include "src/graph/data_op/feature_lookuper_op/feature.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

namespace embedx {
namespace graph_op {
namespace {

const vec_pair_t EMPTY_FEATURE = {{0, 0}};

}  // namespace

bool Feature::LookupFeature(const vec_int_t& nodes,
                            std::vector<vec_pair_t>* node_feats,
                            std::vector<vec_pair_t>* neigh_feats) const {
  if (!LookupNodeFeature(nodes, node_feats)) {
    return false;
  }

  if (!LookupNeighborFeature(nodes, neigh_feats)) {
    return false;
  }

  return true;
}

bool Feature::LookupNodeFeature(const vec_int_t& nodes,
                                std::vector<vec_pair_t>* node_feats) const {
  node_feats->clear();
  for (auto node : nodes) {
    if (graph_.FindContext(node) == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", node);
    }

    const auto* feat = graph_.FindNodeFeature(node);
    if (feat == nullptr) {
      // insert an empty feature
      node_feats->emplace_back(EMPTY_FEATURE);
    } else {
      node_feats->emplace_back(*feat);
    }
  }

  return nodes.size() == node_feats->size();
}

bool Feature::LookupNeighborFeature(
    const vec_int_t& nodes, std::vector<vec_pair_t>* neighbor_feats) const {
  neighbor_feats->clear();
  for (auto node : nodes) {
    if (graph_.FindContext(node) == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", node);
    }

    const auto* feat = graph_.FindNeighFeature(node);
    if (feat == nullptr) {
      // insert an empty feature
      neighbor_feats->emplace_back(EMPTY_FEATURE);
    } else {
      neighbor_feats->emplace_back(*feat);
    }
  }

  return nodes.size() == neighbor_feats->size();
}

std::unique_ptr<Feature> NewFeature(const InMemoryGraph* graph) {
  std::unique_ptr<Feature> feature;
  feature.reset(new Feature(graph));
  return feature;
}

}  // namespace graph_op
}  // namespace embedx
