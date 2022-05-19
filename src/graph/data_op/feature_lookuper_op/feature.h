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

#pragma once
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {
namespace graph_op {

class Feature {
 private:
  const InMemoryGraph& graph_;

 public:
  explicit Feature(const InMemoryGraph* graph) : graph_(*graph) {}

 public:
  bool LookupFeature(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* node_feats,
                     std::vector<vec_pair_t>* neigh_feats) const;
  bool LookupNodeFeature(const vec_int_t& nodes,
                         std::vector<vec_pair_t>* node_feats) const;
  bool LookupNeighborFeature(const vec_int_t& nodes,
                             std::vector<vec_pair_t>* neighbor_feats) const;
};

std::unique_ptr<Feature> NewFeature(const InMemoryGraph* graph);

}  // namespace graph_op
}  // namespace embedx
