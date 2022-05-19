// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/sampler/neighbor_sampler.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::find_if
#include <utility>    // std::move

namespace embedx {

bool NeighborSampler::Sample(
    int count, const vec_int_t& nodes,
    std::vector<vec_int_t>* neighbor_nodes_list) const {
  neighbor_nodes_list->clear();
  neighbor_nodes_list->resize(nodes.size());

  int empty_node_num = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    DoSampling(nodes[i], count, &(*neighbor_nodes_list)[i]);
    if ((*neighbor_nodes_list)[i].empty()) {
      empty_node_num += 1;
    }
  }

  // if all nodes are not in the graph, return false
  return (int)nodes.size() > empty_node_num;
}

void NeighborSampler::DoSampling(int_t node, int count,
                                 vec_int_t* neighbor_nodes) const {
  const auto* context = sampler_builder_.sampler_source().FindContext(node);
  DXCHECK(context != nullptr);
  int neighbor_size = (int)context->size();

  if (count < 0 || count == neighbor_size) {
    FullSampling(node, neighbor_nodes);
  } else if (count < neighbor_size) {
    NoReplacementSampling(node, count, neighbor_nodes);
  } else {
    WithReplacementSampling(node, count, neighbor_nodes);
  }
}

void NeighborSampler::FullSampling(int_t node,
                                   vec_int_t* neighbor_nodes) const {
  neighbor_nodes->clear();

  const auto* context = sampler_builder_.sampler_source().FindContext(node);
  if (context == nullptr) {
    return;
  }

  for (const auto& pair : *context) {
    neighbor_nodes->emplace_back(pair.first);
  }
}

void NeighborSampler::NoReplacementSampling(int_t node, int count,
                                            vec_int_t* neighbor_nodes) const {
  neighbor_nodes->clear();

  int_t next_node;
  while (neighbor_nodes->size() < (size_t)count) {
    DXCHECK(sampler_builder_.Next(node, &next_node));
    // o(n) !!!
    auto it = std::find_if(neighbor_nodes->begin(), neighbor_nodes->end(),
                           [next_node](int_t neighbor_node) {
                             return neighbor_node == next_node;
                           });
    if (it == neighbor_nodes->end()) {
      neighbor_nodes->emplace_back(next_node);
    }
  }
}

void NeighborSampler::WithReplacementSampling(int_t node, int count,
                                              vec_int_t* neighbor_nodes) const {
  neighbor_nodes->clear();

  int_t next_node;
  for (int i = 0; i < count; ++i) {
    DXCHECK(sampler_builder_.Next(node, &next_node));
    neighbor_nodes->emplace_back(next_node);
  }
}

std::unique_ptr<NeighborSampler> NewNeighborSampler(
    const SamplerBuilder* sampler_builder) {
  std::unique_ptr<NeighborSampler> sampler;
  sampler.reset(new NeighborSampler(sampler_builder));
  return sampler;
}

}  // namespace embedx
