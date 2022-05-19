// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanqing Guo (yuanqingsunny1180@gmail.com)
//

#include "src/graph/cache/cache_node_builder.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::stable_sort
#include <cinttypes>  // PRIu64
#include <utility>    // std::pair
#include <vector>

#include "src/common/random.h"

namespace embedx {
namespace {

bool RandomCache(const vec_int_t& nodes, double percent,
                 vec_int_t* cached_nodes) {
  cached_nodes->clear();
  for (auto& node : nodes) {
    if (ThreadLocalRandom() < percent) {
      cached_nodes->emplace_back(node);
    }
  }
  return !cached_nodes->empty();
}

bool DegreeCache(const InMemoryGraph* graph, const vec_int_t& nodes,
                 double percent, vec_int_t* cached_nodes) {
  cached_nodes->clear();

  std::vector<std::pair<int_t, int_t>> tmp_out_degrees;
  int count = nodes.size() * percent;

  for (auto& node : nodes) {
    int tmp_out_degree = graph->GetOutDegree(node);
    if (tmp_out_degree < 0) {
      DXERROR("Output degree of node: %" PRIu64
              " must be greater than or equal to 0.",
              node);
      return false;
    }
    tmp_out_degrees.emplace_back(std::make_pair(node, tmp_out_degree));
  }

  std::stable_sort(
      tmp_out_degrees.begin(), tmp_out_degrees.end(),
      [=](const std::pair<int_t, int_t>& a, const std::pair<int_t, int_t>& b) {
        return a.second > b.second;
      });

  for (int i = 0; i < count; ++i) {
    cached_nodes->emplace_back(tmp_out_degrees[i].first);
  }

  return !cached_nodes->empty();
}

bool ImportanceCache(const InMemoryGraph* graph, const vec_int_t& nodes,
                     double importance_factor, vec_int_t* cached_nodes) {
  cached_nodes->clear();

  for (auto& node : nodes) {
    int tmp_out_degree = graph->GetOutDegree(node);
    int tmp_in_degree = graph->GetInDegree(node);
    DXCHECK(tmp_out_degree >= 0 && tmp_in_degree >= 0);
    float_t node_importance =
        (tmp_in_degree + 1) / ((tmp_out_degree + 1) * 1.0);
    if (node_importance > importance_factor) {
      cached_nodes->emplace_back(node);
    }
  }

  return !cached_nodes->empty();
}

}  // namespace

bool CacheNodeBuilder::Build(const InMemoryGraph* graph, int cache_type,
                             double cache_thld) {
  nodes_.clear();
  if (cache_thld == 0) {
    DXINFO("Cache_thld == 0, cache was not enabled.");
    return true;
  } else if (cache_thld < 0) {
    DXERROR("Need cache_thld > 0, got cache_thld: %f.", cache_thld);
    return false;
  }

  auto& node_keys = graph->node_keys();
  DXINFO("Cache_type = %d,cache_thld = %f.", cache_type, cache_thld);

  if (cache_type == 0) {
    return RandomCache(node_keys, cache_thld, &nodes_);
  } else if (cache_type == 1) {
    return DegreeCache(graph, node_keys, cache_thld, &nodes_);
  } else if (cache_type == 2) {
    return ImportanceCache(graph, node_keys, cache_thld, &nodes_);
  } else {
    DXERROR("Need type: random(0) || degree(1) || importance(2), got type: %d.",
            (int)cache_type);
    return false;
  }
  return true;
}

std::unique_ptr<CacheNodeBuilder> CacheNodeBuilder::Create(
    const InMemoryGraph* graph, int cache_type, double cache_thld) {
  std::unique_ptr<CacheNodeBuilder> cache_node_builder;
  cache_node_builder.reset(new CacheNodeBuilder());

  if (!cache_node_builder->Build(graph, cache_type, cache_thld)) {
    DXERROR("Failed to build cache node builder.");
    cache_node_builder.reset();
  }

  return cache_node_builder;
}

}  // namespace embedx
