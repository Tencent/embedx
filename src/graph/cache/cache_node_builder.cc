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
#include "src/io/io_util.h"

namespace embedx {

bool CacheNodeBuilder::RandomCache(const vec_int_t& nodes, int thread_id) {
  DXINFO("RandomCache thread: %d is processing ...", thread_id);
  if (cache_thld_ * nodes.size() < 1) {
    DXERROR("RandomCache error.cache nodes need >= 1");
    return false;
  }
  for (auto& node : nodes) {
    if (ThreadLocalRandom() < cache_thld_) {
      std::lock_guard<std::mutex> guard(mtx_);
      nodes_.emplace_back(node);
    }
  }
  DXINFO("RandomCache thread: %d Done", thread_id);
  return true;
}

bool CacheNodeBuilder::DegreeCache(const vec_int_t& nodes, int thread_id) {
  std::vector<std::pair<int_t, int_t>> tmp_out_degrees;
  int count = nodes.size() * cache_thld_;
  if (count < 1) {
    DXERROR("DegreeCache error.cache nodes need >= 1");
    return false;
  }
  for (auto& node : nodes) {
    int tmp_out_degree = graph_->GetOutDegree(node);
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
    std::lock_guard<std::mutex> guard(mtx_);
    nodes_.emplace_back(tmp_out_degrees[i].first);
  }
  DXINFO("DegreeCache thread: %d Done", thread_id);
  return true;
}

bool CacheNodeBuilder::ImportanceCache(const vec_int_t& nodes, int thread_id) {
  DXINFO("ImportanceCache thread: %d is processing ...", thread_id);
  for (auto& node : nodes) {
    int tmp_out_degree = graph_->GetOutDegree(node);
    int tmp_in_degree = graph_->GetInDegree(node);
    DXCHECK(tmp_out_degree >= 0 && tmp_in_degree >= 0);
    float_t node_importance =
        (tmp_in_degree + 1) / ((tmp_out_degree + 1) * 1.0);
    if (node_importance <= 0) {
      DXERROR("Node: %" PRIu64 " importance factor error.", node);
      return false;
    }
    if (node_importance > cache_thld_) {
      std::lock_guard<std::mutex> guard(mtx_);
      nodes_.emplace_back(node);
    }
  }
  DXINFO("ImportanceCache thread: %d Done", thread_id);
  return true;
}

bool CacheNodeBuilder::Build(const InMemoryGraph* graph, int cache_type,
                             double cache_thld, int thread_num) {
  nodes_.clear();
  if (cache_thld == 0) {
    DXINFO("Cache_thld == 0, cache was not enabled.");
    return true;
  } else if (cache_thld < 0) {
    DXERROR("Need cache_thld > 0, got cache_thld: %f.", cache_thld);
    return false;
  }
  cache_thld_ = cache_thld;
  graph_ = graph;

  auto& nodes = graph->node_keys();
  DXINFO("Cache_type = %d,cache_thld = %f.", cache_type, cache_thld);

  if (cache_type == 0) {
    // return RandomCache(node_keys, cache_thld, &nodes_);
    return io_util::ParallelProcess<int_t>(
        nodes,
        [this](const vec_int_t& nodes, int thread_id) {
          return RandomCache(nodes, thread_id);
        },
        thread_num);
  } else if (cache_type == 1) {
    // return DegreeCache(graph, node_keys, cache_thld, &nodes_);
    return io_util::ParallelProcess<int_t>(
        nodes,
        [this](const vec_int_t& nodes, int thread_id) {
          return DegreeCache(nodes, thread_id);
        },
        thread_num);
  } else if (cache_type == 2) {
    // return ImportanceCache(graph, node_keys, cache_thld, &nodes_);

    return io_util::ParallelProcess<int_t>(
        nodes,
        [this](const vec_int_t& nodes, int thread_id) {
          return ImportanceCache(nodes, thread_id);
        },
        thread_num);
  } else {
    DXERROR("Need type: random(0) || degree(1) || importance(2), got type: %d.",
            (int)cache_type);
    return false;
  }
  return true;
}

std::unique_ptr<CacheNodeBuilder> CacheNodeBuilder::Create(
    const InMemoryGraph* graph, int cache_type, double cache_thld,
    int thread_num) {
  std::unique_ptr<CacheNodeBuilder> cache_node_builder;
  cache_node_builder.reset(new CacheNodeBuilder());

  if (!cache_node_builder->Build(graph, cache_type, cache_thld, thread_num)) {
    DXERROR("Failed to build cache node builder.");
    cache_node_builder.reset();
  }

  return cache_node_builder;
}

}  // namespace embedx
