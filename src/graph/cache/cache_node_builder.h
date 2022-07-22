// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanqingguo (yuanqingsunny1180@gmail.com)
//

#pragma once
#include <memory>  // std::unique_ptr
#include <mutex>

#include "src/common/data_types.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {

class CacheNodeBuilder {
 private:
  vec_int_t nodes_;
  const InMemoryGraph* graph_;
  double cache_thld_;
  std::mutex mtx_;

 public:
  static std::unique_ptr<CacheNodeBuilder> Create(const InMemoryGraph* graph,
                                                  int cache_type,
                                                  double cache_thld,
                                                  int thread_num);

 public:
  const vec_int_t* nodes() const noexcept { return &nodes_; }

 private:
  bool Build(const InMemoryGraph* graph, int cache_type, double cache_thld,
             int thread_num);
  bool RandomCache(const vec_int_t& nodes, int thread_id);
  bool DegreeCache(const vec_int_t& nodes, int thread_id);
  bool ImportanceCache(const vec_int_t& nodes, int thread_id);

 private:
  CacheNodeBuilder() = default;
};

}  // namespace embedx
