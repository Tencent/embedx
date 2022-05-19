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
#include <string>

#include "src/common/data_types.h"
#include "src/io/value.h"

namespace embedx {

class AdjacencyImpl;

class Adjacency {
 private:
  std::unique_ptr<AdjacencyImpl> impl_;

 public:
  explicit Adjacency(std::unique_ptr<AdjacencyImpl>&& impl);
  ~Adjacency();

 public:
  void Clear() noexcept;
  void Reserve(uint64_t estimated_size);
  bool AddContext(AdjValue* value);
  bool AddFeature(AdjValue* value);

 public:
  size_t Size() const noexcept;
  bool Empty() const noexcept;
  const vec_int_t& Keys() const noexcept;

 public:
  const vec_pair_t* FindNeighbor(int_t node) const;
  std::string Print(int_t node) const;
  int GetInDegree(int_t dst_node) const;
  int GetOutDegree(int_t src_node) const;
};

enum class AdjacencyEnum : int {
  ADJ_LIST = 0,
  ADJ_MATRIX = 1,
};

std::unique_ptr<Adjacency> NewAdjacency(AdjacencyEnum type);

}  // namespace embedx
