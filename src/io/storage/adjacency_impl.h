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
#include <algorithm>  // std::stable_sort
#include <memory>     // std::unique_ptr
#include <string>

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/io/value.h"

namespace embedx {

class AdjacencyImpl {
 public:
  AdjacencyImpl() = default;
  virtual ~AdjacencyImpl() = default;

 public:
  virtual void Clear() noexcept = 0;
  virtual void Reserve(uint64_t estimated_size) = 0;
  virtual bool AddContext(AdjValue* value) = 0;
  virtual bool AddFeature(AdjValue* value) = 0;

 public:
  virtual size_t Size() const noexcept = 0;
  virtual bool Empty() const noexcept = 0;
  virtual const vec_int_t& Keys() const noexcept = 0;

 public:
  virtual const vec_pair_t* FindNeighbor(int_t node) const = 0;
  virtual std::string Print(int_t node) const = 0;
  virtual int GetInDegree(int_t dst_node) const = 0;
  virtual int GetOutDegree(int_t src_node) const = 0;

 protected:
  void SortByNode(vec_pair_t* context) const {
    std::stable_sort(context->begin(), context->end(),
                     [&](const pair_t& a, const pair_t& b) {
                       uint16_t type_a = io_util::GetNodeType(a.first);
                       uint16_t type_b = io_util::GetNodeType(b.first);
                       if (type_a == type_b) {
                         return a.first < b.first;
                       } else {
                         return type_a < type_b;
                       }
                     });
  }

  void SortByWeight(vec_pair_t* context) const {
    std::stable_sort(context->begin(), context->end(),
                     [&](const pair_t& a, const pair_t& b) {
                       uint16_t type_a = io_util::GetNodeType(a.first);
                       uint16_t type_b = io_util::GetNodeType(b.first);
                       if (type_a == type_b) {
                         return a.second < b.second;
                       } else {
                         return type_a < type_b;
                       }
                     });
  }
};

std::unique_ptr<AdjacencyImpl> NewAdjListImpl();
std::unique_ptr<AdjacencyImpl> NewAdjMatrixImpl();

}  // namespace embedx
