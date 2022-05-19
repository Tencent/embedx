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

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64
#include <memory>     // std::unique_ptr
#include <sstream>    // std::stringstream
#include <string>

#include "src/common/data_types.h"
#include "src/io/storage/adjacency_impl.h"
#include "src/io/value.h"

namespace embedx {

class AdjListImpl : public AdjacencyImpl {
 private:
  vec_int_t keys_;
  adj_list_t adj_list_;
  index_map_t in_degree_;

 public:
  ~AdjListImpl() override = default;

 public:
  size_t Size() const noexcept override { return adj_list_.size(); }
  bool Empty() const noexcept override { return adj_list_.empty(); }
  const vec_int_t& Keys() const noexcept override { return keys_; }

 public:
  void Clear() noexcept override {
    adj_list_.clear();
    in_degree_.clear();
    keys_.clear();
  }

  void Reserve(uint64_t estimated_size) override {
    adj_list_.reserve(estimated_size);
    in_degree_.reserve(estimated_size);
    keys_.reserve(estimated_size);
  }

  bool AddContext(AdjValue* value) override {
    if (FindNeighbor(value->node) != nullptr) {
      DXERROR(
          "Need unique node in the graph file, got duplicate node: %" PRIu64,
          value->node);
      return false;
    }

    // TODO(longsail): which sorting function to use
    AdjacencyImpl::SortByNode(&value->pairs);
    keys_.emplace_back(value->node);
    adj_list_.emplace(value->node, value->pairs);

    for (auto& pair : value->pairs) {
      auto it = in_degree_.find(pair.first);
      if (it != in_degree_.end()) {
        ++it->second;
      } else {
        in_degree_.emplace(pair.first, 1);
      }
    }

    return true;
  }

  bool AddFeature(AdjValue* value) override {
    if (FindNeighbor(value->node) != nullptr) {
      DXERROR(
          "Need unique node in the feature file, got duplicate node: %" PRIu64,
          value->node);
      return false;
    }

    keys_.emplace_back(value->node);
    adj_list_.emplace(value->node, value->pairs);

    return true;
  }

  const vec_pair_t* FindNeighbor(int_t node) const override {
    auto it = adj_list_.find(node);
    if (it != adj_list_.end()) {
      return &it->second;
    }

    return nullptr;
  }

  std::string Print(int_t node) const override {
    std::stringstream ss;
    ss << "Key:" << node;
    ss << " value:";
    auto* it = FindNeighbor(node);
    if (it != nullptr) {
      for (auto& pair : *it) {
        ss << " " << pair.first << ":" << pair.second;
      }
    } else {
      ss << " is nullptr.";
    }

    return ss.str();
  }

  int GetInDegree(int_t node) const override {
    auto it = in_degree_.find(node);
    if (it != in_degree_.end()) {
      return it->second;
    }
    return 0;
  }

  int GetOutDegree(int_t node) const override {
    auto it = adj_list_.find(node);
    if (it != adj_list_.end()) {
      return it->second.size();
    }
    return 0;
  }
};

std::unique_ptr<AdjacencyImpl> NewAdjListImpl() {
  std::unique_ptr<AdjacencyImpl> adjacency_impl;
  adjacency_impl.reset(new AdjListImpl);
  return adjacency_impl;
}

}  // namespace embedx
