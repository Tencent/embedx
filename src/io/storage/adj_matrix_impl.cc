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
#include <vector>

#include "src/common/data_types.h"
#include "src/io/indexing.h"
#include "src/io/storage/adjacency_impl.h"
#include "src/io/storage/graph_statics.h"
#include "src/io/value.h"

namespace embedx {

class AdjMatrixImpl : public AdjacencyImpl {
 private:
  Indexing src_indexing_;
  Indexing dst_indexing_;
  std::vector<vec_pair_t> adj_matrix_;
  std::unique_ptr<GraphStatics> graph_statics_;

 public:
  AdjMatrixImpl() {
    graph_statics_ = NewGraphStatics(&src_indexing_, &dst_indexing_);
  }
  ~AdjMatrixImpl() override = default;

 public:
  size_t Size() const noexcept override { return adj_matrix_.size(); }
  bool Empty() const noexcept override { return adj_matrix_.empty(); }
  const vec_int_t& Keys() const noexcept override {
    return *graph_statics_->src_node_list();
  }

  void Clear() noexcept override {
    src_indexing_.Clear();
    dst_indexing_.Clear();
    adj_matrix_.clear();
  }

  void Reserve(uint64_t estimated_size) override {
    src_indexing_.Reserve(estimated_size);
    dst_indexing_.Reserve(estimated_size);
    adj_matrix_.reserve(estimated_size);
  }

  bool AddContext(AdjValue* value) override {
    if (src_indexing_.Find(value->node)) {
      DXERROR(
          "Need unique node in the graph file, got duplicate node: %" PRIu64,
          value->node);
      return false;
    }

    // TODO(longsail): which sorting function to use
    AdjacencyImpl::SortByNode(&value->pairs);
    src_indexing_.Add(value->node);
    adj_matrix_.emplace_back(value->pairs);

    for (auto& pair : value->pairs) {
      dst_indexing_.Add(pair.first);
      // edge : [value->node, pair.first]
      if (!graph_statics_->Add(value->node, pair.first)) {
        return false;
      }
    }

    return true;
  }

  bool AddFeature(AdjValue* value) override {
    if (src_indexing_.Find(value->node)) {
      DXERROR(
          "Need unique node in the feature file, got duplicate node: %" PRIu64,
          value->node);
      return false;
    }

    src_indexing_.Add(value->node);
    adj_matrix_.emplace_back(value->pairs);

    return true;
  }

  const vec_pair_t* FindNeighbor(int_t node) const override {
    int src_index = src_indexing_.Get(node);
    int adj_matrix_size = (int)adj_matrix_.size();
    if (src_index < 0 || src_index > (int)adj_matrix_.size()) {
      DXERROR(
          "Need 0 <= src_index < adj_matrix.size(), Got src_index: %d vs "
          "adj_matrix.size(): %d",
          src_index, adj_matrix_size);
      return nullptr;
    }
    return &adj_matrix_[src_index];
  }

  std::string Print(int_t node) const override {
    std::stringstream ss;
    ss << "Key:" << node;
    ss << " value:";
    int src_index = src_indexing_.Get(node);
    if (src_index >= 0) {
      if (src_index >= (int)adj_matrix_.size()) {
        DXTHROW_INVALID_ARGUMENT(
            "Invalid src_index, src_index: %d must be less than %zu.",
            src_index, adj_matrix_.size());
      }

      auto& adj_node = adj_matrix_[src_index];
      for (auto& pair : adj_node) {
        ss << " " << pair.first << ":" << pair.second;
      }
    } else {
      ss << " is nullptr.";
    }

    return ss.str();
  }

  int GetInDegree(int_t node) const override {
    return graph_statics_->GetInDegree(node);
  }

  int GetOutDegree(int_t node) const override {
    return graph_statics_->GetOutDegree(node);
  }
};

std::unique_ptr<AdjacencyImpl> NewAdjMatrixImpl() {
  std::unique_ptr<AdjacencyImpl> adjacency_impl;
  adjacency_impl.reset(new AdjMatrixImpl);
  return adjacency_impl;
}

}  // namespace embedx
