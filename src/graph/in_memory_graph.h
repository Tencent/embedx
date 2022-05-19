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
#include "src/graph/graph_builder.h"
#include "src/graph/graph_config.h"
#include "src/graph/post_builder.h"

namespace embedx {

class InMemoryGraph {
 private:
  std::unique_ptr<GraphBuilder> graph_builder_;
  std::unique_ptr<PostBuilder> post_builder_;

 public:
  static std::unique_ptr<InMemoryGraph> Create(const GraphConfig& config);

 public:
  int ns_size() const noexcept { return post_builder_->ns_size(); }
  const std::vector<vec_int_t>& uniq_nodes_list() const noexcept {
    return post_builder_->uniq_nodes_list();
  }
  const std::vector<vec_float_t>& uniq_freqs_list() const noexcept {
    return post_builder_->uniq_freqs_list();
  }
  const vec_int_t& total_freqs() const noexcept {
    return post_builder_->total_freqs();
  }
  const id_name_t& id_name_map() const noexcept {
    return post_builder_->id_name_map();
  }

 public:
  // degree
  int GetInDegree(int_t dst_node) const {
    return graph_builder_->context_storage()->GetInDegree(dst_node);
  }
  int GetOutDegree(int_t src_node) const {
    return graph_builder_->context_storage()->GetOutDegree(src_node);
  }

  // keys
  const vec_int_t& node_keys() const noexcept {
    return graph_builder_->context_storage()->Keys();
  }
  const vec_int_t& node_feature_keys() const noexcept {
    return graph_builder_->node_feature_storage()->Keys();
  }
  const vec_int_t& neigh_feature_keys() const noexcept {
    return graph_builder_->neigh_feature_storage()->Keys();
  }

  // find
  const vec_pair_t* FindContext(int_t node) const {
    return graph_builder_->context_storage()->FindNeighbor(node);
  }
  const vec_pair_t* FindNodeFeature(int_t node) const {
    return graph_builder_->node_feature_storage()->FindNeighbor(node);
  }
  const vec_pair_t* FindNeighFeature(int_t node) const {
    return graph_builder_->neigh_feature_storage()->FindNeighbor(node);
  }

  // size
  size_t node_size() const noexcept {
    return graph_builder_->context_storage()->Size();
  }
  size_t node_feature_size() const noexcept {
    return graph_builder_->node_feature_storage()->Size();
  }
  size_t neigh_feature_size() const noexcept {
    return graph_builder_->neigh_feature_storage()->Size();
  }

  // empty
  bool node_empty() const noexcept {
    return graph_builder_->context_storage()->Empty();
  }
  bool node_feature_empty() const noexcept {
    return graph_builder_->node_feature_storage()->Empty();
  }
  bool neigh_feature_empty() const noexcept {
    return graph_builder_->neigh_feature_storage()->Empty();
  }

 private:
  bool Build(const GraphConfig& config);
  bool CheckSizeValid() const;
  void PrintGraphTopo() const;

 private:
  InMemoryGraph() = default;
};

}  // namespace embedx
