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
#include <vector>

#include "src/common/data_types.h"
#include "src/io/indexing.h"
#include "src/io/storage/graph_statics.h"
#include "src/io/value.h"

namespace embedx {

class EdgeVector {
 private:
  vec_int_t src_node_list_;
  vec_int_t dst_node_list_;
  vec_float_t weight_list_;

  std::vector<vec_int_t> adj_node_list_;
  std::vector<vec_int_t> adj_edge_list_;

  Indexing src_indexing_;
  Indexing dst_indexing_;
  std::unique_ptr<GraphStatics> graph_statics_;

 public:
  EdgeVector();
  ~EdgeVector() = default;

 public:
  void Clear() noexcept;
  void Reserve(uint64_t estimated_size);
  bool Add(EdgeValue* value);

  size_t Size() const noexcept { return src_node_list_.size(); }
  bool Empty() const noexcept { return src_node_list_.empty(); }

 public:
  bool GetSrcNode(int_t edge_id, int_t* src_node) const;
  bool GetDstNode(int_t edge_id, int_t* dst_node) const;
  bool GetWeight(int_t edge_id, float_t* weight) const;
  const vec_int_t* FindNeighborNode(int_t node) const;
  const vec_int_t* FindNeighborEdge(int_t node) const;
  std::string Print(int_t edge_id) const;
  int GetInDegree(int_t node) const;
  int GetOutDegree(int_t node) const;

 public:
  // An EDGE is made up of [src_node, dst_node, weight].
  const vec_int_t& src_node_list() const noexcept { return src_node_list_; }
  const vec_int_t& dst_node_list() const noexcept { return dst_node_list_; }
  const vec_float_t& weight_list() const noexcept { return weight_list_; }

  const std::vector<vec_int_t>& adj_node_list() const noexcept {
    return adj_node_list_;
  }
  const std::vector<vec_int_t>& adj_edge_list() const noexcept {
    return adj_edge_list_;
  }

  const Indexing& src_indexing() const noexcept { return src_indexing_; }
  const Indexing& dst_indexing() const noexcept { return dst_indexing_; }
};

std::unique_ptr<EdgeVector> NewEdgeVector(int store_type);

}  // namespace embedx
