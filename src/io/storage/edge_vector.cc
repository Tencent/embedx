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

#include "src/io/storage/edge_vector.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

namespace embedx {

EdgeVector::EdgeVector() {
  graph_statics_ = NewGraphStatics(&src_indexing_, &dst_indexing_);
}

void EdgeVector::Clear() noexcept {
  src_node_list_.clear();
  dst_node_list_.clear();
  weight_list_.clear();

  adj_node_list_.clear();
  adj_edge_list_.clear();

  src_indexing_.Clear();
  dst_indexing_.Clear();
}

void EdgeVector::Reserve(uint64_t estimated_size) {
  src_node_list_.reserve(estimated_size);
  dst_node_list_.reserve(estimated_size);
  weight_list_.reserve(estimated_size);

  adj_node_list_.reserve(estimated_size);
  adj_edge_list_.reserve(estimated_size);

  src_indexing_.Reserve(estimated_size);
  dst_indexing_.Reserve(estimated_size);
}

bool EdgeVector::Add(EdgeValue* value) {
  auto edge_id = (int_t)src_node_list_.size();
  src_node_list_.emplace_back(value->src_node);
  dst_node_list_.emplace_back(value->dst_node);
  weight_list_.emplace_back(value->weight);

  // graph stat
  src_indexing_.Add(value->src_node);
  dst_indexing_.Add(value->dst_node);
  if (!graph_statics_->Add(value->src_node, value->dst_node)) {
    return false;
  }

  // fill adj_node and adj_edge
  auto src_index = src_indexing_.Get(value->src_node);
  if (src_index < (int)adj_edge_list_.size()) {
    adj_node_list_[src_index].emplace_back(value->dst_node);
    adj_edge_list_[src_index].emplace_back(edge_id);
  } else {
    vec_int_t dst_node_list(1, value->dst_node);
    adj_node_list_.emplace_back(dst_node_list);

    vec_int_t edge_id_list(1, edge_id);
    adj_edge_list_.emplace_back(edge_id_list);
  }

  return false;
}

bool EdgeVector::GetSrcNode(int_t edge_id, int_t* src_node) const {
  auto src_list_size = (int_t)src_node_list_.size();
  if (edge_id > src_list_size) {
    DXERROR("Need edge_id < src_node_list.size(), got edge_id: %" PRIu64
            " vs src_node_list.size(): %" PRIu64,
            edge_id, src_list_size);
    return false;
  }

  *src_node = src_node_list_[edge_id];
  return true;
}

bool EdgeVector::GetDstNode(int_t edge_id, int_t* dst_node) const {
  auto dst_list_size = (int_t)dst_node_list_.size();
  if (edge_id > dst_list_size) {
    DXERROR("Need edge_id < dst_node_list.size(), got edge_id: %" PRIu64
            " vs dst_node_list.size(): %" PRIu64,
            edge_id, dst_list_size);
    return false;
  }

  *dst_node = dst_node_list_[edge_id];
  return true;
}

bool EdgeVector::GetWeight(int_t edge_id, float_t* weight) const {
  auto weight_list_size = (int_t)weight_list_.size();
  if (edge_id > weight_list_size) {
    DXERROR("Need edge_id < weight_list.size(), got edge_id: %" PRIu64
            " vs weight_list.size(): %" PRIu64,
            edge_id, weight_list_size);
    return false;
  }

  *weight = weight_list_[edge_id];
  return true;
}

const vec_int_t* EdgeVector::FindNeighborNode(int_t node) const {
  int src_index = src_indexing_.Get(node);
  int adj_node_size = (int)adj_node_list_.size();
  if (src_index < 0 || src_index > adj_node_size) {
    DXERROR(
        "Need 0 <= src_index < adj_node_list.size(), got src_index: %d vs "
        "adj_node_list.size(): %d",
        src_index, adj_node_size);
    return nullptr;
  }

  return &adj_node_list_[src_index];
}

const vec_int_t* EdgeVector::FindNeighborEdge(int_t node) const {
  int src_index = src_indexing_.Get(node);
  int adj_edge_size = (int)adj_edge_list_.size();
  if (src_index < 0 || src_index > adj_edge_size) {
    DXERROR(
        "Need 0 <= src_index < adj_edge_list.size(), got src_index: %d vs "
        "adj_edge_list.size(): %d",
        src_index, adj_edge_size);
    return nullptr;
  }

  return &adj_edge_list_[src_index];
}

std::string EdgeVector::Print(int_t edge_id) const {
  std::stringstream ss;
  int_t node;
  float_t weight;
  DXCHECK(GetSrcNode(edge_id, &node));
  ss << "src_node:" << node;
  DXCHECK(GetDstNode(edge_id, &node));
  ss << " dst_node:" << node;
  DXCHECK(GetWeight(edge_id, &weight));
  ss << " weight:" << weight;
  return ss.str();
}

int EdgeVector::GetInDegree(int_t node) const {
  return graph_statics_->GetInDegree(node);
}

int EdgeVector::GetOutDegree(int_t node) const {
  return graph_statics_->GetOutDegree(node);
}

std::unique_ptr<EdgeVector> NewEdgeVector(int /* store_type */) {
  std::unique_ptr<EdgeVector> edge_vector;
  edge_vector.reset(new EdgeVector());
  return edge_vector;
}

}  // namespace embedx
