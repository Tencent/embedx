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

#include "src/io/indexing_wrapper.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

#include "src/io/io_util.h"

namespace embedx {

void IndexingWrapper::Clear() noexcept {
  subgraph_size_ = 0;

  for (int i = 0; i < ns_size_; ++i) {
    subgraph_offset_[i] = 0;
    auto& subgraph_indexing = subgraph_indexings_[i];
    for (auto& indexing : subgraph_indexing) {
      indexing.Clear();
    }
  }
}

void IndexingWrapper::BuildFrom(const vec_int_t& nodes) {
  DXCHECK(!nodes.empty());

  uint16_t ns_id = 0;
  if (ns_size_ > 1) {
    ns_id = io_util::GetNodeType(nodes[0]);
  }
  DXCHECK(ns_id < ns_size_);

  auto& subgraph_indexing = subgraph_indexings_[ns_id];
  subgraph_indexing.resize(1);
  subgraph_indexing[0].Clear();

  for (auto node : nodes) {
    subgraph_indexing[0].Add(node);
  }
}

void IndexingWrapper::BuildFrom(const vec_set_t& level_nodes) {
  DXCHECK(!level_nodes.empty() && !level_nodes[0].empty());

  uint16_t ns_id = 0;
  if (ns_size_ > 1) {
    auto first_node = *(level_nodes[0].begin());
    ns_id = io_util::GetNodeType(first_node);
  }
  DXCHECK(ns_id < ns_size_);

  auto& subgraph_indexing = subgraph_indexings_[ns_id];
  subgraph_indexing.resize(level_nodes.size());

  int k = 0;
  for (size_t i = 0; i < level_nodes.size(); ++i) {
    subgraph_indexing[i].Clear();
    for (auto node : level_nodes[i]) {
      subgraph_indexing[i].Emplace(node, k);
      k += 1;
    }
  }

  subgraph_offset_[ns_id] = subgraph_size_;
  subgraph_size_ += subgraph_indexing[0].Size();
}

int IndexingWrapper::GlobalGet(int_t node) const {
  uint16_t ns_id = 0;
  if (ns_size_ > 1) {
    ns_id = io_util::GetNodeType(node);
  }
  DXCHECK(ns_id < ns_size_);

  const auto& indexing = subgraph_indexings_[ns_id][0];
  if (indexing.Get(node) == -1) {
    return -1;
  } else {
    return indexing.Get(node) + subgraph_offset_[ns_id];
  }
}

bool IndexingWrapper::Init(const std::string& config) {
  id_name_t id_name_map;
  if (!io_util::LoadConfig(config, &ns_size_, &id_name_map)) {
    return false;
  }

  subgraph_size_ = 0;
  subgraph_offset_.resize(ns_size_);
  subgraph_indexings_.resize(ns_size_);
  return true;
}

std::unique_ptr<IndexingWrapper> IndexingWrapper::Create(
    const std::string& config) {
  std::unique_ptr<IndexingWrapper> indexing_wrapper;
  indexing_wrapper.reset(new IndexingWrapper());

  if (!indexing_wrapper->Init(config)) {
    DXERROR("Failed to init indexing wrapper.");
    indexing_wrapper.reset();
  }

  return indexing_wrapper;
}

}  // namespace embedx
