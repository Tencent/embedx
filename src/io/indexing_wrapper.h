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

namespace embedx {

class IndexingWrapper {
 public:
  using subgraph_indexing_t = std::vector<Indexing>;

 private:
  uint16_t ns_size_ = 1;
  std::vector<subgraph_indexing_t> subgraph_indexings_;

  int subgraph_size_ = 0;
  std::vector<uint16_t> subgraph_offset_;

 public:
  static std::unique_ptr<IndexingWrapper> Create(const std::string& config);

 public:
  void Clear() noexcept;
  void BuildFrom(const vec_int_t& nodes);
  void BuildFrom(const vec_set_t& level_nodes);
  int Index(int_t node) const;

  const subgraph_indexing_t& subgraph_indexing(uint16_t id) const noexcept {
    return subgraph_indexings_[id];
  }
  subgraph_indexing_t& subgraph_indexing(uint16_t id) noexcept {
    return subgraph_indexings_[id];
  }

 private:
  bool Init(const std::string& config);

 private:
  IndexingWrapper() = default;
};

}  // namespace embedx
