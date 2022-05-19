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

#include "src/io/indexing.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

namespace embedx {

void Indexing::Reserve(uint64_t estimated_size) {
  index_map_.reserve(estimated_size);
}

void Indexing::Clear() noexcept { index_map_.clear(); }

void Indexing::Add(int_t node) {
  int index = (int)index_map_.size();
  index_map_.emplace(node, index);
}

void Indexing::Emplace(int_t k, int v) { index_map_.emplace(k, v); }

int Indexing::Get(int_t node) const {
  auto it = index_map_.find(node);
  if (it == index_map_.end()) {
    DXERROR("Couldn't find Node: %" PRIu64 " in index the table.", node);
    return -1;
  } else {
    return it->second;
  }
}

bool Indexing::Find(int_t node) const {
  return index_map_.find(node) != index_map_.end();
}

size_t Indexing::Size() const noexcept { return index_map_.size(); }

}  // namespace embedx
