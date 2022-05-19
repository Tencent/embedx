// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanqing Guo (yuanqingsunny1180@gmail.com)
//

#pragma once
#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"

namespace embedx {

class CacheStorage {
 private:
  adj_list_t context_map_;
  adj_list_t node_feat_map_;
  adj_list_t feat_map_;

 public:
  CacheStorage() = default;
  ~CacheStorage() = default;

 public:
  const vec_pair_t* FindContext(int_t node) const;
  const vec_pair_t* FindNodeFeature(int_t node) const;
  const vec_pair_t* FindFeature(int_t node) const;

 public:
  const adj_list_t* context_map() const noexcept { return &context_map_; }
  const adj_list_t* node_feat_map() const noexcept { return &node_feat_map_; }
  const adj_list_t* feat_map() const noexcept { return &feat_map_; }

  adj_list_t* context_map() noexcept { return &context_map_; }
  adj_list_t* node_feat_map() noexcept { return &node_feat_map_; }
  adj_list_t* feat_map() noexcept { return &feat_map_; }
};

std::unique_ptr<CacheStorage> NewCacheStorage();

}  // namespace embedx
