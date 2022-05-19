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

#include "src/graph/cache/cache_storage.h"

namespace embedx {
namespace {

const vec_pair_t* FindCache(const adj_list_t& cache_info, int_t node) {
  auto it = cache_info.find(node);
  if (it != cache_info.end()) {
    return &it->second;
  } else {
    return nullptr;
  }
}

}  // namespace

const vec_pair_t* CacheStorage::FindContext(int_t node) const {
  return FindCache(context_map_, node);
}

const vec_pair_t* CacheStorage::FindNodeFeature(int_t node) const {
  return FindCache(node_feat_map_, node);
}

const vec_pair_t* CacheStorage::FindFeature(int_t node) const {
  return FindCache(feat_map_, node);
}

std::unique_ptr<CacheStorage> NewCacheStorage() {
  std::unique_ptr<CacheStorage> cache_storage;
  cache_storage.reset(new CacheStorage());
  return cache_storage;
}

}  // namespace embedx
