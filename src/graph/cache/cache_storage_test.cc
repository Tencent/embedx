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

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"

namespace embedx {

class CacheStorageTest : public ::testing::Test {
 protected:
  std::unique_ptr<CacheStorage> cache_storage_;

 protected:
  void SetUp() override { cache_storage_ = NewCacheStorage(); }
};

TEST_F(CacheStorageTest, FindContext) {
  auto* context_map = cache_storage_->context_map();
  vec_pair_t context = {{1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}};
  context_map->emplace(100, context);

  EXPECT_EQ(cache_storage_->FindContext(101), nullptr);
  EXPECT_EQ(cache_storage_->FindContext(100)->size(), 5u);
}

TEST_F(CacheStorageTest, FindNodeFeature) {
  auto* node_feat_map = cache_storage_->node_feat_map();
  vec_pair_t node_feat = {{1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}};
  node_feat_map->emplace(100, node_feat);

  EXPECT_EQ(cache_storage_->FindNodeFeature(101), nullptr);
  EXPECT_EQ(cache_storage_->FindNodeFeature(100)->size(), 5u);
}

TEST_F(CacheStorageTest, FindFeature) {
  auto* feat_map = cache_storage_->feat_map();
  vec_pair_t feat = {{1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}};
  feat_map->emplace(100, feat);

  EXPECT_EQ(cache_storage_->FindFeature(101), nullptr);
  EXPECT_EQ(cache_storage_->FindFeature(100)->size(), 5u);
}

}  // namespace embedx
