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

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>

#include "src/io/loader/loader.h"
#include "src/io/storage/adjacency.h"

namespace embedx {

class FeatureLoaderTest : public ::testing::Test {
 protected:
  std::unique_ptr<Loader> loader_;

 protected:
  const std::string NODE_FEATURE = "testdata/node_feature";
  const int THREAD_NUM = 3;
  const uint64_t ESTIMATED_SIZE = 10;

  int shard_num_ = 1;
  int shard_id_ = 0;

 protected:
  void TestLocal(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(NODE_FEATURE, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 12u);
    EXPECT_TRUE(!store->Empty());

    for (auto node : store->Keys()) {
      EXPECT_EQ(store->GetInDegree(node), 0);
      EXPECT_EQ(store->GetOutDegree(node), 0);
    }
  }

  void TestRemoteShard0(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(NODE_FEATURE, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 6u);
    EXPECT_TRUE(!store->Empty());

    for (size_t i = 0; i < store->Keys().size(); ++i) {
      auto node = store->Keys()[i];
      auto* feature = store->FindNeighbor(node);
      EXPECT_EQ(node % 2, 0u);
      EXPECT_EQ(feature->size(), 2u);
      EXPECT_EQ(store->GetInDegree(node), 0);
      EXPECT_EQ(store->GetOutDegree(node), 0);
    }
  }

  void TestRemoteShard1(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(NODE_FEATURE, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 6u);
    EXPECT_TRUE(!store->Empty());

    for (size_t i = 0; i < store->Keys().size(); ++i) {
      auto node = store->Keys()[i];
      auto* feature = store->FindNeighbor(node);
      EXPECT_EQ(node % 2, 1u);
      EXPECT_EQ(feature->size(), 2u);
      EXPECT_EQ(store->GetInDegree(node), 0);
      EXPECT_EQ(store->GetOutDegree(node), 0);
    }
  }
};

TEST_F(FeatureLoaderTest, Load_Local) {
  // AdjList
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestLocal(loader_.get());

  // AdjMatrix
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestLocal(loader_.get());
}

TEST_F(FeatureLoaderTest, Load_Remote_AdjList) {
  // shard 0
  shard_num_ = 2;
  shard_id_ = 0;
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestRemoteShard0(loader_.get());

  // shard 1
  shard_id_ = 1;
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestRemoteShard1(loader_.get());
}

TEST_F(FeatureLoaderTest, Load_Remote_AdjMatrix) {
  // shard 0
  shard_num_ = 2;
  shard_id_ = 0;
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestRemoteShard0(loader_.get());

  // shard 1
  shard_id_ = 1;
  loader_ =
      NewFeatureLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestRemoteShard1(loader_.get());
}

}  // namespace embedx
