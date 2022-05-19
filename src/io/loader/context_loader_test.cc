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

class ContextLoaderTest : public ::testing::Test {
 protected:
  std::unique_ptr<Loader> loader_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const int THREAD_NUM = 3;
  const uint64_t ESTIMATED_SIZE = 10;

  int shard_num_ = 1;
  int shard_id_ = 0;

 protected:
  void TestLocal(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(CONTEXT, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 13u);
    EXPECT_TRUE(!store->Empty());
    for (auto node : store->Keys()) {
      EXPECT_EQ(store->GetInDegree(node), 3);
      EXPECT_EQ(store->GetOutDegree(node), 3);
    }
  }

  void TestRemoteShard0(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(CONTEXT, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 7u);
    EXPECT_TRUE(!store->Empty());

    // in context storage, for example: 0, 2
    EXPECT_EQ(store->GetInDegree(0), 1);
    EXPECT_EQ(store->GetOutDegree(0), 3);
    EXPECT_EQ(store->GetInDegree(2), 1);
    EXPECT_EQ(store->GetOutDegree(2), 3);

    // out context storage, for example: 1, 3
    EXPECT_EQ(store->GetInDegree(1), 2);
    EXPECT_EQ(store->GetOutDegree(1), 0);
    EXPECT_EQ(store->GetInDegree(3), 2);
    EXPECT_EQ(store->GetOutDegree(3), 0);
  }

  void TestRemoteShard1(Loader* loader) {
    loader->Clear();
    loader->Reserve(ESTIMATED_SIZE);
    EXPECT_TRUE(loader->Load(CONTEXT, THREAD_NUM));

    const auto* store = loader->storage();
    EXPECT_EQ(store->Size(), 6u);
    EXPECT_TRUE(!store->Empty());

    // in context storage, for example: 1, 3
    EXPECT_EQ(store->GetInDegree(1), 1);
    EXPECT_EQ(store->GetOutDegree(1), 3);
    EXPECT_EQ(store->GetInDegree(3), 1);
    EXPECT_EQ(store->GetOutDegree(3), 3);

    // out context storage, for example: 0, 2
    EXPECT_EQ(store->GetInDegree(0), 2);
    EXPECT_EQ(store->GetOutDegree(0), 0);
    EXPECT_EQ(store->GetInDegree(2), 2);
    EXPECT_EQ(store->GetOutDegree(2), 0);
  }
};

TEST_F(ContextLoaderTest, Load_Local_AdjList) {
  shard_num_ = 1;
  shard_id_ = 0;

  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestLocal(loader_.get());

  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestLocal(loader_.get());
}

TEST_F(ContextLoaderTest, Load_Remote_AdjList) {
  // shard 0
  shard_num_ = 2;
  shard_id_ = 0;
  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestRemoteShard0(loader_.get());

  // shard 1
  shard_id_ = 1;
  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_LIST);
  TestRemoteShard1(loader_.get());
}

TEST_F(ContextLoaderTest, Load_Remote__AdjMatrix) {
  // shard 0
  shard_num_ = 2;
  shard_id_ = 0;
  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestRemoteShard0(loader_.get());

  // shard 1
  shard_id_ = 1;
  loader_ =
      NewContextLoader(shard_num_, shard_id_, (int)AdjacencyEnum::ADJ_MATRIX);
  TestRemoteShard1(loader_.get());
}

}  // namespace embedx
