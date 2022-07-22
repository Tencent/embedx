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

#include "src/graph/cache/cache_node_builder.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr

#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {

class CacheNodeBuilderTest : public ::testing::Test {
 protected:
  std::unique_ptr<InMemoryGraph> graph_;
  std::unique_ptr<CacheNodeBuilder> cache_node_builder_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";

 protected:
  void SetUp() override {
    config_.set_node_graph(CONTEXT);
    graph_ = InMemoryGraph::Create(config_);
    DXCHECK(graph_ != nullptr);
  }
};

TEST_F(CacheNodeBuilderTest, Build) {
  cache_node_builder_ = CacheNodeBuilder::Create(graph_.get(), 0, -1.0, 2);
  EXPECT_TRUE(cache_node_builder_ == nullptr);
  cache_node_builder_ = CacheNodeBuilder::Create(graph_.get(), 4, 1.0, 2);
  EXPECT_TRUE(cache_node_builder_ == nullptr);
}

TEST_F(CacheNodeBuilderTest, RandomCache) {
  cache_node_builder_ = CacheNodeBuilder::Create(graph_.get(), 0, 1.0, 2);
  EXPECT_TRUE(cache_node_builder_ != nullptr);
  EXPECT_EQ(cache_node_builder_->nodes()->size(), 13u);
}

TEST_F(CacheNodeBuilderTest, DegreeCache) {
  cache_node_builder_ = CacheNodeBuilder::Create(graph_.get(), 1, 0.5, 2);
  EXPECT_TRUE(cache_node_builder_ != nullptr);
  EXPECT_EQ(cache_node_builder_->nodes()->size(), 6u);
}

TEST_F(CacheNodeBuilderTest, ImportanceCache) {
  cache_node_builder_ = CacheNodeBuilder::Create(graph_.get(), 2, 0.5, 2);
  EXPECT_TRUE(cache_node_builder_ != nullptr);
  EXPECT_EQ(cache_node_builder_->nodes()->size(), 13u);
}

}  // namespace embedx
