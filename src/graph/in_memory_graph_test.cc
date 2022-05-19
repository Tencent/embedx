// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chunchen Su (chunchen.scut@gmail.com)
//

#include "src/graph/in_memory_graph.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/io/storage/adjacency.h"

namespace embedx {

class InMemoryGraphTest : public ::testing::Test {
 protected:
  std::unique_ptr<InMemoryGraph> graph_;
  GraphConfig config_;

 private:
  const std::string CONTEXT = "testdata/context";
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";
  const std::string NODE_FEATURE = "testdata/node_feature";
  const std::string NEIGHBOR_FEATURE = "testdata/neigh_feature";

  const int SHARD_NUM = 1;
  const int SHARD_ID = 0;

  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    config_.set_node_graph(CONTEXT);
    config_.set_node_feature(NODE_FEATURE);
    config_.set_neighbor_feature(NEIGHBOR_FEATURE);

    config_.set_shard_num(SHARD_NUM);
    config_.set_shard_id(SHARD_ID);

    config_.set_thread_num(THREAD_NUM);
  }

  void TestOneNameSpace() {
    graph_ = InMemoryGraph::Create(config_);
    EXPECT_TRUE(graph_ != nullptr);

    // topology
    EXPECT_EQ(graph_->uniq_nodes_list()[0].size(), 13u);
    EXPECT_EQ(graph_->total_freqs()[0], (int_t)52);

    // size
    EXPECT_EQ(graph_->node_size(), 13u);
    EXPECT_EQ(graph_->node_feature_size(), 12u);
    EXPECT_EQ(graph_->neigh_feature_size(), 12u);

    // empty
    EXPECT_TRUE(!graph_->node_empty());
    EXPECT_TRUE(!graph_->node_feature_empty());
    EXPECT_TRUE(!graph_->neigh_feature_empty());

    // degree
    for (auto node : graph_->node_keys()) {
      EXPECT_EQ(graph_->GetInDegree(node), 3);
      EXPECT_EQ(graph_->GetOutDegree(node), 3);
    }
  }

  void TestTwoNameSpace() {
    config_.set_node_graph(USER_ITEM_CONTEXT);
    config_.set_node_config(USER_ITEM_CONFIG);
    graph_ = InMemoryGraph::Create(config_);
    EXPECT_TRUE(graph_ != nullptr);

    // user topology
    EXPECT_EQ(graph_->uniq_nodes_list()[0].size(), 13u);
    EXPECT_EQ(graph_->total_freqs()[0], (int_t)52);

    // item topology
    EXPECT_EQ(graph_->uniq_nodes_list()[1].size(), 13u);
    EXPECT_EQ(graph_->total_freqs()[1], (int_t)52);

    // size
    EXPECT_EQ(graph_->node_size(), 26u);
    EXPECT_EQ(graph_->node_feature_size(), 12u);
    EXPECT_EQ(graph_->neigh_feature_size(), 12u);

    // empty
    EXPECT_TRUE(!graph_->node_empty());
    EXPECT_TRUE(!graph_->node_feature_empty());
    EXPECT_TRUE(!graph_->neigh_feature_empty());

    // degree
    for (auto node : graph_->node_keys()) {
      EXPECT_EQ(graph_->GetInDegree(node), 3);
      EXPECT_EQ(graph_->GetOutDegree(node), 3);
    }
  }

  void TestShard0() {
    config_.set_shard_num(2);
    config_.set_shard_id(0);

    graph_ = InMemoryGraph::Create(config_);
    EXPECT_TRUE(graph_ != nullptr);

    // topology
    EXPECT_EQ(graph_->uniq_nodes_list()[0].size(), 13u);
    EXPECT_EQ(graph_->total_freqs()[0], (int_t)28);
  }

  void TestShard1() {
    config_.set_shard_num(2);
    config_.set_shard_id(1);

    graph_ = InMemoryGraph::Create(config_);
    EXPECT_TRUE(graph_ != nullptr);

    // topology
    EXPECT_EQ(graph_->uniq_nodes_list()[0].size(), 13u);
    EXPECT_EQ(graph_->total_freqs()[0], (int_t)24);
  }
};

TEST_F(InMemoryGraphTest, Build_OneNameSpace) {
  // AdjList
  config_.set_store_type((int)AdjacencyEnum::ADJ_LIST);
  TestOneNameSpace();

  // AdjMatrix
  config_.set_store_type((int)AdjacencyEnum::ADJ_MATRIX);
  TestOneNameSpace();
}

TEST_F(InMemoryGraphTest, Build_TwoNameSpace) {
  // AdjList
  config_.set_store_type((int)AdjacencyEnum::ADJ_LIST);
  TestTwoNameSpace();

  // AdjMatrix
  config_.set_store_type((int)AdjacencyEnum::ADJ_MATRIX);
  TestTwoNameSpace();
}

TEST_F(InMemoryGraphTest, Build_Shard0) {
  // AdjList
  config_.set_store_type((int)AdjacencyEnum::ADJ_LIST);
  TestShard0();

  // AdjMatrix
  config_.set_store_type((int)AdjacencyEnum::ADJ_MATRIX);
  TestShard0();
}

TEST_F(InMemoryGraphTest, Build_Shard1) {
  // AdjList
  config_.set_store_type((int)AdjacencyEnum::ADJ_LIST);
  TestShard1();

  // AdjMatrix
  config_.set_store_type((int)AdjacencyEnum::ADJ_MATRIX);
  TestShard1();
}

}  // namespace embedx
