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

#include <algorithm>  // std::find_if
#include <memory>     // std::unique_ptr

#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {

class LocalGraphClientImplTest : public ::testing::Test {
 protected:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";

  const std::string NODE_FEATURE = "testdata/node_feature";
  const std::string NEIGHBOR_FEATURE = "testdata/neigh_feature";

  const int THREAD_NUM = 3;
  const uint64_t ESTIMATED_SIZE = 1000000;
  const int NUMBER_TEST = 10;

 protected:
  void SetUp() override {
    config_.set_node_graph(CONTEXT);
    config_.set_node_feature(NODE_FEATURE);
    config_.set_neighbor_feature(NEIGHBOR_FEATURE);

    config_.set_thread_num(THREAD_NUM);

    config_.set_negative_sampler_type(0);
    config_.set_neighbor_sampler_type(0);
    config_.set_random_walker_type(0);

    graph_client_ = NewGraphClient(config_, GraphClientEnum::LOCAL);
    DXCHECK(graph_client_ != nullptr);
  }
};

TEST_F(LocalGraphClientImplTest, SharedSampleNegative) {
  int count = 10;
  vec_int_t nodes = {0, 9};
  vec_int_t excluded_nodes = {1, 10};
  std::vector<vec_int_t> sampled_nodes_list;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->SharedSampleNegative(
        count, nodes, excluded_nodes, &sampled_nodes_list));
    EXPECT_EQ(sampled_nodes_list.size(), 1u);
    EXPECT_EQ(count, (int)sampled_nodes_list[0].size());

    // exclude
    EXPECT_TRUE(std::find(sampled_nodes_list[0].begin(),
                          sampled_nodes_list[0].end(),
                          1) == sampled_nodes_list[0].end());
    EXPECT_TRUE(std::find(sampled_nodes_list[0].begin(),
                          sampled_nodes_list[0].end(),
                          1) == sampled_nodes_list[0].end());
  }
}

TEST_F(LocalGraphClientImplTest, IndepSampleNegative) {
  int count = 10;
  vec_int_t nodes = {0, 9};
  vec_int_t excluded_nodes = {1, 10};
  std::vector<vec_int_t> sampled_nodes_list;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->IndepSampleNegative(count, nodes, excluded_nodes,
                                                   &sampled_nodes_list));
    EXPECT_EQ(sampled_nodes_list.size(), 2u);

    // exclude
    for (size_t i = 0; i < nodes.size(); ++i) {
      EXPECT_TRUE(std::find(sampled_nodes_list[i].begin(),
                            sampled_nodes_list[i].end(),
                            1) == sampled_nodes_list[i].end());
      EXPECT_TRUE(std::find(sampled_nodes_list[i].begin(),
                            sampled_nodes_list[i].end(),
                            10) == sampled_nodes_list[i].end());
    }
  }
}

TEST_F(LocalGraphClientImplTest, RandomSampleNeighbor) {
  int count = 3;
  vec_int_t nodes = {0, 9};
  std::vector<vec_int_t> neighbor_nodes_list;

  vec_int_t candidates_0 = {12, 11, 10};
  vec_int_t candidates_9 = {8, 7, 6};

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->RandomSampleNeighbor(count, nodes,
                                                    &neighbor_nodes_list));
    EXPECT_EQ(nodes.size(), neighbor_nodes_list.size());

    EXPECT_TRUE(candidates_0.size() == neighbor_nodes_list[0].size());
    for (auto node : neighbor_nodes_list[0]) {
      EXPECT_TRUE(std::find(candidates_0.begin(), candidates_0.end(), node) !=
                  candidates_0.end());
    }

    EXPECT_TRUE(candidates_9.size() == neighbor_nodes_list[1].size());
    for (auto node : neighbor_nodes_list[1]) {
      EXPECT_TRUE(std::find(candidates_9.begin(), candidates_9.end(), node) !=
                  candidates_9.end());
    }
  }
}

TEST_F(LocalGraphClientImplTest, StaticTraverse) {
  vec_int_t cur_nodes = {0, 9};
  std::vector<int> walk_lens = {3, 3};
  WalkerInfo walker_info;
  std::vector<vec_int_t> seqs;
  std::vector<vec_pair_t> contexts;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    graph_client_->StaticTraverse(cur_nodes, walk_lens, walker_info, &seqs);
    EXPECT_EQ(cur_nodes.size(), seqs.size());

    vec_int_t pre_nodes = {0, 9};
    for (size_t i = 0; i < seqs.size(); ++i) {
      for (size_t j = 0; j < seqs[i].size(); ++j) {
        EXPECT_TRUE(graph_client_->LookupContext(pre_nodes, &contexts));
        auto seq_node = seqs[i][j];
        // in context
        auto it = std::find_if(contexts[i].begin(), contexts[i].end(),
                               [seq_node](const pair_t& entry) {
                                 return entry.first == seq_node;
                               });
        EXPECT_TRUE(it != contexts[i].end());
        pre_nodes[i] = seqs[i][j];
      }
    }
  }
}

TEST_F(LocalGraphClientImplTest, LookupFeature) {
  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> node_feats;
  std::vector<vec_pair_t> neigh_feats;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->LookupFeature(nodes, &node_feats, &neigh_feats));
    // node feature list
    EXPECT_EQ(node_feats[0].size(), 2u);
    EXPECT_EQ(node_feats[1].size(), 2u);
    // node(12) has no features, insert an empty feature
    EXPECT_EQ(node_feats[2].size(), 1u);
    // node(13) does not exist in graph, insert an empty feature
    EXPECT_EQ(node_feats[3].size(), 1u);

    // neighbor feature list
    EXPECT_EQ(neigh_feats[0].size(), 2u);
    EXPECT_EQ(neigh_feats[1].size(), 2u);
    // node(12) has no features, insert an empty feature
    EXPECT_EQ(neigh_feats[2].size(), 1u);
    // node(13) does not exist in graph, insert an empty feature
    EXPECT_EQ(neigh_feats[3].size(), 1u);
  }
}

TEST_F(LocalGraphClientImplTest, LookupNodeFeature) {
  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> node_feats;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->LookupNodeFeature(nodes, &node_feats));
    // node feature list
    EXPECT_EQ(node_feats[0].size(), 2u);
    EXPECT_EQ(node_feats[1].size(), 2u);
    // node(12) has no features, insert an empty feature
    EXPECT_EQ(node_feats[2].size(), 1u);
    // node(13) does not exist in graph, insert an empty feature
    EXPECT_EQ(node_feats[3].size(), 1u);
  }
}

TEST_F(LocalGraphClientImplTest, LookupNeighborFeature) {
  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> neigh_feats;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->LookupNeighborFeature(nodes, &neigh_feats));
    EXPECT_EQ(neigh_feats[0].size(), 2u);
    EXPECT_EQ(neigh_feats[1].size(), 2u);
    // node(12) has no features, insert an empty feature
    EXPECT_EQ(neigh_feats[2].size(), 1u);
    // node(13) does not exist in graph, insert an empty feature
    EXPECT_EQ(neigh_feats[3].size(), 1u);
  }
}

TEST_F(LocalGraphClientImplTest, LookupContext) {
  vec_int_t nodes = {0, 1, 2};
  std::vector<vec_pair_t> contexts;

  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(graph_client_->LookupContext(nodes, &contexts));
    EXPECT_EQ(nodes.size(), contexts.size());

    EXPECT_EQ(contexts[0].size(), 3u);
    EXPECT_EQ(contexts[1].size(), 3u);
    EXPECT_EQ(contexts[2].size(), 3u);
  }
}

}  // namespace embedx
