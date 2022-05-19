// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/model/data_flow/neighbor_aggregation_flow.h"

#include <deepx_core/graph/tensor_map.h>  // Instance
#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <unordered_map>
#include <vector>

#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/model/instance_reader_util.h"

namespace embedx {
namespace {

void CreateIndexings(const vec_set_t& level_nodes,
                     std::vector<std::unordered_map<int_t, int>>* indexings) {
  indexings->clear();
  indexings->resize(level_nodes.size());
  int k = 0;
  for (size_t i = 0; i < level_nodes.size(); ++i) {
    for (auto node : level_nodes[i]) {
      (*indexings)[i].emplace(node, k);
      k += 1;
    }
  }
}

}  // namespace

class NeighborAggregationFlowTest : public ::testing::Test {
 protected:
  std::unique_ptr<NeighborAggregationFlow> flow_;
  std::unique_ptr<GraphClient> client_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const std::string NODE_FEATURE = "testdata/node_feature";
  const std::string NEIGHBOR_FEATURE = "testdata/neigh_feature";
  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    config_.set_node_graph(CONTEXT);
    config_.set_node_feature(NODE_FEATURE);
    config_.set_neighbor_feature(NEIGHBOR_FEATURE);
    config_.set_thread_num(THREAD_NUM);

    client_ = NewGraphClient(config_, GraphClientEnum::LOCAL);
    DXCHECK(client_ != nullptr);
    flow_.reset(new NeighborAggregationFlow(client_.get()));
  }
};

TEST_F(NeighborAggregationFlowTest, CreateIndexings) {
  vec_set_t level_nodes;
  vec_map_neigh_t level_neighs;
  flow_->SampleSubGraph({3, 4, 5}, {2, 3, 4}, &level_nodes, &level_neighs);

  std::vector<Indexing> indexings;
  inst_util::CreateIndexings(level_nodes, &indexings);

  std::vector<std::unordered_map<int_t, int>> expected_indexings;
  CreateIndexings(level_nodes, &expected_indexings);

  for (size_t i = 0; i < indexings.size(); ++i) {
    for (auto& entry : expected_indexings[i]) {
      EXPECT_EQ(indexings[i].Get(entry.first),
                expected_indexings[i].at(entry.first));
    }
  }
}

TEST_F(NeighborAggregationFlowTest, FillNodeFeature) {
  vec_set_t level_nodes;
  vec_map_neigh_t level_neighs;
  flow_->SampleSubGraph({3}, {3}, &level_nodes, &level_neighs);

  deepx_core::Instance inst;
  std::string NODE_FEATURE_NAME = "TEST_NODE_FEATURE_NAME";

  // mask_prob equals to 0
  flow_->set_feature_mask_prob(0);
  flow_->FillLevelNodeFeature(&inst, NODE_FEATURE_NAME, level_nodes);
  auto* node_feat_ptr = &inst.get_or_insert<csr_t>(NODE_FEATURE_NAME);
  EXPECT_TRUE(!node_feat_ptr->empty());
  EXPECT_EQ(node_feat_ptr->row(), 4);
  EXPECT_EQ(node_feat_ptr->col_size(), (int_t)8);

  // mask_prob equals to 0.9
  flow_->set_feature_mask_prob(0.9);
  flow_->FillLevelNodeFeature(&inst, NODE_FEATURE_NAME, level_nodes);
  node_feat_ptr = &inst.get_or_insert<csr_t>(NODE_FEATURE_NAME);
  EXPECT_LE(node_feat_ptr->row(), 4);
  EXPECT_GE(node_feat_ptr->col_size(), (int_t)0);
  EXPECT_NE(node_feat_ptr->col_size(), (int_t)8);
  EXPECT_LT(node_feat_ptr->col_size(), (int_t)8);

  // mask_prob equals to 1.0
  flow_->set_feature_mask_prob(1.0);
  flow_->FillLevelNodeFeature(&inst, NODE_FEATURE_NAME, level_nodes);
  node_feat_ptr = &inst.get_or_insert<csr_t>(NODE_FEATURE_NAME);
  EXPECT_TRUE(node_feat_ptr->empty());
}

TEST_F(NeighborAggregationFlowTest, FillNeighFeature) {
  vec_set_t level_nodes;
  vec_map_neigh_t level_neighs;
  flow_->SampleSubGraph({3}, {3}, &level_nodes, &level_neighs);

  deepx_core::Instance inst;
  std::string NEIGH_FEATURE_NAME = "TEST_NEIGH_FEATURE_NAME";

  // mask_prob equals to 0
  flow_->set_feature_mask_prob(0);
  flow_->FillLevelNeighFeature(&inst, NEIGH_FEATURE_NAME, level_nodes);
  auto* neigh_feat_ptr = &inst.get_or_insert<csr_t>(NEIGH_FEATURE_NAME);
  EXPECT_TRUE(!neigh_feat_ptr->empty());
  EXPECT_EQ(neigh_feat_ptr->row(), 4);
  EXPECT_EQ(neigh_feat_ptr->col_size(), (int_t)8);

  // mask_prob equals to 0.9
  flow_->set_feature_mask_prob(0.9);
  flow_->FillLevelNeighFeature(&inst, NEIGH_FEATURE_NAME, level_nodes);
  neigh_feat_ptr = &inst.get_or_insert<csr_t>(NEIGH_FEATURE_NAME);
  EXPECT_LE(neigh_feat_ptr->row(), 4);
  EXPECT_GE(neigh_feat_ptr->col_size(), (int_t)0);
  EXPECT_NE(neigh_feat_ptr->col_size(), (int_t)8);
  EXPECT_LT(neigh_feat_ptr->col_size(), (int_t)8);

  // mask_prob equals to 1.0
  flow_->set_feature_mask_prob(1.0);
  flow_->FillLevelNeighFeature(&inst, NEIGH_FEATURE_NAME, level_nodes);
  neigh_feat_ptr = &inst.get_or_insert<csr_t>(NEIGH_FEATURE_NAME);
  EXPECT_TRUE(neigh_feat_ptr->empty());
}

TEST_F(NeighborAggregationFlowTest, FillSelfAndNeighGraphBlock) {
  vec_set_t level_nodes;
  vec_map_neigh_t level_neighs;
  flow_->SampleSubGraph({3}, {3}, &level_nodes, &level_neighs);

  std::vector<Indexing> indexings;
  inst_util::CreateIndexings(level_nodes, &indexings);

  deepx_core::Instance inst;
  std::string SELF_BLOCK_NAME = "TEST_SELF_BLOCK_NAME";
  std::string NEIGH_BLOCK_NAME = "TEST_NEIGH_BLOCK_NAME";

  // drop_prob equals to 0
  flow_->set_edge_drop_prob(0);
  flow_->FillSelfAndNeighGraphBlock(&inst, SELF_BLOCK_NAME, NEIGH_BLOCK_NAME,
                                    level_nodes, level_neighs, indexings,
                                    false);
  for (size_t i = 0; i < level_nodes.size() - 1; ++i) {
    auto* self_block =
        &inst.get_or_insert<csr_t>(SELF_BLOCK_NAME + std::to_string(i));
    EXPECT_EQ(self_block->row(), 1);
    EXPECT_EQ(self_block->col_size(), (int_t)1);

    auto* neigh_block =
        &inst.get_or_insert<csr_t>(NEIGH_BLOCK_NAME + std::to_string(i));
    EXPECT_EQ(neigh_block->row(), 1);
    EXPECT_EQ(neigh_block->col_size(), (int_t)3);
  }

  // drop_prob equals to 0.9
  flow_->set_edge_drop_prob(0.9);
  flow_->FillSelfAndNeighGraphBlock(&inst, SELF_BLOCK_NAME, NEIGH_BLOCK_NAME,
                                    level_nodes, level_neighs, indexings,
                                    false);
  for (size_t i = 0; i < level_nodes.size() - 1; ++i) {
    auto* self_block =
        &inst.get_or_insert<csr_t>(SELF_BLOCK_NAME + std::to_string(i));
    EXPECT_EQ(self_block->row(), 1);
    EXPECT_EQ(self_block->col_size(), (int_t)1);

    auto* neigh_block =
        &inst.get_or_insert<csr_t>(NEIGH_BLOCK_NAME + std::to_string(i));
    EXPECT_LE(neigh_block->row(), 1);
    EXPECT_NE(neigh_block->col_size(), (int_t)3);
    EXPECT_LT(neigh_block->col_size(), (int_t)3);
  }

  // drop_prob equals to 1.0
  flow_->set_edge_drop_prob(1.0);
  flow_->FillSelfAndNeighGraphBlock(&inst, SELF_BLOCK_NAME, NEIGH_BLOCK_NAME,
                                    level_nodes, level_neighs, indexings,
                                    false);
  for (size_t i = 0; i < level_nodes.size() - 1; ++i) {
    auto* self_block =
        &inst.get_or_insert<csr_t>(SELF_BLOCK_NAME + std::to_string(i));
    EXPECT_EQ(self_block->row(), 1);
    EXPECT_EQ(self_block->col_size(), (int_t)1);

    auto* neigh_block =
        &inst.get_or_insert<csr_t>(NEIGH_BLOCK_NAME + std::to_string(i));
    EXPECT_EQ(neigh_block->row(), 1);
    EXPECT_EQ(neigh_block->col_size(), (int_t)0);
  }
}

}  // namespace embedx
