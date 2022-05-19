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

#include "src/graph/data_op/feature_lookuper_op/feature.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {
namespace graph_op {

class FeatureLookupTest : public ::testing::Test {
 protected:
  std::unique_ptr<InMemoryGraph> graph_;
  std::unique_ptr<Feature> feature_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const std::string NODE_FEATURE = "testdata/node_feature";
  const std::string NEIGHBOR_FEATURE = "testdata/neigh_feature";

 protected:
  void SetUp() override {
    config_.set_node_graph(CONTEXT);
    config_.set_node_feature(NODE_FEATURE);
    config_.set_neighbor_feature(NEIGHBOR_FEATURE);
  }
};

TEST_F(FeatureLookupTest, LookupNodeFeature) {
  graph_ = InMemoryGraph::Create(config_);
  EXPECT_TRUE(graph_ != nullptr);

  feature_ = NewFeature(graph_.get());
  EXPECT_TRUE(feature_ != nullptr);

  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> feats;

  EXPECT_TRUE(feature_->LookupNodeFeature(nodes, &feats));
  EXPECT_EQ(feats[0].size(), 2u);
  EXPECT_EQ(feats[1].size(), 2u);
  // node(12) has no features, insert an empty feature
  EXPECT_EQ(feats[2].size(), 1u);
  // node(13) does not exist in graph, insert an empty feature
  EXPECT_EQ(feats[3].size(), 1u);
}

TEST_F(FeatureLookupTest, LookupNeighborFeature) {
  graph_ = InMemoryGraph::Create(config_);
  EXPECT_TRUE(graph_ != nullptr);

  feature_ = NewFeature(graph_.get());
  EXPECT_TRUE(feature_ != nullptr);

  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> feats;

  EXPECT_TRUE(feature_->LookupNeighborFeature(nodes, &feats));
  EXPECT_EQ(feats[0].size(), 2u);
  EXPECT_EQ(feats[1].size(), 2u);
  // node(12) has no features, insert an empty feature
  EXPECT_EQ(feats[2].size(), 1u);
  // node(13) does not exist in graph, insert an empty feature
  EXPECT_EQ(feats[3].size(), 1u);
}

}  // namespace graph_op
}  // namespace embedx
