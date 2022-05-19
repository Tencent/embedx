// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//

#include <gtest/gtest.h>

#include <memory>   // std::unique_ptr
#include <numeric>  // std::accumulate
#include <string>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class GraphSamplerSourceTest : public ::testing::Test {
 protected:
  GraphConfig config_;
  std::unique_ptr<InMemoryGraph> graph_;
  std::unique_ptr<SamplerSource> sampler_source_;

 protected:
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";
  const std::string NODE_FEATURE = "testdata/node_feature";
  const std::string NEIGHBOR_FEATURE = "testdata/neigh_feature";

  const int SHARD_NUM = 1;
  const int SHARD_ID = 0;

  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    config_.set_node_graph(USER_ITEM_CONTEXT);
    config_.set_node_config(USER_ITEM_CONFIG);
    config_.set_node_feature(NODE_FEATURE);
    config_.set_neighbor_feature(NEIGHBOR_FEATURE);

    config_.set_shard_num(SHARD_NUM);
    config_.set_shard_id(SHARD_ID);

    config_.set_thread_num(THREAD_NUM);

    graph_ = InMemoryGraph::Create(config_);
    EXPECT_TRUE(graph_ != nullptr);

    sampler_source_ = NewGraphSamplerSource(graph_.get());
    EXPECT_TRUE(sampler_source_ != nullptr);
  }
};

TEST_F(GraphSamplerSourceTest, All) {
  EXPECT_EQ(sampler_source_->ns_size(), 2);

  EXPECT_EQ(sampler_source_->nodes_list().size(), 2u);
  EXPECT_EQ(sampler_source_->nodes_list()[0].size(), 13u);
  EXPECT_EQ(sampler_source_->nodes_list()[1].size(), 13u);

  EXPECT_EQ(sampler_source_->freqs_list().size(), 2u);
  EXPECT_EQ(sampler_source_->nodes_list()[0].size(),
            sampler_source_->freqs_list()[0].size());
  EXPECT_EQ(sampler_source_->nodes_list()[1].size(),
            sampler_source_->freqs_list()[1].size());
  float_t sum0 = std::accumulate(sampler_source_->freqs_list()[0].begin(),
                                 sampler_source_->freqs_list()[0].end(), 0.0);
  EXPECT_EQ(sum0, 52.0);
  float_t sum1 = std::accumulate(sampler_source_->freqs_list()[1].begin(),
                                 sampler_source_->freqs_list()[1].end(), 0.0);
  EXPECT_EQ(sum1, 52.0);
}

}  // namespace embedx
