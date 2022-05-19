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

#include <memory>  // std::unique_ptr
#include <vector>

#include "src/deep/client/deep_client.h"
#include "src/deep/deep_config.h"
#include "src/sampler/sampling.h"

namespace embedx {

class LocalDeepClientImplTest : public ::testing::Test {
 protected:
  DeepConfig config_;
  std::unique_ptr<DeepClient> deep_client_;

 protected:
  const std::string FREQ_FILE = "testdata/user_item_freq";
  const std::string NODE_CONFIG = "testdata/user_item_config";
  const std::string ITEM_FEATURE_FILE = "testdata/context";
  const std::string INST_FILE = "testdata/inst";
  const int THREAD_NUM = 3;
  const int NUMBER_TEST = 10;

 protected:
  void SetUp() override {
    config_.set_node_config(NODE_CONFIG);
    config_.set_freq_file(FREQ_FILE);
    config_.set_item_feature(ITEM_FEATURE_FILE);
    config_.set_inst_file(INST_FILE);

    config_.set_negative_sampler_type((int)SamplingEnum::UNIFORM);

    config_.set_thread_num(THREAD_NUM);

    deep_client_ = NewDeepClient(config_, DeepClientEnum::LOCAL);
    EXPECT_TRUE(deep_client_ != nullptr);
  }
};

TEST_F(LocalDeepClientImplTest, SharedSampleNegative) {
  int count = 10;
  vec_int_t nodes = {0, 416653778443095};
  vec_int_t excluded_nodes = {1, 470770042482842};
  std::vector<vec_int_t> sampled_nodes_list;
  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(deep_client_->SharedSampleNegative(count, nodes, excluded_nodes,
                                                   &sampled_nodes_list));
    EXPECT_EQ(sampled_nodes_list.size(), 2u);
    EXPECT_EQ(count, (int)sampled_nodes_list[0].size());
    EXPECT_EQ(count, (int)sampled_nodes_list[1].size());

    // exclude
    EXPECT_TRUE(std::find(sampled_nodes_list[0].begin(),
                          sampled_nodes_list[0].end(),
                          1) == sampled_nodes_list[0].end());
    EXPECT_TRUE(std::find(sampled_nodes_list[0].begin(),
                          sampled_nodes_list[0].end(),
                          1) == sampled_nodes_list[0].end());
  }
}

TEST_F(LocalDeepClientImplTest, LookupItemFeature) {
  vec_int_t failed_items = {13};
  vec_int_t items = {2, 5, 8};
  std::vector<vec_pair_t> item_feats;
  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_FALSE(deep_client_->LookupItemFeature(failed_items, &item_feats));
    EXPECT_TRUE(deep_client_->LookupItemFeature(items, &item_feats));
    EXPECT_EQ(item_feats.size(), 3u);
    EXPECT_EQ(item_feats[0].size(), 3u);
    EXPECT_EQ(item_feats[1].size(), 3u);
    EXPECT_EQ(item_feats[2].size(), 3u);
  }
}

TEST_F(LocalDeepClientImplTest, SampleInstance) {
  int count = 16;  // all test instances
  vec_int_t insts;
  std::vector<vecl_t> vec_labels_list;
  for (int i = 0; i < NUMBER_TEST; ++i) {
    EXPECT_TRUE(deep_client_->SampleInstance(count, &insts, &vec_labels_list));
    EXPECT_EQ(insts.size(), 16u);
    EXPECT_EQ(vec_labels_list.size(), 16u);

    for (int j = 0; j < (int)insts.size(); ++j) {
      if (insts[j] == 10) {
        EXPECT_EQ(vec_labels_list[j][0], 1);
        EXPECT_EQ(vec_labels_list[j][1], 0);
        EXPECT_EQ(vec_labels_list[j][2], 1);
      }
    }
  }
}

}  // namespace embedx
