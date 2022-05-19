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

#include "src/deep/deep_data.h"

#include <gtest/gtest.h>

#include <memory>   // std::unique_ptr
#include <numeric>  // std::accumulate
#include <string>

#include "src/deep/deep_config.h"
#include "src/deep/deep_data.h"

namespace embedx {

class DeepDataTest : public ::testing::Test {
 protected:
  DeepConfig config_;
  std::unique_ptr<DeepData> deep_data_;

 protected:
  const std::string FREQ_FILE = "testdata/user_item_freq";
  const std::string NODE_CONFIG = "testdata/user_item_config";
  const std::string ITEM_FEATURE_FILE = "testdata/context";
  const std::string INST_FILE = "testdata/inst";
};

TEST_F(DeepDataTest, FailOnMissingFreqFile) {
  DeepConfig EMPTY_CONFIG;
  deep_data_ = DeepData::Create(EMPTY_CONFIG);
  EXPECT_TRUE(deep_data_ == nullptr);

  config_.set_item_feature(ITEM_FEATURE_FILE);
  deep_data_ = DeepData::Create(config_);
  EXPECT_TRUE(deep_data_ != nullptr);

  config_.set_freq_file(FREQ_FILE);
  config_.set_node_config(NODE_CONFIG);
  deep_data_ = DeepData::Create(config_);
  EXPECT_TRUE(deep_data_ != nullptr);
}

TEST_F(DeepDataTest, LoadFreqFile) {
  config_.set_node_config(NODE_CONFIG);
  config_.set_freq_file(FREQ_FILE);
  deep_data_ = DeepData::Create(config_);
  EXPECT_TRUE(deep_data_ != nullptr);

  EXPECT_EQ(deep_data_->ns_size(), 2);

  EXPECT_EQ(deep_data_->id_name_map().size(), 2u);

  EXPECT_EQ(deep_data_->nodes_list().size(), 2u);
  EXPECT_EQ(deep_data_->nodes_list()[0].size(), 13u);
  EXPECT_EQ(deep_data_->nodes_list()[1].size(), 13u);

  EXPECT_EQ(deep_data_->freqs_list().size(), 2u);
  EXPECT_EQ(deep_data_->nodes_list()[0].size(),
            deep_data_->freqs_list()[0].size());
  EXPECT_EQ(deep_data_->nodes_list()[1].size(),
            deep_data_->freqs_list()[1].size());
  float_t sum0 = std::accumulate(deep_data_->freqs_list()[0].begin(),
                                 deep_data_->freqs_list()[0].end(), 0.0);
  EXPECT_EQ(sum0, 13.0);
  float_t sum1 = std::accumulate(deep_data_->freqs_list()[1].begin(),
                                 deep_data_->freqs_list()[1].end(), 0.0);
  EXPECT_EQ(sum1, 39.0);
}

TEST_F(DeepDataTest, LoadItemFeature) {
  config_.set_node_config(NODE_CONFIG);
  config_.set_freq_file(FREQ_FILE);
  config_.set_item_feature(ITEM_FEATURE_FILE);
  deep_data_ = DeepData::Create(config_);
  EXPECT_TRUE(deep_data_ != nullptr);

  const auto* item_feature = deep_data_->FindItemFeature(13);
  EXPECT_TRUE(item_feature == nullptr);

  item_feature = deep_data_->FindItemFeature(10);
  EXPECT_TRUE(item_feature != nullptr);
  EXPECT_EQ(item_feature->size(), 3u);
}

TEST_F(DeepDataTest, LoadInstFile) {
  config_.set_node_config(NODE_CONFIG);
  config_.set_freq_file(FREQ_FILE);
  config_.set_inst_file(INST_FILE);
  deep_data_ = DeepData::Create(config_);
  EXPECT_TRUE(deep_data_ != nullptr);

  EXPECT_EQ(deep_data_->insts().size(), 16u);
  EXPECT_EQ(deep_data_->vec_labels_list().size(), 16u);
  auto labels = deep_data_->vec_labels_list()[10];
  EXPECT_EQ(labels.size(), 3u);
  EXPECT_EQ(labels[0], 1);
  EXPECT_EQ(labels[1], 0);
  EXPECT_EQ(labels[2], 1);
}

}  // namespace embedx
