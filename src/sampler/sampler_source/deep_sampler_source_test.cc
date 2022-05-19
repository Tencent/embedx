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
#include "src/deep/deep_config.h"
#include "src/deep/deep_data.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class DeepSamplerSourceTest : public ::testing::Test {
 protected:
  DeepConfig config_;
  std::unique_ptr<DeepData> deep_data_;
  std::unique_ptr<SamplerSource> sampler_source_;

 protected:
  const std::string FREQ_FILE = "testdata/user_item_freq";
  const std::string NODE_CONFIG = "testdata/user_item_config";

 protected:
  void SetUp() override {
    config_.set_node_config(NODE_CONFIG);
    config_.set_freq_file(FREQ_FILE);
    deep_data_ = DeepData::Create(config_);
    EXPECT_TRUE(deep_data_ != nullptr);
    sampler_source_ = NewDeepSamplerSource(deep_data_.get());
    EXPECT_TRUE(sampler_source_ != nullptr);
  }
};

TEST_F(DeepSamplerSourceTest, Implemented) {
  EXPECT_EQ(sampler_source_->ns_size(), 2);
  EXPECT_EQ(sampler_source_->id_name_map().size(), 2u);
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
  EXPECT_EQ(sum0, 13.0);
  float_t sum1 = std::accumulate(sampler_source_->freqs_list()[1].begin(),
                                 sampler_source_->freqs_list()[1].end(), 0.0);
  EXPECT_EQ(sum1, 39.0);
}

TEST_F(DeepSamplerSourceTest, Unimplemented) {
  EXPECT_EQ(sampler_source_->node_keys().size(), 1u);
  EXPECT_EQ(sampler_source_->node_keys()[0], 0u);
  EXPECT_EQ(sampler_source_->FindContext(0u), nullptr);
}

}  // namespace embedx
