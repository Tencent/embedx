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

#include "src/sampler/negative_sampler.h"

#include <gtest/gtest.h>

#include <algorithm>  //std::find
#include <memory>     // std::unique_ptr
#include <string>
#include <unordered_set>
#include <vector>

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class NegativeSamplerTest : public ::testing::Test {
 protected:
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> sampler_builder_;

  std::unique_ptr<NegativeSampler> sampler_;
  std::unordered_set<uint16_t> ns_id_set_;
  int count_;
  vec_int_t nodes_;
  vec_int_t excluded_nodes_;
  std::vector<vec_int_t> sampled_nodes_list_;
  const std::vector<uint16_t> NEGATIVE_SAMPLER_TYPE{0, 1, 2};

 protected:
  const std::string CONTEXT = "testdata/context";
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";

  const int THREAD_NUM = 3;

 protected:
  void SetUp() override { count_ = 10; }
};

TEST_F(NegativeSamplerTest, SharedSample_OneNameSpace) {
  for (auto sampler_type : NEGATIVE_SAMPLER_TYPE) {
    sampler_source_ = NewMockSamplerSource(CONTEXT, "", THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);

    sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                         SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                         sampler_type, THREAD_NUM);
    sampler_ =
        NewNegativeSampler(sampler_builder_.get(), NegativeSamplerEnum::SHARED);
    EXPECT_TRUE(sampler_);

    nodes_ = {0, 9};
    excluded_nodes_ = {1, 10};
    EXPECT_TRUE(sampler_->Sample(count_, nodes_, excluded_nodes_,
                                 &sampled_nodes_list_));
    EXPECT_EQ(sampled_nodes_list_.size(), 1u);
    EXPECT_EQ(count_, (int)sampled_nodes_list_[0].size());

    // exclude
    EXPECT_TRUE(std::find(sampled_nodes_list_[0].begin(),
                          sampled_nodes_list_[0].end(),
                          1) == sampled_nodes_list_[0].end());
    EXPECT_TRUE(std::find(sampled_nodes_list_[0].begin(),
                          sampled_nodes_list_[0].end(),
                          1) == sampled_nodes_list_[0].end());
    EXPECT_TRUE(std::find(sampled_nodes_list_[0].begin(),
                          sampled_nodes_list_[0].end(),
                          10) == sampled_nodes_list_[0].end());
    EXPECT_TRUE(std::find(sampled_nodes_list_[0].begin(),
                          sampled_nodes_list_[0].end(),
                          10) == sampled_nodes_list_[0].end());
  }
}

TEST_F(NegativeSamplerTest, IndepSample_OneNameSpace) {
  for (auto sampler_type : NEGATIVE_SAMPLER_TYPE) {
    sampler_source_ = NewMockSamplerSource(CONTEXT, "", THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);

    sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                         SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                         sampler_type, THREAD_NUM);
    sampler_ = NewNegativeSampler(sampler_builder_.get(),
                                  NegativeSamplerEnum::INDEPENDENT);
    EXPECT_TRUE(sampler_);

    nodes_ = {0, 9};
    excluded_nodes_ = {1, 10};
    EXPECT_TRUE(sampler_->Sample(count_, nodes_, excluded_nodes_,
                                 &sampled_nodes_list_));
    EXPECT_EQ(sampled_nodes_list_.size(), 2u);

    // exclude
    for (size_t i = 0; i < nodes_.size(); ++i) {
      EXPECT_TRUE(std::find(sampled_nodes_list_[i].begin(),
                            sampled_nodes_list_[i].end(),
                            1) == sampled_nodes_list_[i].end());
      EXPECT_TRUE(std::find(sampled_nodes_list_[i].begin(),
                            sampled_nodes_list_[i].end(),
                            10) == sampled_nodes_list_[i].end());
      EXPECT_EQ(count_, (int)sampled_nodes_list_[i].size());
    }
  }
}

TEST_F(NegativeSamplerTest, SharedSample_TwoNameSpace) {
  for (auto sampler_type : NEGATIVE_SAMPLER_TYPE) {
    sampler_source_ =
        NewMockSamplerSource(USER_ITEM_CONTEXT, USER_ITEM_CONFIG, THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);

    sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                         SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                         sampler_type, THREAD_NUM);
    sampler_ =
        NewNegativeSampler(sampler_builder_.get(), NegativeSamplerEnum::SHARED);
    EXPECT_TRUE(sampler_);

    nodes_ = {1, 3, 546760133882592, 301675120304337};
    excluded_nodes_ = {2, 416653778443095};
    EXPECT_TRUE(sampler_->Sample(count_, nodes_, excluded_nodes_,
                                 &sampled_nodes_list_));
    EXPECT_EQ(sampled_nodes_list_.size(), 2u);

    // exclude
    io_util::ParseMaxNodeType(2, nodes_, &ns_id_set_);
    for (auto ns_id : ns_id_set_) {
      EXPECT_TRUE(std::find(sampled_nodes_list_[ns_id].begin(),
                            sampled_nodes_list_[ns_id].end(),
                            2) == sampled_nodes_list_[ns_id].end());
      EXPECT_TRUE(std::find(sampled_nodes_list_[ns_id].begin(),
                            sampled_nodes_list_[ns_id].end(),
                            416653778443095) ==
                  sampled_nodes_list_[ns_id].end());

      EXPECT_EQ(count_, (int)sampled_nodes_list_[ns_id].size());

      // same namespace
      for (auto node : sampled_nodes_list_[ns_id]) {
        EXPECT_EQ(io_util::GetNodeType(node), ns_id);
      }
    }
  }
}

TEST_F(NegativeSamplerTest, IndepSample_TwoNameSpace) {
  for (auto sampler_type : NEGATIVE_SAMPLER_TYPE) {
    sampler_source_ =
        NewMockSamplerSource(USER_ITEM_CONTEXT, USER_ITEM_CONFIG, THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);

    sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                         SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                         sampler_type, THREAD_NUM);
    sampler_ = NewNegativeSampler(sampler_builder_.get(),
                                  NegativeSamplerEnum::INDEPENDENT);
    EXPECT_TRUE(sampler_);

    nodes_ = {1, 3, 546760133882592, 301675120304337};
    excluded_nodes_ = {2, 416653778443095};
    EXPECT_TRUE(sampler_->Sample(count_, nodes_, excluded_nodes_,
                                 &sampled_nodes_list_));
    EXPECT_EQ(sampled_nodes_list_.size(), 4u);

    // exclude
    for (size_t i = 0; i < nodes_.size(); ++i) {
      EXPECT_TRUE(std::find(sampled_nodes_list_[i].begin(),
                            sampled_nodes_list_[i].end(),
                            2) == sampled_nodes_list_[i].end());
      EXPECT_TRUE(std::find(sampled_nodes_list_[i].begin(),
                            sampled_nodes_list_[i].end(),
                            416653778443095) == sampled_nodes_list_[i].end());

      EXPECT_EQ(count_, (int)sampled_nodes_list_[i].size());
      // same namespace
      auto ns_id = io_util::GetNodeType(nodes_[i]);
      for (const auto& node : sampled_nodes_list_[i]) {
        EXPECT_EQ(io_util::GetNodeType(node), ns_id);
      }
    }
  }
}

}  // namespace embedx
