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
#include <string>

#include "src/common/data_types.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {

class NegativeSamplerBuilderTest : public ::testing::Test {
 protected:
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> sampler_builder_;

 protected:
  const std::string FREQ_FILE = "testdata/freq";
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";
  const int THREAD_NUM = 3;
};

TEST_F(NegativeSamplerBuilderTest, Init_OneNameSpace) {
  sampler_source_ = NewMockSamplerSource(FREQ_FILE, "", THREAD_NUM);
  EXPECT_TRUE(sampler_source_ != nullptr);
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, 1);
  EXPECT_TRUE(sampler_builder_ != nullptr);
}

TEST_F(NegativeSamplerBuilderTest, Init_TwoNameSpace) {
  sampler_source_ =
      NewMockSamplerSource(USER_ITEM_CONTEXT, USER_ITEM_CONFIG, THREAD_NUM);
  EXPECT_TRUE(sampler_source_ != nullptr);
  auto sampler_builder_ = NewSamplerBuilder(
      sampler_source_.get(), SamplerBuilderEnum::NEGATIVE_SAMPLER,
      (int)SamplingEnum::UNIFORM, 1);
  EXPECT_TRUE(sampler_builder_ != nullptr);
}

TEST_F(NegativeSamplerBuilderTest, Next) {
  sampler_source_ =
      NewMockSamplerSource(USER_ITEM_CONTEXT, USER_ITEM_CONFIG, THREAD_NUM);
  EXPECT_TRUE(sampler_source_ != nullptr);

  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEGATIVE_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  EXPECT_TRUE(sampler_builder_ != nullptr);

  int_t next;
  EXPECT_TRUE(sampler_builder_->Next(0, &next));
  const auto& node_keys = sampler_source_->node_keys();
  auto it = std::find_if(node_keys.begin(), node_keys.end(),
                         [next](const int_t node) { return node == next; });
  EXPECT_TRUE(it != node_keys.end());
}

}  // namespace embedx
