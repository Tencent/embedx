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
#include <unordered_set>

#include "src/common/data_types.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {

class NeighborSamplerBuilderTest : public ::testing::Test {
 protected:
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> sampler_builder_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    sampler_source_ = NewMockSamplerSource(CONTEXT, "", THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);
  }
};

TEST_F(NeighborSamplerBuilderTest, Init) {
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  EXPECT_TRUE(sampler_builder_ != nullptr);
}

TEST_F(NeighborSamplerBuilderTest, Next) {
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  EXPECT_TRUE(sampler_builder_ != nullptr);

  int_t next;
  EXPECT_TRUE(sampler_builder_->Next(9u, &next));
  std::unordered_set<int_t> expected = {6u, 7u, 8u};
  auto* context = sampler_source_->FindContext(9u);
  EXPECT_TRUE(context != nullptr);
  auto it =
      std::find_if(context->begin(), context->end(),
                   [next](const pair_t& entry) { return entry.first == next; });
  EXPECT_TRUE(it != context->end());
}

TEST_F(NeighborSamplerBuilderTest, RangeNext) {
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  EXPECT_TRUE(sampler_builder_ != nullptr);

  int_t next;
  EXPECT_TRUE(sampler_builder_->Next(0u, 2, 3, &next));
  EXPECT_EQ(next, 12u);
}
}  // namespace embedx
