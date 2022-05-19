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

#include "src/sampler/neighbor_sampler.h"

#include <gtest/gtest.h>

#include <algorithm>  // std::find_if
#include <memory>     // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {

class NeighborSamplerTest : public ::testing::Test {
 protected:
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> sampler_builder_;
  std::unique_ptr<NeighborSampler> neighbor_sampler_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    sampler_source_ = NewMockSamplerSource(CONTEXT, "", THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);
  }
};

TEST_F(NeighborSamplerTest, Full_Sample) {
  int count = -1;
  vec_int_t nodes = {0, 9};
  std::vector<vec_int_t> neighbor_nodes_list;

  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  neighbor_sampler_.reset(new NeighborSampler(sampler_builder_.get()));
  EXPECT_TRUE(neighbor_sampler_);
  EXPECT_TRUE(neighbor_sampler_->Sample(count, nodes, &neighbor_nodes_list));
  EXPECT_EQ(neighbor_nodes_list.size(), nodes.size());

  std::vector<vec_int_t> expected{{10, 11, 12}, {6, 7, 8}};
  for (size_t i = 0; i < neighbor_nodes_list.size(); ++i) {
    for (size_t j = 0; j < neighbor_nodes_list[i].size(); ++j) {
      EXPECT_EQ(neighbor_nodes_list[i][j], expected[i][j]);
    }
  }
}

TEST_F(NeighborSamplerTest, Uniform_Sample) {
  int count = 3;
  vec_int_t nodes = {0, 9};
  std::vector<vec_int_t> neighbor_nodes_list;

  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  neighbor_sampler_.reset(new NeighborSampler(sampler_builder_.get()));
  EXPECT_TRUE(neighbor_sampler_);
  EXPECT_TRUE(neighbor_sampler_->Sample(count, nodes, &neighbor_nodes_list));
  EXPECT_EQ(neighbor_nodes_list.size(), nodes.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto* context = sampler_source_->FindContext(nodes[i]);
    // in context
    for (auto node : neighbor_nodes_list[i]) {
      auto it = std::find_if(
          context->begin(), context->end(),
          [node](const pair_t& entry) { return entry.first == node; });
      EXPECT_TRUE(it != context->end());
    }
  }
}

TEST_F(NeighborSamplerTest, Frequency_Sample) {
  int count = 3;
  vec_int_t nodes = {0, 9};
  std::vector<vec_int_t> neighbor_nodes_list;

  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::ALIAS, THREAD_NUM);
  neighbor_sampler_.reset(new NeighborSampler(sampler_builder_.get()));
  EXPECT_TRUE(neighbor_sampler_);
  EXPECT_TRUE(neighbor_sampler_->Sample(count, nodes, &neighbor_nodes_list));
  EXPECT_EQ(neighbor_nodes_list.size(), nodes.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto* context = sampler_source_->FindContext(nodes[i]);
    // in context
    for (auto node : neighbor_nodes_list[i]) {
      auto it = std::find_if(
          context->begin(), context->end(),
          [node](const pair_t& entry) { return entry.first == node; });
      EXPECT_TRUE(it != context->end());
    }
  }
}

}  // namespace embedx
