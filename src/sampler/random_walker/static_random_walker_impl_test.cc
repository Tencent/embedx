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

#include <gtest/gtest.h>

#include <algorithm>  // std::find_if
#include <memory>     // std::unique_ptr
#include <string>

#include "src/common/data_types.h"
#include "src/sampler/random_walker.h"
#include "src/sampler/random_walker_data_types.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {

class StaticRandomWalkerImplTest : public ::testing::Test {
 protected:
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> sampler_builder_;
  std::unique_ptr<RandomWalker> random_walker_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const int THREAD_NUM = 3;

 protected:
  void SetUp() override {
    sampler_source_ = NewMockSamplerSource(CONTEXT, "", THREAD_NUM);
    EXPECT_TRUE(sampler_source_ != nullptr);
  }
};

TEST_F(StaticRandomWalkerImplTest, Traverse_UniformNeighborSampler) {
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::UNIFORM, THREAD_NUM);
  random_walker_ =
      NewRandomWalker(sampler_builder_.get(), RandomWalkerEnum::STATIC);
  EXPECT_TRUE(random_walker_);

  vec_int_t cur_nodes = {0, 9};
  std::vector<int> walk_lens = {3, 3};
  WalkerInfo walker_info;
  std::vector<vec_int_t> seqs;

  random_walker_->Traverse(cur_nodes, walk_lens, walker_info, &seqs, nullptr);
  EXPECT_EQ(seqs.size(), cur_nodes.size());

  vec_int_t pre_nodes = {0, 9};
  for (size_t i = 0; i < seqs.size(); ++i) {
    for (size_t j = 0; j < seqs[i].size(); ++j) {
      auto* context = sampler_source_->FindContext(pre_nodes[i]);
      auto seq_node = seqs[i][j];
      // in context
      auto it = std::find_if(
          context->begin(), context->end(),
          [seq_node](const pair_t& entry) { return entry.first == seq_node; });
      EXPECT_TRUE(it != context->end());
      pre_nodes[i] = seqs[i][j];
    }
  }
}

TEST_F(StaticRandomWalkerImplTest, Traverse_FrequencyNeighborSampler) {
  sampler_builder_ = NewSamplerBuilder(sampler_source_.get(),
                                       SamplerBuilderEnum::NEIGHBOR_SAMPLER,
                                       (int)SamplingEnum::ALIAS, THREAD_NUM);
  random_walker_ =
      NewRandomWalker(sampler_builder_.get(), RandomWalkerEnum::STATIC);
  EXPECT_TRUE(random_walker_);

  vec_int_t cur_nodes = {0, 9};
  std::vector<int> walk_lens = {3, 3};
  WalkerInfo walker_info;
  std::vector<vec_int_t> seqs;

  random_walker_->Traverse(cur_nodes, walk_lens, walker_info, &seqs, nullptr);
  EXPECT_EQ(seqs.size(), cur_nodes.size());

  vec_int_t pre_nodes = {0, 9};
  for (size_t i = 0; i < seqs.size(); ++i) {
    for (size_t j = 0; j < seqs[i].size(); ++j) {
      auto* context = sampler_source_->FindContext(pre_nodes[i]);
      auto seq_node = seqs[i][j];
      // in context
      auto it = std::find_if(
          context->begin(), context->end(),
          [seq_node](const pair_t& entry) { return entry.first == seq_node; });
      EXPECT_TRUE(it != context->end());
      pre_nodes[i] = seqs[i][j];
    }
  }
}

}  // namespace embedx
