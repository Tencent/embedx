// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yong Zhou (zhouyongnju@gmail.com)
//

#include "src/sampler/sampling.h"

#include <deepx_core/dx_log.h>
#include <gtest/gtest.h>

#include <memory>   // std::unique_ptr
#include <numeric>  // std::accumulate
#include <string>

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/io/line_parser.h"
#include "src/io/value.h"
#include "src/sampler/sampling/sampling_validator.h"

namespace embedx {

class SamplingTest : public ::testing::Test {
 protected:
  LineParser parser_;

 protected:
  const std::string FREQ_FILE = "testdata/freq";
  const int BATCH = 32;

 protected:
  vec_int_t nodes_;
  vec_float_t probs_;
  vec_float_t normed_probs_;
  vec_pair_t normed_distribution_;
  vec_pair_t uniform_distribution_;

  std::unique_ptr<Sampling> sampler_;
  int count_ = 1000000;
  vec_int_t sampled_nodes_;

 protected:
  void SetUp() override {
    vec_str_t freq_files;
    EXPECT_TRUE(io_util::ListFile(FREQ_FILE, &freq_files));

    std::vector<NodeValue> node_freqs;
    for (const auto& file : freq_files) {
      EXPECT_TRUE(parser_.Open(file));
      // (node, frequency)
      while (parser_.NextBatch<NodeValue>(BATCH, &node_freqs)) {
        for (auto& node_freq : node_freqs) {
          nodes_.emplace_back(node_freq.node);
          probs_.emplace_back(node_freq.weight);
        }
      }
    }

    // construct distribution
    float sum = std::accumulate(probs_.begin(), probs_.end(), 0.0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
      uniform_distribution_.emplace_back(
          std::make_pair(nodes_[i], 1.0 / nodes_.size()));

      normed_probs_.emplace_back(probs_[i] / sum);
      normed_distribution_.emplace_back(
          std::make_pair(nodes_[i], probs_[i] / sum));
    }
  }

  void DoSampling(vec_int_t* sampled_nodes) {
    sampled_nodes->clear();
    while (sampled_nodes->size() < (size_t)count_) {
      int next = sampler_->Next();
      sampled_nodes->emplace_back(nodes_[next]);
    }
  }
};

TEST_F(SamplingTest, UniformSampling) {
  vec_float_t uniform_probs(nodes_.size(), 1.0 / nodes_.size());

  sampler_ = NewSampling(&uniform_probs, SamplingEnum::UNIFORM);
  DoSampling(&sampled_nodes_);
  EXPECT_TRUE(SamplingValidator::Test(uniform_distribution_, sampled_nodes_));
}

TEST_F(SamplingTest, AliasSampling) {
  sampler_ = NewSampling(&normed_probs_, SamplingEnum::ALIAS);
  DoSampling(&sampled_nodes_);
  EXPECT_TRUE(SamplingValidator::Test(normed_distribution_, sampled_nodes_));
}

TEST_F(SamplingTest, PartialSumSampling) {
  sampler_ = NewSampling(&normed_probs_, SamplingEnum::PARTIAL_SUM);
  DoSampling(&sampled_nodes_);
  EXPECT_TRUE(SamplingValidator::Test(normed_distribution_, sampled_nodes_));
}

}  // namespace embedx
