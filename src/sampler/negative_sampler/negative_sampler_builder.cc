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

#include "src/sampler/negative_sampler/negative_sampler_builder.h"

#include <deepx_core/dx_log.h>

#include <cmath>
#include <utility>  // std::move

#include "src/common/random.h"
#include "src/io/io_util.h"

namespace embedx {
namespace {

void NormalizeProbs(const vec_float_t& probs, vec_float_t* norm_probs) {
  float_t sum = 0.;
  for (auto prob : probs) {
    DXCHECK(prob > 0);
    sum += std::pow(prob, 0.75);
  }

  norm_probs->clear();
  for (auto prob : probs) {
    norm_probs->emplace_back(std::pow(prob, 0.75) / sum);
  }
}

}  // namespace

std::unique_ptr<SamplerBuilder> NegativeSamplerBuilder::Create(
    const SamplerSource* sampler_source, int sampler_type, int thread_num) {
  std::unique_ptr<SamplerBuilder> sampler_builder;
  sampler_builder.reset(
      new NegativeSamplerBuilder(sampler_source, sampler_type, thread_num));
  if (!sampler_builder->Init()) {
    DXERROR("Failed to init negative sampler builder.");
    sampler_builder.reset();
  }
  return sampler_builder;
}

bool NegativeSamplerBuilder::InitUniformFuncs() {
  DXINFO("Initing uniform negative sampler funcs...");

  next_func_ = [this](int_t cur_node, int_t* next_node) -> bool {
    auto ns_id = io_util::GetNodeType(cur_node);
    auto& candidate_nodes = sampler_source_.nodes_list()[ns_id];
    int k = int(ThreadLocalRandom() * candidate_nodes.size());
    *next_node = candidate_nodes[k];
    return true;
  };

  range_next_func_ = [this](int_t cur_node, int begin, int end,
                            int_t* next_node) -> bool {
    auto ns_id = io_util::GetNodeType(cur_node);
    auto& candidate_nodes = sampler_source_.nodes_list()[ns_id];
    int k = begin + int(ThreadLocalRandom() * (end - begin));
    *next_node = candidate_nodes[k];
    return true;
  };

  DXINFO("Done.");
  return true;
}

bool NegativeSamplerBuilder::InitFrequencySampler() {
  DXINFO("Initing frequency negative sampler, with sampler_type: %d...",
         sampling_type_);
  const auto& probs_list = sampler_source_.freqs_list();
  samplings_.resize(sampler_source_.ns_size());
  vec_float_t norm_probs;
  for (auto& entry : sampler_source_.id_name_map()) {
    auto ns_id = entry.first;
    NormalizeProbs(probs_list[ns_id], &norm_probs);
    auto sampling = NewSampling(&norm_probs, (SamplingEnum)sampling_type_);
    if (!sampling) {
      return false;
    }
    samplings_[ns_id] = std::move(sampling);
  }

  DXINFO("Done.");
  return true;
}

bool NegativeSamplerBuilder::InitFrequencyFuncs() {
  DXINFO("Initing frequency negative sampler func, with sampler_type: %d...",
         sampling_type_);

  next_func_ = [this](int_t cur_node, int_t* next_node) -> bool {
    auto ns_id = io_util::GetNodeType(cur_node);
    auto& candidate_nodes = sampler_source_.nodes_list()[ns_id];
    if (!samplings_[ns_id]) {
      DXERROR("The sampler of namespace: %d is nullptr.", (int)ns_id);
      return false;
    }

    auto k = samplings_[ns_id]->Next();
    *next_node = candidate_nodes[k];

    return true;
  };

  range_next_func_ = [this](int_t cur_node, int begin, int end,
                            int_t* next_node) -> bool {
    auto ns_id = io_util::GetNodeType(cur_node);
    auto& candidate_nodes = sampler_source_.nodes_list()[ns_id];
    if (!samplings_[ns_id]) {
      DXERROR("The sampler of namespace: %d is nullptr.", (int)ns_id);
      return false;
    }

    auto k = samplings_[ns_id]->Next(begin, end);
    *next_node = candidate_nodes[k];
    return true;
  };

  DXINFO("Done.");
  return true;
}

std::unique_ptr<SamplerBuilder> NewNegativeSamplerBuilder(
    const SamplerSource* sampler_source, int sampler_type, int thread_num) {
  return NegativeSamplerBuilder::Create(sampler_source, sampler_type,
                                        thread_num);
}

}  // namespace embedx
