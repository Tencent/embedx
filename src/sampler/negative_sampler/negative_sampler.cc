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

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::find_if
#include <utility>    // std::move

namespace embedx {

bool NegativeSampler::DoSampling(int count, const vec_int_t& candidates,
                                 const vec_int_t& excluded_nodes,
                                 vec_int_t* sampled_nodes) const {
  sampled_nodes->clear();
  int_t next_node;
  while (sampled_nodes->size() < (size_t)count) {
    if (!sampler_builder_.Next(candidates[0], &next_node)) {
      return false;
    }

    // o(n) !!!
    auto it =
        std::find_if(excluded_nodes.begin(), excluded_nodes.end(),
                     [next_node](int_t node) { return node == next_node; });
    if (it == excluded_nodes.end()) {
      sampled_nodes->emplace_back(next_node);
    }
  }

  return sampled_nodes->size() == (size_t)count;
}

std::unique_ptr<NegativeSampler> NewSharedNegativeSampler(
    const SamplerBuilder* sampler_builder);
std::unique_ptr<NegativeSampler> NewIndepNegativeSampler(
    const SamplerBuilder* sampler_builder);

std::unique_ptr<NegativeSampler> NewNegativeSampler(
    const SamplerBuilder* sampler_builder, NegativeSamplerEnum type) {
  std::unique_ptr<NegativeSampler> sampler;
  switch (type) {
    case NegativeSamplerEnum::SHARED:
      sampler = NewSharedNegativeSampler(sampler_builder);
      break;
    case NegativeSamplerEnum::INDEPENDENT:
      sampler = NewIndepNegativeSampler(sampler_builder);
      break;
    default:
      DXERROR("Need type: SHARED(0) || INDEPENDENT(1), got type: %d.",
              (int)type);
      break;
  }
  return sampler;
}

}  // namespace embedx
