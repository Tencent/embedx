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

#include "src/sampler/sampler_builder.h"

#include <utility>  // std::move

#include "src/sampler/sampling.h"

namespace embedx {

bool SamplerBuilder::Init() {
  if (sampling_type_ == (int)SamplingEnum::UNIFORM) {
    return InitUniformFuncs();
  } else {
    return InitFrequencySampler() && InitFrequencyFuncs();
  }
}

std::unique_ptr<SamplerBuilder> NewNeighborSamplerBuilder(
    const SamplerSource* sampler_source, int sampler_type, int thread_num);
std::unique_ptr<SamplerBuilder> NewNegativeSamplerBuilder(
    const SamplerSource* sampler_source, int sampler_type, int thread_num);

std::unique_ptr<SamplerBuilder> NewSamplerBuilder(
    const SamplerSource* sampler_source, SamplerBuilderEnum type,
    int sampler_type, int thread_num) {
  std::unique_ptr<SamplerBuilder> sampler_builder;
  switch (type) {
    case SamplerBuilderEnum::NEIGHBOR_SAMPLER:
      sampler_builder =
          NewNeighborSamplerBuilder(sampler_source, sampler_type, thread_num);
      break;
    case SamplerBuilderEnum::NEGATIVE_SAMPLER:
      sampler_builder =
          NewNegativeSamplerBuilder(sampler_source, sampler_type, thread_num);
      break;
    default:
      DXERROR(
          "Need type: NEIGHBOR_SAMPLER(0) || NEGATIVE_SAMPLER(1), got type: "
          "%d.",
          (int)type);
      break;
  }
  return sampler_builder;
}

}  // namespace embedx
