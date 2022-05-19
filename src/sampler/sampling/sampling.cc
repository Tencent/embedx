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

#include "src/sampler/sampling.h"

namespace embedx {

std::unique_ptr<Sampling> NewUniformSampling(const vec_float_t* probs);
std::unique_ptr<Sampling> NewAliasSampling(const vec_float_t* probs);
std::unique_ptr<Sampling> NewWord2vecSampling(const vec_float_t* probs);
std::unique_ptr<Sampling> NewPartialSumSampling(const vec_float_t* probs);

std::unique_ptr<Sampling> NewSampling(const vec_float_t* probs,
                                      SamplingEnum type) {
  std::unique_ptr<Sampling> sampling;
  switch (type) {
    case SamplingEnum::UNIFORM:
      sampling = NewUniformSampling(probs);
      break;
    case SamplingEnum::ALIAS:
      sampling = NewAliasSampling(probs);
      break;
    case SamplingEnum::WORD2VEC:
      sampling = NewWord2vecSampling(probs);
      break;
    case SamplingEnum::PARTIAL_SUM:
      sampling = NewPartialSumSampling(probs);
      break;
    default:
      DXERROR(
          "Need type: UNIFORM(0) || ALIAS(1) || WORD2VEC(2) || PARTIAL_SUM(3), "
          "got type: %d.",
          (int)type);
      break;
  }
  return sampling;
}

}  // namespace embedx
