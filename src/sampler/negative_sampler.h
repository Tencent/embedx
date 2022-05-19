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

#pragma once
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/sampler/sampler_builder.h"

namespace embedx {

class NegativeSampler {
 protected:
  const SamplerBuilder& sampler_builder_;

 public:
  explicit NegativeSampler(const SamplerBuilder* sampler_builder)
      : sampler_builder_(*sampler_builder) {}
  virtual ~NegativeSampler() = default;

 public:
  virtual bool Sample(int count, const vec_int_t& nodes,
                      const vec_int_t& excluded_nodes,
                      std::vector<vec_int_t>* sampled_nodes_list) const = 0;

 protected:
  bool DoSampling(int count, const vec_int_t& candidates,
                  const vec_int_t& excluded_nodes,
                  vec_int_t* sampled_nodes) const;
};

enum class NegativeSamplerEnum : int {
  SHARED = 0,
  INDEPENDENT = 1,
};

std::unique_ptr<NegativeSampler> NewNegativeSampler(
    const SamplerBuilder* sampler_builder, NegativeSamplerEnum type);

}  // namespace embedx
