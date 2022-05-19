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
#include <mutex>
#include <vector>

#include "src/common/data_types.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {

class NegativeSamplerBuilder : public SamplerBuilder {
 private:
  std::mutex mtx_;
  std::vector<std::unique_ptr<Sampling>> samplings_;

 public:
  ~NegativeSamplerBuilder() override = default;

 public:
  static std::unique_ptr<SamplerBuilder> Create(
      const SamplerSource* sampler_source, int sampler_type, int thread_num);

 private:
  bool InitUniformFuncs() override;
  bool InitFrequencySampler() override;
  bool InitFrequencyFuncs() override;

 private:
  NegativeSamplerBuilder(const SamplerSource* sampler_source, int sampler_type,
                         int thread_num)
      : SamplerBuilder(sampler_source, sampler_type, thread_num) {}
};

}  // namespace embedx
