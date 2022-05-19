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

#pragma once
#include <functional>
#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class SamplerBuilder {
 protected:
  const SamplerSource& sampler_source_;
  int sampling_type_ = 0;
  int thread_num_ = 1;

 protected:
  std::function<bool(int_t cur_node, int_t* next_node)> next_func_;
  std::function<bool(int_t cur_node, int begin, int end, int_t* next_node)>
      range_next_func_;

 public:
  SamplerBuilder(const SamplerSource* sampler_source, int sampling_type,
                 int thread_num)
      : sampler_source_(*sampler_source),
        sampling_type_(sampling_type),
        thread_num_(thread_num) {}
  virtual ~SamplerBuilder() = default;

 public:
  virtual bool Init();

 public:
  const SamplerSource& sampler_source() const noexcept {
    return sampler_source_;
  }

 public:
  bool Next(int_t cur_node, int_t* next_node) const noexcept {
    return next_func_(cur_node, next_node);
  }

  bool Next(int_t cur_node, int begin, int end,
            int_t* next_node) const noexcept {
    return range_next_func_(cur_node, begin, end, next_node);
  }

 protected:
  virtual bool InitUniformFuncs() = 0;
  virtual bool InitFrequencySampler() = 0;
  virtual bool InitFrequencyFuncs() = 0;
};

enum class SamplerBuilderEnum : int {
  NEIGHBOR_SAMPLER = 0,
  NEGATIVE_SAMPLER = 1,
};

std::unique_ptr<SamplerBuilder> NewSamplerBuilder(
    const SamplerSource* sampler_source, SamplerBuilderEnum type,
    int sampler_type, int thread_num);

}  // namespace embedx
