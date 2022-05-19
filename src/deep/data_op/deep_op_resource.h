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

#pragma once
#include <memory>   // std::unique_ptr
#include <utility>  // std::move

#include "src/deep/deep_config.h"
#include "src/deep/deep_data.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"

namespace embedx {
namespace deep_op {

class LocalDeepOpResource {
 private:
  DeepConfig deep_config_;
  std::unique_ptr<DeepData> deep_data_;
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> negative_sampler_builder_;

 public:
  const DeepConfig& deep_config() const noexcept { return deep_config_; }
  const DeepData* deep_data() const noexcept { return deep_data_.get(); }
  const SamplerSource* sampler_source() const noexcept {
    return sampler_source_.get();
  }
  const SamplerBuilder* negative_sampler_builder() const noexcept {
    return negative_sampler_builder_.get();
  }

 public:
  void set_deep_config(const DeepConfig& deep_config) {
    deep_config_ = deep_config;
  }
  void set_deep_data(std::unique_ptr<DeepData> deep_data) {
    deep_data_ = std::move(deep_data);
  }
  void set_sampler_source(std::unique_ptr<SamplerSource> sampler_source) {
    sampler_source_ = std::move(sampler_source);
  }
  void set_negative_sampler_builder(
      std::unique_ptr<SamplerBuilder> sampler_builder) {
    negative_sampler_builder_ = std::move(sampler_builder);
  }
};

}  // namespace deep_op
}  // namespace embedx
