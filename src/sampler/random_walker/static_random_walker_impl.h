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
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/sampler/random_walker/random_walker_impl.h"
#include "src/sampler/random_walker_data_types.h"
#include "src/sampler/sampler_builder.h"

namespace embedx {

class StaticRandomWalkerImpl : public RandomWalkerImpl {
 private:
  const SamplerBuilder& neighbor_sampler_builder_;

 public:
  ~StaticRandomWalkerImpl() override = default;

 public:
  static std::unique_ptr<RandomWalkerImpl> Create(
      const SamplerBuilder* sampler_builder);

 public:
  void Traverse(const vec_int_t& cur_nodes, const std::vector<int>& walk_lens,
                const WalkerInfo& walker_info, std::vector<vec_int_t>* seqs,
                PrevInfo* prev_info) const override;

 private:
  explicit StaticRandomWalkerImpl(const SamplerBuilder* sampler_builder)
      : neighbor_sampler_builder_(*sampler_builder) {}
};

}  // namespace embedx
