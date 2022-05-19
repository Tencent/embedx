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
#include "src/sampler/sampler_builder.h"

namespace embedx {

class NeighborSampler {
 private:
  const SamplerBuilder& sampler_builder_;

 public:
  explicit NeighborSampler(const SamplerBuilder* sampler_builder)
      : sampler_builder_(*sampler_builder) {}

 public:
  bool Sample(int count, const vec_int_t& nodes,
              std::vector<vec_int_t>* neighbor_nodes_list) const;

 private:
  void DoSampling(int_t node, int count, vec_int_t* neighbor_nodes) const;
  void FullSampling(int_t node, vec_int_t* neighbor_nodes) const;
  void NoReplacementSampling(int_t node, int count,
                             vec_int_t* neighbor_nodes) const;
  void WithReplacementSampling(int_t node, int count,
                               vec_int_t* neighbor_nodes) const;
};

std::unique_ptr<NeighborSampler> NewNeighborSampler(
    const SamplerBuilder* sampler_builder);

}  // namespace embedx
