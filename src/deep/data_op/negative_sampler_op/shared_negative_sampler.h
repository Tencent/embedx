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
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/data_op/deep_op.h"
#include "src/deep/data_op/deep_op_resource.h"
#include "src/sampler/negative_sampler.h"

namespace embedx {
namespace deep_op {

class SharedNegativeSampler : public LocalDeepOp {
 private:
  std::unique_ptr<NegativeSampler> negative_sampler_;

 public:
  ~SharedNegativeSampler() override = default;

 public:
  bool Run(int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
           std::vector<vec_int_t>* sampled_nodes_list) const;

 private:
  bool Init(const LocalDeepOpResource* resource) override {
    negative_sampler_ = NewNegativeSampler(resource->negative_sampler_builder(),
                                           NegativeSamplerEnum::SHARED);
    return negative_sampler_ != nullptr;
  }
};

}  // namespace deep_op
}  // namespace embedx
