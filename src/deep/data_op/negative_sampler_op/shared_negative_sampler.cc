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

#include "shared_negative_sampler.h"

#include <deepx_core/dx_log.h>

#include "src/deep/data_op/deep_op_registry.h"

namespace embedx {
namespace deep_op {

bool SharedNegativeSampler::Run(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  if (!negative_sampler_->Sample(count, nodes, excluded_nodes,
                                 sampled_nodes_list)) {
    DXERROR("Failed to shared sample node.");
    return false;
  }
  return true;
}

REGISTER_LOCAL_DEEP_OP("SharedNegativeSampler", SharedNegativeSampler);

}  // namespace deep_op
}  // namespace embedx
