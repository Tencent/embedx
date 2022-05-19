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

#include "src/graph/data_op/negative_sampler_op/shared_negative_sampler.h"

#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

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

int SharedNegativeSampler::HandleRpc(
    const SharedNegativeSamplerRequest& req,
    SharedNegativeSamplerResponse* resp) const {
  if (!Run(req.count, req.nodes, req.excluded_nodes,
           &resp->sampled_nodes_list)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("SharedNegativeSampler", SharedNegativeSampler);

}  // namespace graph_op
}  // namespace embedx
