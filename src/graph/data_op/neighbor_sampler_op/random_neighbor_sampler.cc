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

#include "src/graph/data_op/neighbor_sampler_op/random_neighbor_sampler.h"

#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool RandomNeighborSampler::Run(
    int count, const vec_int_t& nodes,
    std::vector<vec_int_t>* neighbor_nodes_list) const {
  if (!neighbor_sampler_->Sample(count, nodes, neighbor_nodes_list)) {
    DXERROR("Failed to sample neighbor.");
    return false;
  }

  return true;
}

int RandomNeighborSampler::HandleRpc(
    const RandomNeighborSamplerRequest& req,
    RandomNeighborSamplerResponse* resp) const {
  if (!Run(req.count, req.nodes, &resp->neighbor_nodes_list)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("RandomNeighborSampler", RandomNeighborSampler);

}  // namespace graph_op
}  // namespace embedx
