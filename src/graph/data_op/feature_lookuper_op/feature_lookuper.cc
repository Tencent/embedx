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

#include "src/graph/data_op/feature_lookuper_op/feature_lookuper.h"

#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool FeatureLookuper::Run(const vec_int_t& nodes,
                          std::vector<vec_pair_t>* node_feats,
                          std::vector<vec_pair_t>* neigh_feats) const {
  if (!feature_->LookupFeature(nodes, node_feats, neigh_feats)) {
    DXERROR("Failed to get node and neighbor feature.");
    return false;
  }
  return true;
}

int FeatureLookuper::HandleRpc(const FeatureLookuperRequest& req,
                               FeatureLookuperResponse* resp) const {
  if (!Run(req.nodes, &resp->node_feats, &resp->neigh_feats)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("FeatureLookuper", FeatureLookuper);

}  // namespace graph_op
}  // namespace embedx
