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

#include "src/graph/data_op/feature_lookuper_op/node_feature_lookuper.h"

#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool NodeFeatureLookuper::Run(const vec_int_t& nodes,
                              std::vector<vec_pair_t>* node_feats) const {
  if (!feature_->LookupNodeFeature(nodes, node_feats)) {
    DXERROR("Failed to lookup node feature.");
    return false;
  }

  return true;
}

int NodeFeatureLookuper::HandleRpc(const NodeFeatureLookuperRequest& req,
                                   NodeFeatureLookuperResponse* resp) const {
  if (!Run(req.nodes, &resp->node_feats)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("NodeFeatureLookuper", NodeFeatureLookuper);

}  // namespace graph_op
}  // namespace embedx
