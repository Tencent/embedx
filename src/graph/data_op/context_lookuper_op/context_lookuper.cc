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

#include "src/graph/data_op/context_lookuper_op/context_lookuper.h"

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool ContextLookuper::Run(const vec_int_t& nodes,
                          std::vector<vec_pair_t>* contexts) const {
  return context_->Lookup(nodes, contexts);
}

int ContextLookuper::HandleRpc(const ContextLookuperRequest& req,
                               ContextLookuperResponse* resp) const {
  if (Run(req.nodes, &resp->contexts)) {
    return 0;
  }

  return -1;
}

REGISTER_LOCAL_GS_OP("ContextLookuper", ContextLookuper);

}  // namespace graph_op
}  // namespace embedx
