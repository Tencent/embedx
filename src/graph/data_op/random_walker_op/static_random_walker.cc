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

#include "src/graph/data_op/random_walker_op/static_random_walker.h"

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool StaticRandomWalker::Run(const vec_int_t& cur_nodes,
                             const std::vector<int>& walk_lens,
                             const WalkerInfo& walker_info,
                             std::vector<vec_int_t>* seqs) const {
  random_walker_->Traverse(cur_nodes, walk_lens, walker_info, seqs, nullptr);
  return true;
}

int StaticRandomWalker::HandleRpc(const StaticRandomWalkerRequest& req,
                                  StaticRandomWalkerResponse* resp) const {
  if (!Run(req.cur_nodes, req.walk_lens, req.walker_info, &resp->seqs)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("StaticRandomWalker", StaticRandomWalker);

}  // namespace graph_op
}  // namespace embedx
