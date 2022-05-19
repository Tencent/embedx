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

#include "src/graph/data_op/context_lookuper_op/dist_context_lookuper.h"

#include <deepx_core/dx_log.h>

#include <utility>  // std::move

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistContextLookuper::Run(const vec_int_t& nodes,
                              std::vector<vec_pair_t>* contexts) const {
  // prepare
  std::vector<int> masks(shard_num_, 0);
  std::vector<std::vector<int>> indices(shard_num_);
  std::vector<ContextLookuperRequest> requests(shard_num_);
  std::vector<ContextLookuperResponse> responses(shard_num_);

  // map
  for (size_t i = 0; i < nodes.size(); ++i) {
    int shard_id = ModShard(nodes[i]);

    masks[shard_id] += 1;
    indices[shard_id].emplace_back(i);
    requests[shard_id].nodes.emplace_back(nodes[i]);
  }

  // rpc
  auto rpc_type = ContextLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce
  contexts->clear();
  contexts->resize(nodes.size());
  for (int i = 0; i < shard_num_; ++i) {
    if (!masks[i]) {
      continue;
    }

    const auto& cur_indice = indices[i];
    auto& cur_context = responses[i].contexts;

    if (cur_indice.size() != cur_context.size()) {
      DXERROR(
          "DistContextLookuper response context_list size expect: %zu, got: "
          "%zu.",
          cur_indice.size(), cur_context.size());
      return false;
    }

    for (size_t j = 0; j < cur_indice.size(); ++j) {
      (*contexts)[cur_indice[j]] = std::move(cur_context[j]);
    }
  }

  return true;
}

REGISTER_DIST_GS_OP("ContextLookuper", DistContextLookuper);

}  // namespace graph_op
}  // namespace embedx
