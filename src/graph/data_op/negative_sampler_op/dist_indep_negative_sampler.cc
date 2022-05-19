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

#include "src/graph/data_op/negative_sampler_op/dist_indep_negative_sampler.h"

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistIndepNegativeSampler::Run(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  // prepare
  std::vector<int> masks;
  std::vector<std::vector<int>> indices_list(shard_num_);
  std::vector<IndepNegativeSamplerRequest> requests(shard_num_);
  std::vector<IndepNegativeSamplerResponse> responses(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    indices_list[i].clear();
    requests[i].count = count;
    requests[i].excluded_nodes = excluded_nodes;
    requests[i].nodes.clear();
  }

  // map
  masks.assign(shard_num_, 0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    int shard_id = ModShard(nodes[i]);
    indices_list[shard_id].emplace_back((int)i);
    requests[shard_id].nodes.emplace_back(nodes[i]);
    masks[shard_id] += 1;
  }

  // rpc
  auto rpc_type = IndepNegativeSamplerRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce
  sampled_nodes_list->clear();
  sampled_nodes_list->resize(nodes.size());
  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indice_list = indices_list[i];
      const auto& remote_nodes_list = responses[i].sampled_nodes_list;
      for (size_t j = 0; j < remote_nodes_list.size(); ++j) {
        (*sampled_nodes_list)[indice_list[j]] = remote_nodes_list[j];
      }
    }
  }
  return true;
}

REGISTER_DIST_GS_OP("IndepNegativeSampler", DistIndepNegativeSampler);

}  // namespace graph_op
}  // namespace embedx
