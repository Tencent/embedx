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

#include "src/graph/data_op/neighbor_sampler_op/dist_random_neighbor_sampler.h"

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistRandomNeighborSampler::Run(
    int count, const vec_int_t& nodes,
    std::vector<vec_int_t>* neighbor_nodes_list) const {
  // prepare
  std::vector<int> masks;
  std::vector<std::vector<int>> indices_list(shard_num_);
  std::vector<RandomNeighborSamplerRequest> requests(shard_num_);
  std::vector<RandomNeighborSamplerResponse> responses(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    indices_list[i].clear();
    requests[i].count = count;
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
  auto rpc_type = RandomNeighborSamplerRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce
  neighbor_nodes_list->clear();
  neighbor_nodes_list->resize(nodes.size());
  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indices = indices_list[i];
      const auto& remote_neighbor_lists = responses[i].neighbor_nodes_list;
      for (size_t j = 0; j < remote_neighbor_lists.size(); ++j) {
        (*neighbor_nodes_list)[indices[j]] = remote_neighbor_lists[j];
      }
    }
  }
  return true;
}

REGISTER_DIST_GS_OP("RandomNeighborSampler", DistRandomNeighborSampler);

}  // namespace graph_op
}  // namespace embedx
