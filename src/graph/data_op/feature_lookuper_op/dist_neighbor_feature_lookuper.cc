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

#include "src/graph/data_op/feature_lookuper_op/dist_neighbor_feature_lookuper.h"

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistNeighborFeatureLookuper::Run(
    const vec_int_t& nodes, std::vector<vec_pair_t>* neigh_feats) const {
  // prepare
  std::vector<int> masks;
  std::vector<std::vector<int>> indices_list(shard_num_);
  std::vector<NeighborFeatureLookuperRequest> requests(shard_num_);
  std::vector<NeighborFeatureLookuperResponse> responses(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    indices_list[i].clear();
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
  auto rpc_type = NeighborFeatureLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce
  neigh_feats->clear();
  neigh_feats->resize(nodes.size());
  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indices = indices_list[i];
      const auto& remote_feats = responses[i].neigh_feats;
      for (size_t j = 0; j < remote_feats.size(); ++j) {
        const auto& remote_feats_j = remote_feats[j];
        auto& local_feats_j = (*neigh_feats)[indices[j]];
        local_feats_j.insert(local_feats_j.end(), remote_feats_j.begin(),
                             remote_feats_j.end());
      }
    }
  }
  return true;
}

REGISTER_DIST_GS_OP("NeighborFeatureLookuper", DistNeighborFeatureLookuper);

}  // namespace graph_op
}  // namespace embedx
