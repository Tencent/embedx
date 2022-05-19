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

#include "src/graph/data_op/feature_lookuper_op/dist_node_feature_lookuper.h"

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistNodeFeatureLookuper::Run(const vec_int_t& nodes,
                                  std::vector<vec_pair_t>* node_feats) const {
  // prepare
  std::vector<int> masks;
  std::vector<std::vector<int>> indices_list(shard_num_);
  std::vector<NodeFeatureLookuperRequest> requests(shard_num_);
  std::vector<NodeFeatureLookuperResponse> responses(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    indices_list[i].clear();
    requests[i].nodes.clear();
  }

  node_feats->clear();
  node_feats->resize(nodes.size());

  // map
  masks.assign(shard_num_, 0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto* node_feat_ptr =
        resource_->cache_storage()->FindNodeFeature(nodes[i]);

    if (node_feat_ptr != nullptr) {
      // cache hit, get node feature in cache
      (*node_feats)[i].insert((*node_feats)[i].end(), node_feat_ptr->begin(),
                              node_feat_ptr->end());
    } else {
      // cache miss, add nodes to request
      int shard_id = ModShard(nodes[i]);
      indices_list[shard_id].emplace_back((int)i);
      requests[shard_id].nodes.emplace_back(nodes[i]);
      masks[shard_id] += 1;
    }
  }

  // rpc
  auto rpc_type = NodeFeatureLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce
  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indices = indices_list[i];
      const auto& remote_feats = responses[i].node_feats;
      for (size_t j = 0; j < remote_feats.size(); ++j) {
        const auto& remote_feat_j = remote_feats[j];
        auto& local_feat_j = (*node_feats)[indices[j]];
        local_feat_j.insert(local_feat_j.end(), remote_feat_j.begin(),
                            remote_feat_j.end());
      }
    }
  }
  return true;
}

REGISTER_DIST_GS_OP("NodeFeatureLookuper", DistNodeFeatureLookuper);

}  // namespace graph_op
}  // namespace embedx
