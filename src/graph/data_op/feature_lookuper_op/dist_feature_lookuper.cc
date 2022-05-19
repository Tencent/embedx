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

#include "src/graph/data_op/feature_lookuper_op/dist_feature_lookuper.h"

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

bool DistFeatureLookuper::Run(const vec_int_t& nodes,
                              std::vector<vec_pair_t>* node_feats,
                              std::vector<vec_pair_t>* neigh_feats) const {
  // prepare
  std::vector<int> masks;
  std::vector<std::vector<int>> indices_list(shard_num_);
  std::vector<FeatureLookuperRequest> requests(shard_num_);
  std::vector<FeatureLookuperResponse> responses(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    indices_list[i].clear();
    requests[i].nodes.clear();
  }

  node_feats->clear();
  node_feats->resize(nodes.size());
  neigh_feats->clear();
  neigh_feats->resize(nodes.size());

  // map
  masks.assign(shard_num_, 0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto* node_feat_ptr =
        resource_->cache_storage()->FindNodeFeature(nodes[i]);
    const auto* neigh_feat_ptr =
        resource_->cache_storage()->FindFeature(nodes[i]);
    if (node_feat_ptr != nullptr && neigh_feat_ptr != nullptr) {
      // cache hit ,get node feature and neighbor feature in cache
      (*node_feats)[i].insert((*node_feats)[i].end(), node_feat_ptr->begin(),
                              node_feat_ptr->end());
      (*neigh_feats)[i].insert((*neigh_feats)[i].end(), neigh_feat_ptr->begin(),
                               neigh_feat_ptr->end());
    } else {
      // cache miss, add nodes to request
      int shard_id = ModShard(nodes[i]);
      indices_list[shard_id].emplace_back((int)i);
      requests[shard_id].nodes.emplace_back(nodes[i]);
      masks[shard_id] += 1;
    }
  }

  // rpc
  auto rpc_type = FeatureLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  // reduce : node_feat_list

  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indices = indices_list[i];
      const auto& remote_feats = responses[i].node_feats;
      for (size_t j = 0; j < remote_feats.size(); ++j) {
        const auto& remote_feats_j = remote_feats[j];
        auto& local_feats_j = (*node_feats)[indices[j]];
        local_feats_j.insert(local_feats_j.end(), remote_feats_j.begin(),
                             remote_feats_j.end());
      }
    }
  }

  // reduce : neigh_feat_list
  for (int i = 0; i < shard_num_; ++i) {
    if (masks[i]) {
      const auto& indices = indices_list[i];
      const auto& remote_feat_list = responses[i].neigh_feats;
      for (size_t j = 0; j < remote_feat_list.size(); ++j) {
        const auto& remote_feat_list_j = remote_feat_list[j];
        auto& local_feat_list_j = (*neigh_feats)[indices[j]];
        local_feat_list_j.insert(local_feat_list_j.end(),
                                 remote_feat_list_j.begin(),
                                 remote_feat_list_j.end());
      }
    }
  }

  return true;
}

REGISTER_DIST_GS_OP("FeatureLookuper", DistFeatureLookuper);

}  // namespace graph_op
}  // namespace embedx
