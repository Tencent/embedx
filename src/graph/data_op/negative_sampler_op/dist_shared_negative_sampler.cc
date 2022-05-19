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

#include "src/graph/data_op/negative_sampler_op/dist_shared_negative_sampler.h"

#include <unordered_set>

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/proto/graph_service_proto.h"
#include "src/io/io_util.h"

namespace embedx {
namespace graph_op {

bool DistSharedNegativeSampler::Run(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  // prepare
  std::vector<int> masks;
  std::vector<SharedNegativeSamplerRequest> requests(shard_num_);
  std::vector<SharedNegativeSamplerResponse> responses(shard_num_);
  for (int i = 0; i < shard_num_; ++i) {  // NOLINT
    requests[i].count = 0;
    requests[i].nodes = nodes;
    requests[i].excluded_nodes = excluded_nodes;
  }

  // map
  masks.assign(shard_num_, 0);
  for (int i = 0; i < count; ++i) {
    int shard_id = resource_->sampling()->Next();
    ++requests[shard_id].count;
    masks[shard_id] += 1;
  }

  // rpc
  auto rpc_type = SharedNegativeSamplerRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses,
                               &masks) != 0) {
    return false;
  }

  int ns_size = resource_->ns_size();
  // reduce
  sampled_nodes_list->clear();
  sampled_nodes_list->resize(ns_size);

  std::unordered_set<uint16_t> ns_id_set;
  io_util::ParseMaxNodeType(ns_size, nodes, &ns_id_set);

  for (auto ns_id : ns_id_set) {
    for (int i = 0; i < shard_num_; ++i) {
      if (masks[i]) {
        const auto& remote_nodes_list = responses[i].sampled_nodes_list;
        if (!remote_nodes_list.empty()) {
          (*sampled_nodes_list)[ns_id].insert(
              (*sampled_nodes_list)[ns_id].end(),
              remote_nodes_list[ns_id].begin(), remote_nodes_list[ns_id].end());
        }
      }
    }
  }

  return true;
}

REGISTER_DIST_GS_OP("SharedNegativeSampler", DistSharedNegativeSampler);

}  // namespace graph_op
}  // namespace embedx
