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

#include "src/graph/client/graph_client.h"

#include <deepx_core/dx_log.h>

#include <utility>  // std::move

#include "src/graph/client/graph_client_impl.h"

namespace embedx {

GraphClient::GraphClient(std::unique_ptr<GraphClientImpl>&& impl) {
  impl_ = std::move(impl);
}

GraphClient::~GraphClient() {}

bool GraphClient::SharedSampleNegative(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  return impl_->SharedSampleNegative(count, nodes, excluded_nodes,
                                     sampled_nodes_list);
}

bool GraphClient::IndepSampleNegative(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  return impl_->IndepSampleNegative(count, nodes, excluded_nodes,
                                    sampled_nodes_list);
}

bool GraphClient::StaticTraverse(const vec_int_t& cur_nodes,
                                 const std::vector<int>& walk_lens,
                                 const WalkerInfo& walker_info,
                                 std::vector<vec_int_t>* seqs) const {
  return impl_->StaticTraverse(cur_nodes, walk_lens, walker_info, seqs);
}

bool GraphClient::RandomSampleNeighbor(
    int count, const vec_int_t& nodes,
    std::vector<vec_int_t>* neighbor_nodes_list) const {
  return impl_->RandomSampleNeighbor(count, nodes, neighbor_nodes_list);
}

bool GraphClient::LookupFeature(const vec_int_t& nodes,
                                std::vector<vec_pair_t>* node_feats,
                                std::vector<vec_pair_t>* neigh_feats) const {
  return impl_->LookupFeature(nodes, node_feats, neigh_feats);
}

bool GraphClient::LookupNodeFeature(const vec_int_t& nodes,
                                    std::vector<vec_pair_t>* node_feats) const {
  return impl_->LookupNodeFeature(nodes, node_feats);
}

bool GraphClient::LookupNeighborFeature(
    const vec_int_t& nodes, std::vector<vec_pair_t>* neigh_feats) const {
  return impl_->LookupNeighborFeature(nodes, neigh_feats);
}

bool GraphClient::LookupContext(const vec_int_t& nodes,
                                std::vector<vec_pair_t>* contexts) const {
  return impl_->LookupContext(nodes, contexts);
}

std::unique_ptr<GraphClient> NewGraphClient(const GraphConfig& config,
                                            GraphClientEnum type) {
  std::unique_ptr<GraphClient> graph_client;
  switch (type) {
    case GraphClientEnum::LOCAL:
      graph_client.reset(new GraphClient(NewLocalGraphClientImpl(config)));
      break;
    case GraphClientEnum::DIST:
      graph_client.reset(new GraphClient(NewDistGraphClientImpl(config)));
      break;
    default:
      DXERROR("Need type: LOCAL(0) || DIST(1), got type: %d.", (int)type);
      break;
  }
  return graph_client;
}

}  // namespace embedx
