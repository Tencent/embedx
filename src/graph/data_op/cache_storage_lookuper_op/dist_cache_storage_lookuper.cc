// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanqing Guo (yuanqingsunny1180@gmail.com)
//

#include "src/graph/data_op/cache_storage_lookuper_op/dist_cache_storage_lookuper.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64
#include <string>     //std::stoll
#include <vector>

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/data_op/rpc_key.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {
namespace {

using ::embedx::rpc_key::MAX_NODE_PER_RPC;

}  // namespace

int DistCacheStorageLookuper::InitMaxNodePerRpc() const {
  std::vector<MetaLookuperRequest> reqs(shard_num_);
  std::vector<MetaLookuperResponse> resps(shard_num_);

  for (int i = 0; i < shard_num_; ++i) {
    reqs[i].key = MAX_NODE_PER_RPC;
  }

  auto rpc_type = MetaLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, reqs, &resps) != 0) {
    return false;
  }

  int max_node_num = std::stoll(resps[0].value);
  DXCHECK(max_node_num > 0);
  for (int i = 1; i < shard_num_; ++i) {
    DXCHECK(max_node_num == std::stoll(resps[i].value));
  }

  return max_node_num;
}

// The use of max_node_num:
// Commonly there are hundreds of workers at the same time. It will cause
// excessive memory usage in graph server if each worker has a huge request
// volume. The variable(max_node_num) is used to limit the workers' request
// volume.
bool DistCacheStorageLookuper::LookupNode(int max_node_num,
                                          vec_int_t* nodes) const {
  nodes->clear();

  std::vector<CacheNodeLookuperRequest> reqs(shard_num_);
  std::vector<CacheNodeLookuperResponse> resps(shard_num_);
  std::vector<int> masks(shard_num_, 0);

  for (int i = 0; i < shard_num_; ++i) {
    reqs[i].cursor = -max_node_num;
    reqs[i].count = max_node_num;
    masks[i] += 1;
  }

  while (true) {
    for (int i = 0; i < shard_num_; ++i) {
      reqs[i].cursor += max_node_num;
    }

    // rpc
    if (WriteRequestReadResponse(conns_, RPC_TYPE_CACHE_NODE_LOOKUPER, reqs,
                                 &resps, &masks) != 0) {
      return false;
    }

    bool lookup_all_nodes = true;
    for (int i = 0; i < shard_num_; ++i) {
      nodes->insert(nodes->end(), resps[i].nodes.begin(), resps[i].nodes.end());

      if (resps[i].nodes.size() == (size_t)max_node_num) {
        lookup_all_nodes = false;
      } else {
        masks[i] = 0;
      }
    }

    if (lookup_all_nodes) {
      break;
    }
  }
  return true;
}

bool DistCacheStorageLookuper::LookupFeature(const vec_int_t& nodes,
                                             int max_node_num,
                                             adj_list_t* node_feature_map,
                                             adj_list_t* feature_map) const {
  node_feature_map->clear();
  feature_map->clear();

  std::vector<FeatureLookuperRequest> reqs(shard_num_);
  std::vector<FeatureLookuperResponse> resps(shard_num_);
  std::vector<int> masks;

  size_t node_begin = 0;
  while (node_begin < nodes.size()) {
    // prepare requests
    masks.assign(shard_num_, 0);
    for (int i = 0; i < shard_num_; ++i) {
      reqs[i].nodes.clear();
    }

    // requests append nodes
    // from nodes[node_begin] to nodes[node_begin + max_node_num]
    for (size_t i = node_begin;
         i < nodes.size() && i < node_begin + max_node_num; ++i) {
      int shard_id = ModShard(nodes[i]);
      reqs[shard_id].nodes.emplace_back(nodes[i]);
      masks[shard_id] += 1;
      ++node_begin;
    }

    // rpc
    if (WriteRequestReadResponse(conns_, RPC_TYPE_FEATURE_LOOKUPER, reqs,
                                 &resps, &masks) != 0) {
      return false;
    }

    // merge node features & neigh features
    for (int i = 0; i < shard_num_; ++i) {
      if (masks[i]) {
        const auto& ith_node_feats = resps[i].node_feats;
        for (size_t j = 0; j < ith_node_feats.size(); ++j) {
          const auto& key = reqs[i].nodes[j];
          (*node_feature_map)[key] = ith_node_feats[j];
        }
      }
    }

    for (int i = 0; i < shard_num_; ++i) {
      if (masks[i]) {
        const auto& ith_neigh_feats = resps[i].neigh_feats;
        for (size_t j = 0; j < ith_neigh_feats.size(); ++j) {
          const auto& key = reqs[i].nodes[j];
          (*feature_map)[key] = ith_neigh_feats[j];
        }
      }
    }
  }

  return true;
}

bool DistCacheStorageLookuper::LookupContext(const vec_int_t& nodes,
                                             int max_node_num,
                                             adj_list_t* context_map) const {
  context_map->clear();

  std::vector<ContextLookuperRequest> reqs(shard_num_);
  std::vector<ContextLookuperResponse> resps(shard_num_);
  std::vector<int> masks;

  size_t node_begin = 0;
  while (node_begin < nodes.size()) {
    // prepare
    masks.assign(shard_num_, 0);
    for (int i = 0; i < shard_num_; ++i) {
      reqs[i].nodes.clear();
    }

    // requests append nodes
    // from nodes[node_begin] to nodes[node_begin + max_node_num]
    for (size_t i = node_begin;
         i < nodes.size() && i < node_begin + max_node_num; ++i) {
      int shard_id = ModShard(nodes[i]);
      reqs[shard_id].nodes.emplace_back(nodes[i]);
      masks[shard_id] += 1;
      ++node_begin;
    }

    // rpc
    if (WriteRequestReadResponse(conns_, RPC_TYPE_NODE_CONTEXT_LOOKUPER, reqs,
                                 &resps, &masks) != 0) {
      return false;
    }

    // merge node context
    for (int i = 0; i < shard_num_; ++i) {
      if (masks[i]) {
        const auto& ith_contexts = resps[i].contexts;
        for (size_t j = 0; j < ith_contexts.size(); ++j) {
          const auto& key = reqs[i].nodes[j];
          (*context_map)[key] = ith_contexts[j];
        }
      }
    }
  }

  return true;
}

bool DistCacheStorageLookuper::Run(CacheStorage* cache_storage) const {
  auto begin = std::chrono::steady_clock::now();

  int max_node_num = InitMaxNodePerRpc();

  vec_int_t nodes;
  if (!LookupNode(max_node_num, &nodes)) {
    DXERROR("Failed to lookup and fill node.");
    return false;
  }

  if (!LookupContext(nodes, max_node_num, cache_storage->context_map())) {
    DXERROR("Failed to lookup and fill context.");
    return false;
  }

  if (!LookupFeature(nodes, max_node_num, cache_storage->node_feat_map(),
                     cache_storage->feat_map())) {
    DXERROR("Failed to lookup and fill feature.");
    return false;
  }
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

  DXINFO("Got cached nodes info end, total node: %d, total duration: %" PRIu64,
         (int)nodes.size(), duration.count());

  return true;
}

REGISTER_DIST_GS_OP("DistCacheStorageLookuper", DistCacheStorageLookuper);

}  // namespace graph_op
}  // namespace embedx
