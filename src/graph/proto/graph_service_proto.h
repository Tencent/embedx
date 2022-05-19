// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#pragma once
#include <deepx_core/common/stream.h>

#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

constexpr int RPC_TYPE_META_LOOKUPER = 0;
constexpr int RPC_TYPE_SHARED_NEGATIVE_SAMPLER = 1;
constexpr int RPC_TYPE_INDEP_NEGATIVE_SAMPLER = 2;
constexpr int RPC_TYPE_RANDOM_NEIGHBOR_SAMPLER = 3;
constexpr int RPC_TYPE_STATIC_RANDOM_WALKER = 4;
constexpr int RPC_TYPE_FEATURE_LOOKUPER = 5;
constexpr int RPC_TYPE_NODE_FEATURE_LOOKUPER = 6;
constexpr int RPC_TYPE_NODE_CONTEXT_LOOKUPER = 7;
constexpr int RPC_TYPE_NEIGHBOR_FEATURE_LOOKUPER = 8;
constexpr int RPC_TYPE_CACHE_NODE_LOOKUPER = 9;
constexpr int RPC_TYPE_DYNAMIC_RANDOM_WALKER = 10;

using OutputStream = ::deepx_core::OutputStream;
using InputStream = ::deepx_core::InputStream;
/************************************************************************/
/* Meta Lookuper */
/************************************************************************/
struct MetaLookuperRequest {
  std::string key;

  static int rpc_type() noexcept { return RPC_TYPE_META_LOOKUPER; }
};

struct MetaLookuperResponse {
  std::string value;
};

inline OutputStream& operator<<(OutputStream& os,
                                const MetaLookuperRequest& req) {
  os << req.key;
  return os;
}

inline InputStream& operator>>(InputStream& is, MetaLookuperRequest& req) {
  is >> req.key;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const MetaLookuperResponse& resp) {
  os << resp.value;
  return os;
}

inline InputStream& operator>>(InputStream& is, MetaLookuperResponse& resp) {
  is >> resp.value;
  return is;
}

/************************************************************************/
/* Shared Negative Sampling */
/************************************************************************/
struct SharedNegativeSamplerRequest {
  int count;
  vec_int_t nodes;
  vec_int_t excluded_nodes;

  static int rpc_type() noexcept { return RPC_TYPE_SHARED_NEGATIVE_SAMPLER; }
};

struct SharedNegativeSamplerResponse {
  std::vector<vec_int_t> sampled_nodes_list;
};

inline OutputStream& operator<<(OutputStream& os,
                                const SharedNegativeSamplerRequest& req) {
  os << req.count << req.nodes << req.excluded_nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               SharedNegativeSamplerRequest& req) {
  is >> req.count >> req.nodes >> req.excluded_nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const SharedNegativeSamplerResponse& resp) {
  os << resp.sampled_nodes_list;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               SharedNegativeSamplerResponse& resp) {
  is >> resp.sampled_nodes_list;
  return is;
}

/************************************************************************/
/* Indepdent Negative Sampling */
/************************************************************************/
struct IndepNegativeSamplerRequest {
  int count;
  vec_int_t nodes;
  vec_int_t excluded_nodes;

  static int rpc_type() noexcept { return RPC_TYPE_INDEP_NEGATIVE_SAMPLER; }
};

struct IndepNegativeSamplerResponse {
  std::vector<vec_int_t> sampled_nodes_list;
};

inline OutputStream& operator<<(OutputStream& os,
                                const IndepNegativeSamplerRequest& req) {
  os << req.count << req.nodes << req.excluded_nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               IndepNegativeSamplerRequest& req) {
  is >> req.count >> req.nodes >> req.excluded_nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const IndepNegativeSamplerResponse& resp) {
  os << resp.sampled_nodes_list;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               IndepNegativeSamplerResponse& resp) {
  is >> resp.sampled_nodes_list;
  return is;
}

/************************************************************************/
/* Random Neighbor Sampling  */
/************************************************************************/
struct RandomNeighborSamplerRequest {
  int count;
  vec_int_t nodes;

  static int rpc_type() noexcept { return RPC_TYPE_RANDOM_NEIGHBOR_SAMPLER; }
};

struct RandomNeighborSamplerResponse {
  std::vector<vec_int_t> neighbor_nodes_list;
};

inline OutputStream& operator<<(OutputStream& os,
                                const RandomNeighborSamplerRequest& req) {
  os << req.count << req.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               RandomNeighborSamplerRequest& req) {
  is >> req.count >> req.nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const RandomNeighborSamplerResponse& resp) {
  os << resp.neighbor_nodes_list;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               RandomNeighborSamplerResponse& resp) {
  is >> resp.neighbor_nodes_list;
  return is;
}

/************************************************************************/
/* Feature Lookuper */
/************************************************************************/
struct FeatureLookuperRequest {
  vec_int_t nodes;

  static int rpc_type() noexcept { return RPC_TYPE_FEATURE_LOOKUPER; }
};

struct FeatureLookuperResponse {
  std::vector<vec_pair_t> node_feats;
  std::vector<vec_pair_t> neigh_feats;
};

inline OutputStream& operator<<(OutputStream& os,
                                const FeatureLookuperRequest& req) {
  os << req.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is, FeatureLookuperRequest& req) {
  is >> req.nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const FeatureLookuperResponse& resp) {
  os << resp.node_feats << resp.neigh_feats;
  return os;
}

inline InputStream& operator>>(InputStream& is, FeatureLookuperResponse& resp) {
  is >> resp.node_feats >> resp.neigh_feats;
  return is;
}

/************************************************************************/
/* Node Feature Lookuper */
/************************************************************************/
struct NodeFeatureLookuperRequest {
  vec_int_t nodes;
  static int rpc_type() noexcept { return RPC_TYPE_NODE_FEATURE_LOOKUPER; }
};

struct NodeFeatureLookuperResponse {
  std::vector<vec_pair_t> node_feats;
};

inline OutputStream& operator<<(OutputStream& os,
                                const NodeFeatureLookuperRequest& req) {
  os << req.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               NodeFeatureLookuperRequest& req) {
  is >> req.nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const NodeFeatureLookuperResponse& resp) {
  os << resp.node_feats;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               NodeFeatureLookuperResponse& resp) {
  is >> resp.node_feats;
  return is;
}

/************************************************************************/
/* Neighbor Feature Lookuper */
/************************************************************************/
struct NeighborFeatureLookuperRequest {
  vec_int_t nodes;
  static int rpc_type() noexcept { return RPC_TYPE_NEIGHBOR_FEATURE_LOOKUPER; }
};

struct NeighborFeatureLookuperResponse {
  std::vector<vec_pair_t> neigh_feats;
};

inline OutputStream& operator<<(OutputStream& os,
                                const NeighborFeatureLookuperRequest& req) {
  os << req.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               NeighborFeatureLookuperRequest& req) {
  is >> req.nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const NeighborFeatureLookuperResponse& resp) {
  os << resp.neigh_feats;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               NeighborFeatureLookuperResponse& resp) {
  is >> resp.neigh_feats;
  return is;
}

/************************************************************************/
/* Node Context Lookuper*/
/************************************************************************/
struct ContextLookuperRequest {
  vec_int_t nodes;

  static int rpc_type() noexcept { return RPC_TYPE_NODE_CONTEXT_LOOKUPER; }
};

struct ContextLookuperResponse {
  std::vector<vec_pair_t> contexts;
};

inline OutputStream& operator<<(OutputStream& os,
                                const ContextLookuperRequest& req) {
  os << req.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is, ContextLookuperRequest& req) {
  is >> req.nodes;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const ContextLookuperResponse& resp) {
  os << resp.contexts;
  return os;
}

inline InputStream& operator>>(InputStream& is, ContextLookuperResponse& resp) {
  is >> resp.contexts;
  return is;
}

/************************************************************************/
/* Node Cacher*/
/************************************************************************/
struct CacheNodeLookuperRequest {
  int cursor;
  int count;

  static int rpc_type() noexcept { return RPC_TYPE_CACHE_NODE_LOOKUPER; }
};

struct CacheNodeLookuperResponse {
  vec_int_t nodes;
};

inline OutputStream& operator<<(OutputStream& os,
                                const CacheNodeLookuperRequest& req) {
  os << req.cursor << req.count;
  return os;
}

inline InputStream& operator>>(InputStream& is, CacheNodeLookuperRequest& req) {
  is >> req.cursor >> req.count;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const CacheNodeLookuperResponse& resp) {
  os << resp.nodes;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               CacheNodeLookuperResponse& resp) {
  is >> resp.nodes;
  return is;
}

}  // namespace embedx
