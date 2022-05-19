// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//

#pragma once
#include <deepx_core/common/stream.h>

#include <vector>

#include "src/common/data_types.h"
#include "src/graph/proto/graph_service_proto.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {

/************************************************************************/
/* Static Random walker */
/************************************************************************/
struct StaticRandomWalkerRequest {
  vec_int_t cur_nodes;
  std::vector<int> walk_lens;
  WalkerInfo walker_info;

  static int rpc_type() noexcept { return RPC_TYPE_STATIC_RANDOM_WALKER; }
};

struct StaticRandomWalkerResponse {
  std::vector<vec_int_t> seqs;
};

inline OutputStream& operator<<(OutputStream& os,
                                const StaticRandomWalkerRequest& req) {
  os << req.cur_nodes << req.walk_lens << req.walker_info.meta_path
     << req.walker_info.walker_length << req.walker_info.prev_info.nodes
     << req.walker_info.prev_info.contexts;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               StaticRandomWalkerRequest& req) {
  is >> req.cur_nodes >> req.walk_lens >> req.walker_info.meta_path >>
      req.walker_info.walker_length >> req.walker_info.prev_info.nodes >>
      req.walker_info.prev_info.contexts;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const StaticRandomWalkerResponse& resp) {
  os << resp.seqs;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               StaticRandomWalkerResponse& resp) {
  is >> resp.seqs;
  return is;
}

/************************************************************************/
/* Dynamic Random walker */
/************************************************************************/
struct DynamicRandomWalkerRequest {
  vec_int_t cur_nodes;
  std::vector<int> walk_lens;
  WalkerInfo walker_info;

  static int rpc_type() noexcept { return RPC_TYPE_DYNAMIC_RANDOM_WALKER; }
};

struct DynamicRandomWalkerResponse {
  std::vector<vec_int_t> seqs;
  PrevInfo prev_info;
};

inline OutputStream& operator<<(OutputStream& os,
                                const DynamicRandomWalkerRequest& req) {
  os << req.cur_nodes << req.walk_lens << req.walker_info.meta_path
     << req.walker_info.walker_length << req.walker_info.prev_info.nodes
     << req.walker_info.prev_info.contexts;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               DynamicRandomWalkerRequest& req) {
  is >> req.cur_nodes >> req.walk_lens >> req.walker_info.meta_path >>
      req.walker_info.walker_length >> req.walker_info.prev_info.nodes >>
      req.walker_info.prev_info.contexts;
  return is;
}

inline OutputStream& operator<<(OutputStream& os,
                                const DynamicRandomWalkerResponse& resp) {
  os << resp.seqs << resp.prev_info.nodes << resp.prev_info.contexts;
  return os;
}

inline InputStream& operator>>(InputStream& is,
                               DynamicRandomWalkerResponse& resp) {
  is >> resp.seqs >> resp.prev_info.nodes >> resp.prev_info.contexts;
  return is;
}

}  // namespace embedx
