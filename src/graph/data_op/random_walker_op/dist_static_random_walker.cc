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

#include "src/graph/data_op/random_walker_op/dist_static_random_walker.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {
namespace {

bool FinishRpc(const std::vector<int>& continuous_rpcs) {
  int sum = 0;
  for (auto& v : continuous_rpcs) {
    sum += v;
  }
  return sum == 0;
}

}  // namespace

bool DistStaticRandomWalker::Run(const vec_int_t& cur_nodes,
                                 const std::vector<int>& walk_lens,
                                 const WalkerInfo& /*walker_info*/,
                                 std::vector<vec_int_t>* seqs) {
  // prepare
  seqs->clear();
  seqs->resize(cur_nodes.size());
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    (*seqs)[i].reserve((size_t)walk_lens[i]);
    (*seqs)[i].emplace_back(cur_nodes[i]);
  }

  std::vector<int> continuous_rpcs(cur_nodes.size(), 1);
  RpcSession rpc_session;
  while (!FinishRpc(continuous_rpcs)) {
    FillRequest(cur_nodes, walk_lens, *seqs, continuous_rpcs, &rpc_session);

    // call rpc.
    auto rpc_type = StaticRandomWalkerRequest::rpc_type();
    if (WriteRequestReadResponse(conns_, rpc_type, rpc_session.requests,
                                 &rpc_session.responses,
                                 &rpc_session.masks) != 0) {
      return false;
    }

    if (!ParseResponse(rpc_session, walk_lens, seqs, &continuous_rpcs)) {
      return false;
    }
  }

  return true;
}

void DistStaticRandomWalker::FillRequest(
    const vec_int_t& cur_nodes, const std::vector<int>& walk_lens,
    const std::vector<vec_int_t>& seqs, const std::vector<int>& continuous_rpcs,
    RpcSession* rpc_session) {
  // prepare
  rpc_session->Resize(shard_num_);
  for (int i = 0; i < shard_num_; ++i) {
    rpc_session->masks[i] = 0;
    rpc_session->indices_list[i].clear();
    rpc_session->requests[i].cur_nodes.clear();
    rpc_session->requests[i].walk_lens.clear();
    rpc_session->responses[i].seqs.clear();
  }

  // fill
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    if (!continuous_rpcs[i]) {
      continue;
    }

    const auto& cur_seq = seqs[i];
    if (cur_seq.size() >= (size_t)walk_lens[i]) {
      DXTHROW_INVALID_ARGUMENT(
          "Seqs[%zu] size: %zu must be less than walk_lens[%zu]: %d.", i,
          cur_seq.size(), i, walk_lens[i]);
    }

    int_t cur_node = cur_seq.empty() ? cur_nodes[i] : cur_seq.back();
    int shard_id = ModShard(cur_node);
    rpc_session->masks[shard_id] += 1;
    rpc_session->indices_list[shard_id].emplace_back((int)i);
    rpc_session->requests[shard_id].cur_nodes.emplace_back(cur_node);
    rpc_session->requests[shard_id].walk_lens.emplace_back(walk_lens[i] -
                                                           (int)cur_seq.size());
  }
}

bool DistStaticRandomWalker::ParseResponse(const RpcSession& rpc_session,
                                           const std::vector<int>& walk_lens,
                                           std::vector<vec_int_t>* seqs,
                                           std::vector<int>* continuous_rpcs) {
  if (rpc_session.masks.size() != (size_t)shard_num_) {
    DXERROR("Inconsistent rpc_session.masks.size(): %zu vs %d.",
            rpc_session.masks.size(), shard_num_);
    return false;
  }
  for (int i = 0; i < shard_num_; ++i) {
    if (!rpc_session.masks[i]) {
      continue;
    }

    const auto& cur_indices = rpc_session.indices_list[i];
    const auto& remote_seqs = rpc_session.responses[i].seqs;
    if (cur_indices.size() != remote_seqs.size()) {
      DXERROR(
          "Need cur_indices.size() == remote_seqs.size(), Got "
          "cur_indices.size: %zu vs remote_seqs.size: %zu",
          cur_indices.size(), remote_seqs.size());
      return false;
    }

    for (size_t j = 0; j < cur_indices.size(); ++j) {
      size_t origin_idx = cur_indices[j];

      auto& cur_seq = (*seqs)[origin_idx];
      const auto& remote_seq = remote_seqs[j];
      cur_seq.insert(cur_seq.end(), remote_seq.begin(), remote_seq.end());

      if (cur_seq.size() == 1u) {
        if (remote_seq.empty()) {
          DXERROR("Please check Isolated node: %" PRIu64, cur_seq[0]);
          return false;
        }
      }

      // normal exit and stopping rpc
      if (cur_seq.size() >= (size_t)walk_lens[origin_idx]) {
        (*continuous_rpcs)[origin_idx] = 0;
      }
    }
  }
  return true;
}

REGISTER_DIST_GS_OP("StaticRandomWalker", DistStaticRandomWalker);

}  // namespace graph_op
}  // namespace embedx
