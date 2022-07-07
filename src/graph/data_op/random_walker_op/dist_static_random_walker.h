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

#pragma once
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/data_op/gs_op.h"
#include "src/graph/proto/random_walker_proto.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {
namespace graph_op {

class DistStaticRandomWalker : public DistGSOp {
 private:
  struct RpcSession {
    std::vector<int> masks;
    std::vector<std::vector<int>> indices_list;
    std::vector<StaticRandomWalkerRequest> requests;
    std::vector<StaticRandomWalkerResponse> responses;

    void Resize(int shard_num) {
      masks.resize(shard_num);
      indices_list.resize(shard_num);
      requests.resize(shard_num);
      responses.resize(shard_num);
    }
  };

 public:
  ~DistStaticRandomWalker() override = default;

 public:
  bool Run(const vec_int_t& cur_nodes, const std::vector<int>& walk_lens,
           const WalkerInfo& walker_info, std::vector<vec_int_t>* seqs);

 private:
  void FillRequest(const vec_int_t& cur_nodes,
                   const std::vector<int>& walk_lens,
                   const WalkerInfo& walker_info,
                   const std::vector<vec_int_t>& seqs,
                   const std::vector<int>& continuous_rpcs,
                   RpcSession* rpc_session);

  bool ParseResponse(const RpcSession& rpc_session,
                     const std::vector<int>& walk_lens,
                     std::vector<vec_int_t>* seqs,
                     std::vector<int>* continuous_rpcs);
};

}  // namespace graph_op
}  // namespace embedx
