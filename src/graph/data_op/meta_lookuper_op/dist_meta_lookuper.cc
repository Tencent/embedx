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

#include "src/graph/data_op/meta_lookuper_op/dist_meta_lookuper.h"

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/data_op/rpc_key.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {
namespace {

using ::embedx::rpc_key::NODE_FREQ;

}  // namespace

bool DistMetaLookuper::Run(std::vector<vec_int_t>* node_freqs_list) const {
  // prepare
  std::vector<MetaLookuperRequest> requests(shard_num_);
  std::vector<MetaLookuperResponse> responses(shard_num_);
  for (int i = 0; i < shard_num_; ++i) {
    requests[i].key = NODE_FREQ;
  }

  // rpc
  auto rpc_type = MetaLookuperRequest::rpc_type();
  if (WriteRequestReadResponse(conns_, rpc_type, requests, &responses) != 0) {
    return false;
  }

  // init
  vec_str_t node_freq_strs;
  deepx_core::Split(responses[0].value, ",", &node_freq_strs);
  auto ns_size = (int)node_freq_strs.size();

  // [ns][shard]
  node_freqs_list->clear();
  node_freqs_list->resize(ns_size);
  for (auto& node_freqs : *node_freqs_list) {
    node_freqs.resize(shard_num_);
  }

  // get the node freq of each namespace of each machine
  for (int i = 0; i < shard_num_; ++i) {
    deepx_core::Split(responses[i].value, ",", &node_freq_strs);
    DXCHECK(node_freq_strs.size() == (size_t)ns_size);
    for (int j = 0; j < ns_size; ++j) {
      auto freq = (int_t)std::stoull(node_freq_strs[j]);
      (*node_freqs_list)[j][i] = freq;
    }
  }

  return true;
}

REGISTER_DIST_GS_OP("DistMetaLookuper", DistMetaLookuper);

}  // namespace graph_op
}  // namespace embedx
