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

#include "src/graph/data_op/meta_lookuper_op/meta_lookuper.h"

#include <deepx_core/dx_log.h>

#include <sstream>  // std::stringstream

#include "src/common/data_types.h"
#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/data_op/rpc_key.h"

namespace embedx {
namespace graph_op {
namespace {

std::string VecToString(const vec_int_t& vec) {
  std::stringstream ss;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) {
      ss << ",";
    }
    ss << vec[i];
  }
  return ss.str();
}

using ::embedx::rpc_key::MAX_NODE_PER_RPC;
using ::embedx::rpc_key::NODE_FREQ;

}  // namespace

bool MetaLookuper::Run(const std::string& key, std::string* value) const {
  if (key == NODE_FREQ) {
    const auto& total_freqs = graph_->total_freqs();
    DXCHECK(!total_freqs.empty());
    *value = VecToString(total_freqs);
  } else if (key == MAX_NODE_PER_RPC) {
    *value = std::to_string(max_node_per_rpc_);
  } else {
    DXERROR("Only support key: '%s' || '%s'.", NODE_FREQ.c_str(),
            MAX_NODE_PER_RPC.c_str());
    return false;
  }

  return true;
}

int MetaLookuper::HandleRpc(const MetaLookuperRequest& req,
                            MetaLookuperResponse* resp) const {
  if (!Run(req.key, &resp->value)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("MetaLookuper", MetaLookuper);

}  // namespace graph_op
}  // namespace embedx
