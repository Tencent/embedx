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

#include "src/graph/data_op/cache_node_lookuper_op/cache_node_lookuper.h"

#include <deepx_core/dx_log.h>

#include "src/graph/data_op/gs_op_registry.h"

namespace embedx {
namespace graph_op {

bool CacheNodeLookuper::Run(int cursor, int count, vec_int_t* nodes) const {
  nodes->clear();

  const auto* cached_nodes = builder_->nodes();
  if (cursor > (int)cached_nodes->size()) {
    DXERROR(
        "Need cursor <= cached_node_size, got cursor :%d vs cached_node_size: "
        "%zu.",
        cursor, cached_nodes->size());
    return false;
  }

  for (auto i = (size_t)cursor; i < cached_nodes->size(); ++i) {
    nodes->emplace_back((*cached_nodes)[i]);
    if (nodes->size() == (size_t)count) {
      break;
    }
  }

  return nodes->size() <= (size_t)count;
}

int CacheNodeLookuper::HandleRpc(const CacheNodeLookuperRequest& req,
                                 CacheNodeLookuperResponse* resp) {
  if (!Run(req.cursor, req.count, &resp->nodes)) {
    return -1;
  }
  return 0;
}

REGISTER_LOCAL_GS_OP("CacheNodeLookuper", CacheNodeLookuper);

}  // namespace graph_op
}  // namespace embedx
