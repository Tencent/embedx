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

#pragma once

#include "src/common/data_types.h"
#include "src/graph/cache/cache_storage.h"
#include "src/graph/data_op/gs_op.h"

namespace embedx {
namespace graph_op {

class DistCacheStorageLookuper : public DistGSOp {
 public:
  ~DistCacheStorageLookuper() override = default;

 public:
  bool Run(CacheStorage* cache_storage) const;

 private:
  int InitMaxNodePerRpc() const;
  bool LookupNode(int max_node_num, vec_int_t* nodes) const;
  bool LookupFeature(const vec_int_t& nodes, int max_node_num,
                     adj_list_t* node_feat_map, adj_list_t* feat_map) const;
  bool LookupContext(const vec_int_t& nodes, int max_node_num,
                     adj_list_t* context_map) const;
};

}  // namespace graph_op
}  // namespace embedx
