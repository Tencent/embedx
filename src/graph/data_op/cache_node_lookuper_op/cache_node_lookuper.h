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
#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"
#include "src/graph/cache/cache_node_builder.h"
#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

class CacheNodeLookuper : public LocalGSOp {
 private:
  std::unique_ptr<CacheNodeBuilder> builder_;

 public:
  ~CacheNodeLookuper() override = default;

 public:
  bool Run(int cursor, int count, vec_int_t* nodes) const;
  int HandleRpc(const CacheNodeLookuperRequest& req,
                CacheNodeLookuperResponse* resp);

 private:
  bool Init(const LocalGSOpResource* resource) override {
    builder_ = CacheNodeBuilder::Create(resource->graph(),
                                        resource->graph_config().cache_type(),
                                        resource->graph_config().cache_thld(),
                                        resource->graph_config().thread_num());
    return builder_ != nullptr;
  }
};

}  // namespace graph_op
}  // namespace embedx
