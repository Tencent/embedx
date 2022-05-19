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
#include <string>

#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/in_memory_graph.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

class MetaLookuper : public LocalGSOp {
 private:
  const InMemoryGraph* graph_ = nullptr;
  int max_node_per_rpc_ = 0;

 public:
  ~MetaLookuper() override = default;

 public:
  bool Run(const std::string& key, std::string* value) const;
  int HandleRpc(const MetaLookuperRequest& req,
                MetaLookuperResponse* resp) const;

 private:
  bool Init(const LocalGSOpResource* resource) override {
    graph_ = resource->graph();
    max_node_per_rpc_ = resource->graph_config().max_node_per_rpc();
    return true;
  }
};

}  // namespace graph_op
}  // namespace embedx
