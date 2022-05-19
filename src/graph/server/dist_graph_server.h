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
#include <deepx_core/ps/rpc_server.h>

#include <memory>  // std::unique_ptr

#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {

class DistGraphServer {
 private:
  std::unique_ptr<graph_op::LocalGSOpResource> resource_;
  deepx_core::RpcServer rpc_server_;

 public:
  bool Start(const GraphConfig& config);

 private:
  bool InitGraphServer(const GraphConfig& config);
  bool InitRpcServer(const GraphConfig& config);
  void RegisterRequestHandler();

 private:
#define DECLARE_REQUEST_HANDLER(Name) void Name()
  DECLARE_REQUEST_HANDLER(MetaLookuper);
  DECLARE_REQUEST_HANDLER(FeatureLookuper);
  DECLARE_REQUEST_HANDLER(NodeFeatureLookuper);
  DECLARE_REQUEST_HANDLER(NeighborFeatureLookuper);
  DECLARE_REQUEST_HANDLER(ContextLookuper);
  DECLARE_REQUEST_HANDLER(RandomNeighborSampler);
  DECLARE_REQUEST_HANDLER(SharedNegativeSampler);
  DECLARE_REQUEST_HANDLER(IndepNegativeSampler);
  DECLARE_REQUEST_HANDLER(StaticRandomWalker);
  DECLARE_REQUEST_HANDLER(CacheNodeLookuper);

#undef DECLARE_REQUEST_HANDLER
};

}  // namespace embedx
