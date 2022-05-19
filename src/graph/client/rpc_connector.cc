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

#include "src/graph/client/rpc_connector.h"

namespace embedx {

std::unique_ptr<RpcConnector> NewRpcConnector() {
  std::unique_ptr<RpcConnector> rpc_connector;
  rpc_connector.reset(new RpcConnector());
  return rpc_connector;
}

}  // namespace embedx
