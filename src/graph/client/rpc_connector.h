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
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/tcp_connection.h>

#include <memory>  // std::unique_ptr
#include <vector>

namespace embedx {

class RpcConnector {
 private:
  std::unique_ptr<deepx_core::IoContext> io_;
  std::unique_ptr<deepx_core::TcpConnections> conns_;

 public:
  RpcConnector() = default;
  ~RpcConnector() = default;

 public:
  deepx_core::TcpConnections* conns() { return conns_.get(); }

 public:
  bool Connect(const std::vector<deepx_core::TcpEndpoint>& endpoints) {
    if (endpoints.empty()) {
      DXERROR("Please set ip_ports first.");
      return false;
    }

    Close();
    io_.reset(new deepx_core::IoContext);
    conns_.reset(new deepx_core::TcpConnections(io_.get()));
    return conns_->ConnectRetry(endpoints) == 0;
  }

  void Close() {
    if (conns_) {
      conns_->Close();
      conns_.reset();
    }
    if (io_) {
      io_.reset();
    }
  }
};

std::unique_ptr<RpcConnector> NewRpcConnector();

}  // namespace embedx
