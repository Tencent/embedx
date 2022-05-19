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

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/tcp_connection.h>
#include <deepx_core/ps/tcp_server.h>
#include <gflags/gflags.h>

#include <string>

#include "src/common/data_types.h"
#include "src/tools/graph/graph_flags.h"

namespace embedx {
namespace {

bool CloseConnection(const std::string& ip_port_str) {
  vec_str_t ip_ports;
  deepx_core::Split(ip_port_str, ";", &ip_ports);
  if (ip_ports.empty()) {
    DXERROR("Invalid graph server address: %s.", FLAGS_gs_addrs.c_str());
    return false;
  }

  int retries = 10;
  int second = 1;
  for (const auto& ip_port : ip_ports) {
    deepx_core::TcpEndpoint endpoint = deepx_core::MakeTcpEndpoint(ip_port);
    deepx_core::IoContext io;
    deepx_core::TcpConnection conn(&io);
    if (conn.ConnectRetry(endpoint, retries, second) != 0) {
      DXINFO("Failed to connect graph server: %s.", ip_port.c_str());
      continue;
    }
    DXCHECK_THROW(conn.RpcTerminationNotify() == 0);
    DXINFO("Closed graph server: %s.", ip_port.c_str());
  }
  return true;
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  DXCHECK_THROW(!FLAGS_gs_addrs.empty());
  CloseConnection(FLAGS_gs_addrs);

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
