// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//         Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/rpc_client.h>
#include <deepx_core/ps/rpc_server.h>
#include <deepx_core/ps/tcp_connection.h>
#include <deepx_core/ps/tcp_server.h>
#include <gflags/gflags.h>

#include <algorithm>  // std::sort, ...
#include <chrono>
#include <cstdint>
#include <cstdlib>  // getenv
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace embedx {

using namespace deepx_core;  // NOLINT

namespace {

constexpr int RPC_TYPE_SERVER_ADDR_REQUEST = 0x04a1e6c3;  // magic number

template <typename T>
bool _getenv(const char* name, T* value) {
  const char* env = getenv(name);
  if (env == nullptr) {
    return false;
  }

  DXINFO("%s=%s", name, env);
  std::istringstream is(env);
  is >> *value;
  return is && is.eof();
}

/************************************************************************/
/* ServerAddrRequest */
/************************************************************************/
struct ServerAddrRequest {
  // 1, from server
  // 0, from worker
  int is_server = 0;
  int server_id = 0;         // only for server
  uint16_t server_port = 0;  // only for server
};

OutputStream& operator<<(OutputStream& os, const ServerAddrRequest& request) {
  os << request.is_server << request.server_id << request.server_port;
  return os;
}

InputStream& operator>>(InputStream& is, ServerAddrRequest& request) {
  is >> request.is_server >> request.server_id >> request.server_port;
  return is;
}

/************************************************************************/
/* ServerAddrResponse */
/************************************************************************/
struct ServerAddrResponse {
  int server_id = 0;  // only for server
  std::vector<std::string> server_addrs;
};

OutputStream& operator<<(OutputStream& os, const ServerAddrResponse& response) {
  os << response.server_id << response.server_addrs;
  return os;
}

InputStream& operator>>(InputStream& is, ServerAddrResponse& response) {
  is >> response.server_id >> response.server_addrs;
  return is;
}

/************************************************************************/
/* ServerItem */
/************************************************************************/
class ServerItem {
 private:
  int id_ = 0;
  TcpEndpoint addr_;

 public:
  void set_id(int id) noexcept { id_ = id; }
  int id() const noexcept { return id_; }
  const TcpEndpoint& addr() const noexcept { return addr_; }
  std::string addr_str() const { return to_string(addr_); }

 public:
  ServerItem() = default;
  ServerItem(int id, TcpEndpoint addr) : id_(id), addr_(std::move(addr)) {}
};

/************************************************************************/
/* ServerItems */
/************************************************************************/
class ServerItems : public std::vector<ServerItem> {
 public:
  const_iterator find(const TcpEndpoint& addr) const noexcept {
    return std::find_if(begin(), end(), [&addr](const ServerItem& server_item) {
      return server_item.addr() == addr;
    });
  }

  int GetId(const TcpEndpoint& addr) const noexcept {
    auto it = find(addr);
    if (it != end()) {
      return it->id();
    }
    return -1;
  }

  std::vector<std::string> GetAddrs() const {
    std::vector<std::string> server_addrs(size());
    for (size_t i = 0; i < size(); ++i) {
      server_addrs[i] = (*this)[i].addr_str();
    }
    return server_addrs;
  }

  void AllocId() {
    int has_id = 1;
    for (const ServerItem& server_item : *this) {
      if (server_item.id() == -1) {
        has_id = 0;
        break;
      }
    }

    if (has_id) {
      std::sort(begin(), end(),
                [](const ServerItem& left, const ServerItem& right) {
                  return left.id() < right.id();
                });
    } else {
      std::sort(begin(), end(),
                [](const ServerItem& left, const ServerItem& right) {
                  return left.addr() < right.addr();
                });
    }

    for (size_t i = 0; i < size(); ++i) {
      ServerItem& server_item = (*this)[i];
      server_item.set_id((int)i);
      DXINFO("server_addr=%s, server_id=%d", server_item.addr_str().c_str(),
             server_item.id());
    }
  }
};

/************************************************************************/
/* Scheduler */
/************************************************************************/
class Scheduler {
 private:
  TcpEndpoint scheduler_addr_;
  int server_num_ = 0;
  int worker_num_ = 0;
  ServerItems server_items_;
  enum STATE {
    STATE_NONE = 0,
    STATE_PHASE1 = 1,
    STATE_PHASE2 = 2,
  };
  int state_ = STATE_NONE;
  int phase1_finish_num_ = 0;
  std::unordered_set<std::string> phase2_finish_addrs_;

 private:
  void TerminateScheduler() const {
    IoContext io;
    TcpConnection conn(&io);
    DXCHECK(conn.ConnectRetry(scheduler_addr_) == 0);
    DXCHECK(WriteTerminationNotify(&conn) == 0);
  }

 public:
  std::vector<std::string> GetServerAddrs() const {
    return server_items_.GetAddrs();
  }

 public:
  void Init(const TcpEndpoint& scheduler_addr, int server_num, int worker_num) {
    scheduler_addr_ = scheduler_addr;
    server_num_ = server_num;
    worker_num_ = worker_num;
    server_items_.clear();
    state_ = STATE_PHASE1;
    phase1_finish_num_ = 0;
    phase2_finish_addrs_.clear();
    phase2_finish_addrs_.reserve(server_num_ + worker_num_);
  }

  int OnServerAddrRequest(RpcServer::conn_t conn,
                          const ServerAddrRequest& request,
                          ServerAddrResponse* response) {
    TcpEndpoint addr = conn->remote();
    if (request.is_server) {
      addr.port(request.server_port);
    }

    switch (state_) {
      case STATE_PHASE1:
        response->server_id = -1;
        response->server_addrs.clear();
        if (request.is_server) {
          if (server_items_.find(addr) == server_items_.end()) {
            ServerItem server_item(request.server_id, addr);
            server_items_.emplace_back(std::move(server_item));
            if (++phase1_finish_num_ == server_num_) {
              server_items_.AllocId();
              state_ = STATE_PHASE2;
            }
          }
        }
        break;
      case STATE_PHASE2:
        if (request.is_server) {
          response->server_id = server_items_.GetId(addr);
        } else {
          response->server_id = -1;
        }
        response->server_addrs = server_items_.GetAddrs();
        phase2_finish_addrs_.emplace(to_string(addr));
        if ((int)phase2_finish_addrs_.size() == server_num_ + worker_num_) {
          TerminateScheduler();
        }
        break;
    }
    return 0;
  }
};

void RunScheduler(const TcpEndpoint& scheduler_addr, int server_num,
                  int worker_num) {
  Scheduler scheduler;
  scheduler.Init(scheduler_addr, server_num, worker_num);

  TcpServerConfig config;
  config.listen_endpoint = scheduler_addr;
  config.thread = 1;

  RpcServer rpc_server;
  rpc_server.set_config(config);
  rpc_server.RegisterRequestHandler<ServerAddrRequest, ServerAddrResponse>(
      RPC_TYPE_SERVER_ADDR_REQUEST,
      std::bind(&Scheduler::OnServerAddrRequest, &scheduler,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));
  rpc_server.Run();

  std::cout << "scheduler_addr=" << scheduler_addr << std::endl;
  std::cout << "server_addrs=" << Join(scheduler.GetServerAddrs(), ";")
            << std::endl;
}

/************************************************************************/
/* Server */
/************************************************************************/
uint16_t GetFreeTcpPort(int is_v4) {
  IoContext io;
  TcpAcceptor acceptor(io);
  TcpEndpoint addr;
  if (is_v4) {
    addr = TcpEndpoint(asio::ip::tcp::v4(), 0);
  } else {
    addr = TcpEndpoint(asio::ip::tcp::v6(), 0);
  }
  acceptor.open(addr.protocol());
  acceptor.set_option(TcpAcceptor::reuse_address(true));
  acceptor.bind(addr);
  acceptor.listen();
  return acceptor.local_endpoint().port();
}

void RunServer(const TcpEndpoint& scheduler_addr, int server_id) {
  IoContext io;
  TcpConnection conn(&io);
  DXCHECK(conn.ConnectRetry(scheduler_addr) == 0);

  ServerAddrRequest request;
  request.is_server = 1;
  request.server_id = server_id;
  request.server_port =
      GetFreeTcpPort(scheduler_addr.address().is_v4() ? 1 : 0);

  ServerAddrResponse response;
  for (;;) {
    if (WriteRequestReadResponse(&conn, RPC_TYPE_SERVER_ADDR_REQUEST, request,
                                 &response) == 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100));  // magic number
    } else {
      DXINFO("Failed to WriteRequestReadResponse.");
      break;
    }
  }
  DXCHECK(response.server_id != -1);
  DXCHECK(!response.server_addrs.empty());
  conn.Close();

  std::cout << "scheduler_addr=" << scheduler_addr << std::endl;
  std::cout << "server_addrs=" << Join(response.server_addrs, ";") << std::endl;
  std::cout << "server_id=" << response.server_id << std::endl;
}

/************************************************************************/
/* Worker */
/************************************************************************/
void RunWorker(const TcpEndpoint& scheduler_addr) {
  IoContext io;
  TcpConnection conn(&io);
  DXCHECK(conn.ConnectRetry(scheduler_addr) == 0);

  ServerAddrRequest request;
  request.is_server = 0;
  request.server_id = -1;
  request.server_port = 0;

  ServerAddrResponse response;
  for (;;) {
    if (WriteRequestReadResponse(&conn, RPC_TYPE_SERVER_ADDR_REQUEST, request,
                                 &response) == 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100));  // magic number
    } else {
      DXINFO("Failed to WriteRequestReadResponse.");
      break;
    }
  }
  DXCHECK(!response.server_addrs.empty());
  conn.Close();

  std::cout << "scheduler_addr=" << scheduler_addr << std::endl;
  std::cout << "server_addrs=" << Join(response.server_addrs, ";") << std::endl;
}

/************************************************************************/
/* main */
/************************************************************************/
int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string role;
  DXCHECK(_getenv("ROLE", &role));
  DXCHECK(role == "scheduler" || role == "server" || role == "worker");

  std::string scheduler_ip;
  int scheduler_port;
  TcpEndpoint scheduler_addr;
  DXCHECK(_getenv("SCHEDULER_IP", &scheduler_ip));
  DXCHECK(_getenv("SCHEDULER_PORT", &scheduler_port));
  scheduler_addr = MakeTcpEndpoint(scheduler_ip, scheduler_port);

  if (role == "scheduler") {
    int server_num = 0;
    int worker_num = 0;
    DXCHECK(_getenv("SERVER_NUM", &server_num));
    DXCHECK(_getenv("WORKER_NUM", &worker_num));
    RunScheduler(scheduler_addr, server_num, worker_num);
  } else if (role == "server") {
    int server_id;
    if (!_getenv("SERVER_ID", &server_id)) {
      server_id = -1;
    }
    RunServer(scheduler_addr, server_id);
  } else {
    RunWorker(scheduler_addr);
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
