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

#include "src/graph/server/dist_graph_server.h"

#include <deepx_core/common/str_util.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>

#include <string>

#include "src/graph/data_op/cache_node_lookuper_op/cache_node_lookuper.h"
#include "src/graph/data_op/context_lookuper_op/context_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/neighbor_feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/node_feature_lookuper.h"
#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_factory.h"
#include "src/graph/data_op/meta_lookuper_op/meta_lookuper.h"
#include "src/graph/data_op/negative_sampler_op/indep_negative_sampler.h"
#include "src/graph/data_op/negative_sampler_op/shared_negative_sampler.h"
#include "src/graph/data_op/neighbor_sampler_op/random_neighbor_sampler.h"
#include "src/graph/data_op/random_walker_op/static_random_walker.h"
#include "src/graph/graph_config.h"

namespace embedx {
namespace {

void TouchSuccessFile(const GraphConfig& config) {
  if (config.success_out().empty()) {
    DXINFO("'success_out' directory is empty.");
    return;
  }

  std::string out_file =
      config.success_out() + "/_SUCCESS" + std::to_string(config.shard_id());
  DXINFO("Touch 'success_out' file: %s.", out_file.c_str());
  deepx_core::AutoOutputFileStream os;
  DXCHECK_THROW(os.Open(out_file));
  os.Close();
}

}  // namespace

using ::embedx::graph_op::LocalGSOp;
using ::embedx::graph_op::LocalGSOpFactory;
using ::embedx::graph_op::LocalGSOpResource;

bool DistGraphServer::InitGraphServer(const GraphConfig& config) {
  resource_.reset(new graph_op::LocalGSOpResource);

  resource_->set_graph_config(config);

  // data
  auto graph = InMemoryGraph::Create(config);
  if (!graph) {
    return false;
  }
  resource_->set_graph(std::move(graph));

  auto sampler_source = NewGraphSamplerSource(resource_->graph());
  if (!sampler_source) {
    return false;
  }
  resource_->set_sampler_source(std::move(sampler_source));

  auto negative_sampler_builder = NewSamplerBuilder(
      resource_->sampler_source(), SamplerBuilderEnum::NEGATIVE_SAMPLER,
      config.negative_sampler_type(), config.thread_num());
  if (!negative_sampler_builder) {
    return false;
  }
  resource_->set_negative_sampler_builder(std::move(negative_sampler_builder));

  auto neighbor_sampler_builder = NewSamplerBuilder(
      resource_->sampler_source(), SamplerBuilderEnum::NEIGHBOR_SAMPLER,
      config.neighbor_sampler_type(), config.thread_num());
  if (!neighbor_sampler_builder) {
    return false;
  }
  resource_->set_neighbor_sampler_builder(std::move(neighbor_sampler_builder));

  return LocalGSOpFactory::GetInstance()->Init(resource_.get());
}

bool DistGraphServer::InitRpcServer(const GraphConfig& config) {
  vec_str_t ip_ports;
  deepx_core::Split(config.ip_ports(), ";", &ip_ports);
  if (ip_ports.empty()) {
    DXERROR("Invalid graph server address: %s.", config.ip_ports().c_str());
    return false;
  }

  deepx_core::TcpServerConfig tcp_config;
  tcp_config.listen_endpoint =
      deepx_core::MakeTcpEndpoint(ip_ports[config.shard_id()]);
  tcp_config.thread = config.thread_num();
  rpc_server_.set_config(tcp_config);
  return true;
}

// The last declaration of DistGraphServer::Name() function is to avoid compile
// warning of extra ';'
#define DEFINE_REQUEST_HANDLER(Name)                                           \
  void DistGraphServer::Name() {                                               \
    LocalGSOp* gs_op = LocalGSOpFactory::GetInstance()->LookupOrCreate(#Name); \
    DXCHECK(gs_op != nullptr);                                                 \
    auto* op = dynamic_cast<class ::embedx::graph_op::Name*>(gs_op);           \
    auto rpc_type = Name##Request::rpc_type();                                 \
    rpc_server_.RegisterRequestHandler<Name##Request, Name##Response>(         \
        rpc_type, [op](const Name##Request& req, Name##Response* resp) {       \
          return op->HandleRpc(req, resp);                                     \
        });                                                                    \
  }                                                                            \
  void DistGraphServer::Name()

DEFINE_REQUEST_HANDLER(MetaLookuper);
DEFINE_REQUEST_HANDLER(FeatureLookuper);
DEFINE_REQUEST_HANDLER(NodeFeatureLookuper);
DEFINE_REQUEST_HANDLER(NeighborFeatureLookuper);
DEFINE_REQUEST_HANDLER(ContextLookuper);
DEFINE_REQUEST_HANDLER(RandomNeighborSampler);
DEFINE_REQUEST_HANDLER(SharedNegativeSampler);
DEFINE_REQUEST_HANDLER(IndepNegativeSampler);
DEFINE_REQUEST_HANDLER(StaticRandomWalker);
DEFINE_REQUEST_HANDLER(CacheNodeLookuper);

#undef DEFINE_REQUEST_HANDLER

void DistGraphServer::RegisterRequestHandler() {
  MetaLookuper();
  FeatureLookuper();
  NodeFeatureLookuper();
  NeighborFeatureLookuper();
  ContextLookuper();
  RandomNeighborSampler();
  SharedNegativeSampler();
  IndepNegativeSampler();
  StaticRandomWalker();
  CacheNodeLookuper();
}

bool DistGraphServer::Start(const GraphConfig& config) {
  if (!InitGraphServer(config)) {
    DXERROR("Failed to init graph server.");
    return false;
  }

  if (!InitRpcServer(config)) {
    DXERROR("Failed to init rpc server.");
    return false;
  }

  RegisterRequestHandler();
  TouchSuccessFile(config);
  rpc_server_.Run();
  return true;
}

}  // namespace embedx
