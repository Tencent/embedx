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

#include <deepx_core/dx_log.h>
#include <deepx_core/ps/rpc_client.h>

#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client_impl.h"
#include "src/graph/client/resource_post_initializer.h"
#include "src/graph/client/rpc_connector.h"
#include "src/graph/data_op/context_lookuper_op/dist_context_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/dist_feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/dist_neighbor_feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/dist_node_feature_lookuper.h"
#include "src/graph/data_op/gs_op_factory.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/data_op/negative_sampler_op/dist_indep_negative_sampler.h"
#include "src/graph/data_op/negative_sampler_op/dist_shared_negative_sampler.h"
#include "src/graph/data_op/neighbor_sampler_op/dist_random_neighbor_sampler.h"
#include "src/graph/data_op/random_walker_op/dist_static_random_walker.h"
#include "src/graph/graph_config.h"

namespace embedx {
namespace {

struct DistGraphClientTypes {
  using Resource = graph_op::DistGSOpResource;
  using Factory = graph_op::DistGSOpFactory;

  using SharedNegativeSampler = graph_op::DistSharedNegativeSampler;
  using IndepNegativeSampler = graph_op::DistIndepNegativeSampler;
  using RandomNeighborSampler = graph_op::DistRandomNeighborSampler;
  using StaticRandomWalker = graph_op::DistStaticRandomWalker;
  using FeatureLookuper = graph_op::DistFeatureLookuper;
  using NodeFeatureLookuper = graph_op::DistNodeFeatureLookuper;
  using NeighborFeatureLookuper = graph_op::DistNeighborFeatureLookuper;
  using ContextLookuper = graph_op::DistContextLookuper;
};

}  // namespace

class DistGraphClientImpl : public GraphClientImplBase<DistGraphClientTypes> {
 public:
  bool Init(const GraphConfig& config) {
    resource_.reset(new graph_op::DistGSOpResource);

    std::vector<deepx_core::TcpEndpoint> endpoints =
        deepx_core::MakeTcpEndpoints(config.ip_ports());
    auto rpc_connector = NewRpcConnector();
    if (!rpc_connector->Connect(endpoints)) {
      return false;
    }
    resource_->set_rpc_connector(std::move(rpc_connector));

    int shard_num = (int)endpoints.size();
    if (shard_num <= 0) {
      DXERROR("Number of shard: %d must be greater than 0.", shard_num);
      return false;
    }
    DXINFO("Number of shard is: %d.", shard_num);

    // op factory init
    factory_ = graph_op::DistGSOpFactory::GetInstance();
    if (!factory_->Init(resource_.get(), shard_num)) {
      return false;
    }

    return PostInitCacheStorage(resource_.get()) &&
           PostInitServerDistribution(shard_num, resource_.get());
  }
};

std::unique_ptr<GraphClientImpl> NewDistGraphClientImpl(
    const GraphConfig& config) {
  std::unique_ptr<GraphClientImpl> graph_client_impl;
  graph_client_impl.reset(new DistGraphClientImpl);
  if (!dynamic_cast<DistGraphClientImpl*>(graph_client_impl.get())
           ->Init(config)) {
    DXERROR("Failed to new dist graph client impl.");
    graph_client_impl.reset();
  }
  return graph_client_impl;
}

}  // namespace embedx
