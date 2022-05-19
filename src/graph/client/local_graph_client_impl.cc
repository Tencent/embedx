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

#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client_impl.h"
#include "src/graph/data_op/context_lookuper_op/context_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/neighbor_feature_lookuper.h"
#include "src/graph/data_op/feature_lookuper_op/node_feature_lookuper.h"
#include "src/graph/data_op/gs_op_factory.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/data_op/negative_sampler_op/indep_negative_sampler.h"
#include "src/graph/data_op/negative_sampler_op/shared_negative_sampler.h"
#include "src/graph/data_op/neighbor_sampler_op/random_neighbor_sampler.h"
#include "src/graph/data_op/random_walker_op/static_random_walker.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"
#include "src/sampler/sampler_source.h"

namespace embedx {
namespace {

struct LocalGraphClientTypes {
  using Resource = graph_op::LocalGSOpResource;
  using Factory = graph_op::LocalGSOpFactory;

  using SharedNegativeSampler = graph_op::SharedNegativeSampler;
  using IndepNegativeSampler = graph_op::IndepNegativeSampler;
  using RandomNeighborSampler = graph_op::RandomNeighborSampler;
  using StaticRandomWalker = graph_op::StaticRandomWalker;
  using FeatureLookuper = graph_op::FeatureLookuper;
  using NodeFeatureLookuper = graph_op::NodeFeatureLookuper;
  using NeighborFeatureLookuper = graph_op::NeighborFeatureLookuper;
  using ContextLookuper = graph_op::ContextLookuper;
};

}  // namespace

class LocalGraphClientImpl : public GraphClientImplBase<LocalGraphClientTypes> {
 public:
  bool Init(const GraphConfig& config) {
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
    resource_->set_negative_sampler_builder(
        std::move(negative_sampler_builder));

    auto neighbor_sampler_builder = NewSamplerBuilder(
        resource_->sampler_source(), SamplerBuilderEnum::NEIGHBOR_SAMPLER,
        config.neighbor_sampler_type(), config.thread_num());
    if (!neighbor_sampler_builder) {
      return false;
    }
    resource_->set_neighbor_sampler_builder(
        std::move(neighbor_sampler_builder));

    // op factory init
    factory_ = graph_op::LocalGSOpFactory::GetInstance();
    return factory_->Init(resource_.get());
  }
};

std::unique_ptr<GraphClientImpl> NewLocalGraphClientImpl(
    const GraphConfig& config) {
  std::unique_ptr<GraphClientImpl> graph_client_impl;
  graph_client_impl.reset(new LocalGraphClientImpl());
  if (!dynamic_cast<LocalGraphClientImpl*>(graph_client_impl.get())
           ->Init(config)) {
    DXERROR("Failed to new local graph client impl.");
    graph_client_impl.reset();
  }
  return graph_client_impl;
}

}  // namespace embedx
