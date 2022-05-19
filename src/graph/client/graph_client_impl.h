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
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {

class GraphClientImpl {
 public:
  virtual ~GraphClientImpl() = default;

 public:
  // negative sampler
  virtual bool SharedSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const = 0;
  virtual bool IndepSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const = 0;

  // neighbor sampler
  virtual bool RandomSampleNeighbor(
      int count, const vec_int_t& nodes,
      std::vector<vec_int_t>* neighbor_nodes_list) const = 0;

  // random walker
  virtual bool StaticTraverse(const vec_int_t& cur_nodes,
                              const std::vector<int>& walk_lens,
                              const WalkerInfo& walker_info,
                              std::vector<vec_int_t>* seqs) const = 0;

  // feature
  virtual bool LookupFeature(const vec_int_t& nodes,
                             std::vector<vec_pair_t>* node_feats,
                             std::vector<vec_pair_t>* neigh_feats) const = 0;

  virtual bool LookupNodeFeature(const vec_int_t& nodes,
                                 std::vector<vec_pair_t>* node_feats) const = 0;

  virtual bool LookupNeighborFeature(
      const vec_int_t& nodes, std::vector<vec_pair_t>* neigh_feats) const = 0;

  // context
  virtual bool LookupContext(const vec_int_t& nodes,
                             std::vector<vec_pair_t>* contexts) const = 0;
};

template <typename GraphClientTypes>
class GraphClientImplBase : public GraphClientImpl {
 protected:
  std::unique_ptr<typename GraphClientTypes::Resource> resource_;
  typename GraphClientTypes::Factory* factory_;

 public:
  ~GraphClientImplBase() override = default;

 public:
  /************************************************************************/
  /* Negative sampler */
  /************************************************************************/
  bool SharedSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const override {
    auto* op = factory_->LookupOrCreate("SharedNegativeSampler");
    return dynamic_cast<typename GraphClientTypes::SharedNegativeSampler*>(op)
        ->Run(count, nodes, excluded_nodes, sampled_nodes_list);
  }

  bool IndepSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const override {
    auto* op = factory_->LookupOrCreate("IndepNegativeSampler");
    return dynamic_cast<typename GraphClientTypes::IndepNegativeSampler*>(op)
        ->Run(count, nodes, excluded_nodes, sampled_nodes_list);
  }

  /************************************************************************/
  /* Random neighbor sampler */
  /************************************************************************/
  bool RandomSampleNeighbor(
      int count, const vec_int_t& nodes,
      std::vector<vec_int_t>* neighbor_nodes_list) const override {
    auto* op = factory_->LookupOrCreate("RandomNeighborSampler");
    return dynamic_cast<typename GraphClientTypes::RandomNeighborSampler*>(op)
        ->Run(count, nodes, neighbor_nodes_list);
  }

  /************************************************************************/
  /* Random walker */
  /************************************************************************/
  bool StaticTraverse(const vec_int_t& cur_nodes,
                      const std::vector<int>& walk_lens,
                      const WalkerInfo& walker_info,
                      std::vector<vec_int_t>* seqs) const override {
    auto* op = factory_->LookupOrCreate("StaticRandomWalker");
    return dynamic_cast<typename GraphClientTypes::StaticRandomWalker*>(op)
        ->Run(cur_nodes, walk_lens, walker_info, seqs);
  }

  /************************************************************************/
  /* Feature Lookuper */
  /************************************************************************/
  bool LookupFeature(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* node_feats,
                     std::vector<vec_pair_t>* neigh_feats) const override {
    auto* op = factory_->LookupOrCreate("FeatureLookuper");
    return dynamic_cast<typename GraphClientTypes::FeatureLookuper*>(op)->Run(
        nodes, node_feats, neigh_feats);
  }

  bool LookupNodeFeature(const vec_int_t& nodes,
                         std::vector<vec_pair_t>* node_feats) const override {
    auto* op = factory_->LookupOrCreate("NodeFeatureLookuper");
    return dynamic_cast<typename GraphClientTypes::NodeFeatureLookuper*>(op)
        ->Run(nodes, node_feats);
  }

  bool LookupNeighborFeature(
      const vec_int_t& nodes,
      std::vector<vec_pair_t>* neigh_feats) const override {
    auto* op = factory_->LookupOrCreate("NeighborFeatureLookuper");
    return dynamic_cast<typename GraphClientTypes::NeighborFeatureLookuper*>(op)
        ->Run(nodes, neigh_feats);
  }

  /************************************************************************/
  /* Context Lookuper */
  /************************************************************************/
  bool LookupContext(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* contexts) const override {
    auto* op = factory_->LookupOrCreate("ContextLookuper");
    return dynamic_cast<typename GraphClientTypes::ContextLookuper*>(op)->Run(
        nodes, contexts);
  }
};

std::unique_ptr<GraphClientImpl> NewLocalGraphClientImpl(
    const GraphConfig& config);

std::unique_ptr<GraphClientImpl> NewDistGraphClientImpl(
    const GraphConfig& config);

}  // namespace embedx
