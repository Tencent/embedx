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

class GraphClientImpl;

class GraphClient {
 private:
  std::unique_ptr<GraphClientImpl> impl_;

 public:
  explicit GraphClient(std::unique_ptr<GraphClientImpl>&& impl);
  ~GraphClient();

 public:
  // negative sampler
  bool SharedSampleNegative(int count, const vec_int_t& nodes,
                            const vec_int_t& excluded_nodes,
                            std::vector<vec_int_t>* sampled_nodes_list) const;
  bool IndepSampleNegative(int count, const vec_int_t& nodes,
                           const vec_int_t& excluded_nodes,
                           std::vector<vec_int_t>* sampled_nodes_list) const;
  // neighbor sampler
  bool RandomSampleNeighbor(int count, const vec_int_t& nodes,
                            std::vector<vec_int_t>* neighbor_nodes_list) const;

  // random walker
  bool StaticTraverse(const vec_int_t& cur_nodes,
                      const std::vector<int>& walk_lens,
                      const WalkerInfo& walker_info,
                      std::vector<vec_int_t>* seqs) const;

  // feature
  bool LookupFeature(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* node_feats,
                     std::vector<vec_pair_t>* neigh_feats) const;

  bool LookupNodeFeature(const vec_int_t& nodes,
                         std::vector<vec_pair_t>* node_feats) const;

  bool LookupNeighborFeature(const vec_int_t& nodes,
                             std::vector<vec_pair_t>* neigh_feats) const;

  // context
  bool LookupContext(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* contexts) const;
};

enum class GraphClientEnum : int { LOCAL = 0, DIST = 1 };

std::unique_ptr<GraphClient> NewGraphClient(const GraphConfig& config,
                                            GraphClientEnum type);

}  // namespace embedx
