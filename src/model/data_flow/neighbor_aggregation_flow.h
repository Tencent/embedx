// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#pragma once
#include <deepx_core/graph/tensor_map.h>  // Instance

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"
#include "src/io/indexing.h"
#include "src/io/io_util.h"

namespace embedx {

using ::deepx_core::Instance;

class NeighborAggregationFlow : public deepx_core::DataType {
 private:
  const GraphClient& graph_client_;
  float_t edge_drop_prob_ = 0;
  float_t feat_mask_prob_ = 0;

 public:
  explicit NeighborAggregationFlow(const GraphClient* graph_client)
      : graph_client_(*graph_client) {}
  virtual ~NeighborAggregationFlow() = default;

 public:
  void set_edge_drop_prob(float_t edge_drop_prob) noexcept {
    edge_drop_prob_ = edge_drop_prob;
  }

  void set_feature_mask_prob(float_t feat_mask_prob) noexcept {
    feat_mask_prob_ = feat_mask_prob;
  }

  void SampleSubGraph(const vec_int_t& nodes,
                      const std::vector<int>& num_neighbors,
                      vec_set_t* level_nodes,
                      vec_map_neigh_t* level_neighs) const;
  void MergeTo(const vec_int_t& src_nodes, vec_int_t* dst_nodes) const;
  void MergeTo(const std::vector<vec_int_t>& src_nodes_list,
               vec_int_t* dst_nodes) const;
  void ShuffleNodesInBatch(const vec_set_t& level_nodes,
                           vec_int_t* shuffled_nodes) const;
  void ShuffleNodesInGlobal(const vec_set_t& level_nodes,
                            vec_int_t* shuffled_nodes) const;

 public:
  void FillNodeOrIndex(Instance* inst, const std::string& name,
                       const vec_int_t& nodes, const Indexing* indexing) const;
  void FillNodeFeature(Instance* inst, const std::string& name,
                       const vec_int_t& nodes, bool add_self) const;
  void FillLevelNodeFeature(Instance* inst, const std::string& name,
                            const vec_set_t& level_nodes) const;
  void FillLevelNeighFeature(Instance* inst, const std::string& name,
                             const vec_set_t& level_nodes) const;
  void FillSelfAndNeighGraphBlock(Instance* inst, const std::string& self_name,
                                  const std::string& neigh_name,
                                  const vec_set_t& level_nodes,
                                  const vec_map_neigh_t& level_neighs,
                                  const std::vector<Indexing>& indexings,
                                  bool add_self) const;
  void FillLabelAndCheck(Instance* inst, const std::string& y_name,
                         const std::vector<vecl_t>& labels_list, int label_num,
                         int max_label) const;

 public:
  template <class SrcIndexingFunc, class DstIndexingFunc>
  void FillEdgeAndLabel(Instance* inst, const std::string& src_name,
                        const std::string& dst_name, const std::string& y_name,
                        const vec_int_t& src_nodes, const vec_int_t& dst_nodes,
                        const std::vector<vec_int_t>& neg_nodes_list,
                        SrcIndexingFunc&& src_f,
                        DstIndexingFunc&& dst_f) const {
    auto* src_ptr = &inst->get_or_insert<csr_t>(src_name);
    auto* dst_ptr = &inst->get_or_insert<csr_t>(dst_name);
    src_ptr->clear();
    dst_ptr->clear();

    vec_float_t y_bufs;
    for (size_t i = 0; i < src_nodes.size(); ++i) {
      // (src, dst, 1)
      auto src = src_f(src_nodes[i]);
      auto dst = dst_f(dst_nodes[i]);
      src_ptr->emplace(src, 1);
      src_ptr->add_row();
      dst_ptr->emplace(dst, 1);
      dst_ptr->add_row();
      y_bufs.emplace_back(1);
      auto ns = io_util::GetNodeType(dst_nodes[i]);
      for (auto neg_node : neg_nodes_list[ns]) {
        // (src, neg, 0)
        auto neg = dst_f(neg_node);
        src_ptr->emplace(src, 1);
        src_ptr->add_row();
        dst_ptr->emplace(neg, 1);
        dst_ptr->add_row();
        y_bufs.emplace_back(0);
      }
    }
    auto* y_ptr = &inst->get_or_insert<tsr_t>(y_name);
    y_ptr->resize(y_bufs.size(), 1);
    for (size_t i = 0; i < y_bufs.size(); ++i) {
      y_ptr->data(i) = y_bufs[i];
    }
  }
};

std::unique_ptr<NeighborAggregationFlow> NewNeighborAggregationFlow(
    const GraphClient* graph_client);

}  // namespace embedx
