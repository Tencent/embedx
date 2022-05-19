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

#include "src/model/data_flow//random_walk_flow.h"

#include <deepx_core/dx_log.h>

namespace embedx {

void RandomWalkFlow::FillNodeOrIndex(Instance* inst, const std::string& name,
                                     const csr_t& nodes,
                                     const Indexing* indexing) const {
  auto* ptr = &inst->get_or_insert<csr_t>(name);
  ptr->clear();

  CSR_FOR_EACH_ROW(nodes, i) {
    CSR_FOR_EACH_COL(nodes, i) {
      auto node = CSR_COL(nodes);
      if (indexing != nullptr) {
        int index = indexing->Get(node);
        DXCHECK(index >= 0);
        ptr->emplace(index, 1);
        ptr->add_row();
      } else {
        ptr->emplace(node, 1);
        ptr->add_row();
      }
    }
  }
}

void RandomWalkFlow::FillNodeOrIndex(Instance* inst, const std::string& name,
                                     const vec_int_t& nodes,
                                     const Indexing* indexing) const {
  auto* ptr = &inst->get_or_insert<csr_t>(name);
  ptr->clear();

  for (auto node : nodes) {
    if (indexing != nullptr) {
      int index = indexing->Get(node);
      DXCHECK(index >= 0);
      ptr->emplace(index, 1);
      ptr->add_row();
    } else {
      ptr->emplace(node, 1);
      ptr->add_row();
    }
  }
}

void RandomWalkFlow::FillNodeFeature(Instance* inst, const std::string& name,
                                     const vec_int_t& nodes,
                                     bool add_self) const {
  auto* node_feat_ptr = &inst->get_or_insert<csr_t>(name);
  node_feat_ptr->clear();

  std::vector<vec_pair_t> tmp_feats_list;
  graph_client_.LookupNodeFeature(nodes, &tmp_feats_list);

  for (size_t i = 0; i < tmp_feats_list.size(); ++i) {
    for (const auto& entry : tmp_feats_list[i]) {
      node_feat_ptr->emplace(entry.first, entry.second);
    }

    if (add_self) {
      node_feat_ptr->emplace(nodes[i], 1.0);
    }

    node_feat_ptr->add_row();
  }
}

void RandomWalkFlow::FillEdgeAndLabel(
    Instance* inst, const std::string& src_name, const std::string& dst_name,
    const std::string& y_name, const vec_int_t& src_nodes,
    const std::vector<vec_int_t>& dst_nodes_list,
    const std::vector<vec_int_t>& neg_nodes_list) const {
  auto* src_node_ptr = &inst->get_or_insert<csr_t>(src_name);
  auto* dst_node_ptr = &inst->get_or_insert<csr_t>(dst_name);
  src_node_ptr->clear();
  dst_node_ptr->clear();

  vec_float_t y_bufs;
  for (size_t i = 0; i < src_nodes.size(); ++i) {
    // (src_node, dst_node, 1.0)
    for (size_t j = 0; j < dst_nodes_list[i].size(); ++j) {
      src_node_ptr->emplace(src_nodes[i], 1);
      src_node_ptr->add_row();
      dst_node_ptr->emplace(dst_nodes_list[i][j], 1);
      dst_node_ptr->add_row();
      y_bufs.emplace_back(1.0);
      // (src_node, neg_node, 0.0)
      for (auto& neg_node : neg_nodes_list[i]) {
        src_node_ptr->emplace(src_nodes[i], 1);
        src_node_ptr->add_row();
        dst_node_ptr->emplace(neg_node, 1);
        dst_node_ptr->add_row();
        y_bufs.emplace_back(0.0);
      }
    }
  }

  auto* y_ptr = &inst->get_or_insert<tsr_t>(y_name);
  y_ptr->resize(y_bufs.size(), 1);
  for (size_t i = 0; i < y_bufs.size(); ++i) {
    y_ptr->data(i) = y_bufs[i];
  }
}

std::unique_ptr<RandomWalkFlow> NewRandomWalkFlow(
    const GraphClient* graph_client) {
  std::unique_ptr<RandomWalkFlow> flow;
  flow.reset(new RandomWalkFlow(graph_client));
  return flow;
}

}  // namespace embedx
