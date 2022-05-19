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

#include "src/model/data_flow/neighbor_aggregation_flow.h"

#include <deepx_core/dx_log.h>

#include <random>  // std::random_device, std::default_random_engine

#include "src/common/random.h"

namespace embedx {
namespace {

template <class Func>
void FillLevelFeature(Func&& LookupFunc, const vec_set_t& level_nodes,
                      float_t feat_mask_prob, csr_t* csr_feats) {
  csr_feats->clear();

  vec_int_t tmp_nodes;
  std::vector<vec_pair_t> tmp_feats_list;
  for (const auto& level_node : level_nodes) {
    tmp_nodes.assign(level_node.begin(), level_node.end());
    LookupFunc(tmp_nodes, &tmp_feats_list);

    for (size_t j = 0; j < tmp_nodes.size(); ++j) {
      for (const auto& entry : tmp_feats_list[j]) {
        // Consistent with tf and pytorch mask operations
        if (ThreadLocalRandom() <= 1.0 - feat_mask_prob) {
          csr_feats->emplace(entry.first, entry.second);
        }
      }
      csr_feats->add_row();
    }
  }
}

}  // namespace

void NeighborAggregationFlow::SampleSubGraph(
    const vec_int_t& nodes, const std::vector<int>& num_neighbors,
    vec_set_t* level_nodes, vec_map_neigh_t* level_neighs) const {
  int graph_depth = num_neighbors.size();
  level_nodes->resize(graph_depth + 1);
  level_neighs->resize(graph_depth + 1);
  (*level_nodes)[0].clear();
  (*level_nodes)[0].insert(nodes.begin(), nodes.end());

  vec_int_t tmp_nodes;
  std::vector<vec_int_t> tmp_neighbors_list;

  for (size_t i = 0; i < num_neighbors.size(); ++i) {
    (*level_nodes)[i + 1].clear();
    (*level_neighs)[i].clear();

    tmp_nodes.assign((*level_nodes)[i].begin(), (*level_nodes)[i].end());
    graph_client_.RandomSampleNeighbor(num_neighbors[i], tmp_nodes,
                                       &tmp_neighbors_list);
    for (size_t j = 0; j < tmp_nodes.size(); ++j) {
      (*level_nodes)[i + 1].insert(tmp_neighbors_list[j].begin(),
                                   tmp_neighbors_list[j].end());
      (*level_neighs)[i].emplace(tmp_nodes[j], tmp_neighbors_list[j]);
    }
  }
}

void NeighborAggregationFlow::MergeTo(const vec_int_t& src_nodes,
                                      vec_int_t* dst_nodes) const {
  dst_nodes->insert(dst_nodes->begin(), src_nodes.begin(), src_nodes.end());
}

void NeighborAggregationFlow::MergeTo(
    const std::vector<vec_int_t>& src_nodes_list, vec_int_t* dst_nodes) const {
  for (auto& src_nodes : src_nodes_list) {
    MergeTo(src_nodes, dst_nodes);
  }
}

void NeighborAggregationFlow::ShuffleNodesInBatch(
    const vec_set_t& level_nodes, vec_int_t* shuffled_nodes) const {
  shuffled_nodes->clear();
  for (const auto& level_node : level_nodes) {
    shuffled_nodes->insert(shuffled_nodes->end(), level_node.begin(),
                           level_node.end());
  }

  static thread_local std::random_device rd;
  static thread_local std::default_random_engine rng(rd());
  std::shuffle(shuffled_nodes->begin(), shuffled_nodes->end(), rng);
}

void NeighborAggregationFlow::ShuffleNodesInGlobal(
    const vec_set_t& level_nodes, vec_int_t* shuffled_nodes) const {
  shuffled_nodes->clear();
  vec_int_t merged_nodes;
  for (const auto& nodes : level_nodes) {
    merged_nodes.insert(merged_nodes.end(), nodes.begin(), nodes.end());
  }

  // Use negative sampling to do global node sampling.
  // Need to use GetBatchNode function.
  std::vector<vec_int_t> neg_nodes_list;
  DXCHECK(
      graph_client_.IndepSampleNegative(1, merged_nodes, {}, &neg_nodes_list));

  for (const auto& nodes : neg_nodes_list) {
    shuffled_nodes->emplace_back(nodes[0]);
  }
}

void NeighborAggregationFlow::FillNodeOrIndex(Instance* inst,
                                              const std::string& name,
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

void NeighborAggregationFlow::FillNodeFeature(Instance* inst,
                                              const std::string& name,
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

void NeighborAggregationFlow::FillLevelNodeFeature(
    Instance* inst, const std::string& name,
    const vec_set_t& level_nodes) const {
  auto* feat_ptr = &inst->get_or_insert<csr_t>(name);
  auto f = [this](const vec_int_t& nodes,
                  std::vector<vec_pair_t>* tmp_feats_list) {
    graph_client_.LookupNodeFeature(nodes, tmp_feats_list);
  };
  FillLevelFeature(f, level_nodes, feat_mask_prob_, feat_ptr);
}

void NeighborAggregationFlow::FillLevelNeighFeature(
    Instance* inst, const std::string& name,
    const vec_set_t& level_nodes) const {
  auto* feat_ptr = &inst->get_or_insert<csr_t>(name);
  auto f = [this](const vec_int_t& nodes,
                  std::vector<vec_pair_t>* tmp_feats_list) {
    graph_client_.LookupNeighborFeature(nodes, tmp_feats_list);
  };
  FillLevelFeature(f, level_nodes, feat_mask_prob_, feat_ptr);
}

void NeighborAggregationFlow::FillSelfAndNeighGraphBlock(
    Instance* inst, const std::string& self_name, const std::string& neigh_name,
    const vec_set_t& level_nodes, const vec_map_neigh_t& level_neighs,
    const std::vector<Indexing>& indexings, bool add_self) const {
  int graph_depth = level_neighs.size() - 1;
  for (int i = 0; i < graph_depth; ++i) {
    auto* self_block =
        &inst->get_or_insert<csr_t>(self_name + std::to_string(i));
    self_block->clear();
    auto* neigh_block =
        &inst->get_or_insert<csr_t>(neigh_name + std::to_string(i));
    neigh_block->clear();

    for (int j = 0; j < graph_depth - i; ++j) {
      for (auto node : level_nodes[j]) {
        // Fill self node block
        int self_id = indexings[j].Get(node);
        DXCHECK(self_id >= 0);
        self_block->emplace(self_id, 1);
        self_block->add_row();

        // Fill neighbor node block
        for (auto neigh_node : level_neighs[j].at(node)) {
          // Consistent with tf and pytorch drop operations
          if (ThreadLocalRandom() <= 1.0 - edge_drop_prob_) {
            auto neigh_id = indexings[j + 1].Get(neigh_node);
            DXCHECK(neigh_id >= 0);
            neigh_block->emplace(neigh_id, 1);
          }
        }

        // self connection, node -> node
        if (add_self) {
          neigh_block->emplace(self_id, 1);
        }

        neigh_block->add_row();
      }
    }
  }
}

void NeighborAggregationFlow::FillLabelAndCheck(
    Instance* inst, const std::string& y_name,
    const std::vector<vecl_t>& labels_list, int label_num,
    int max_label) const {
  auto* y_ptr = &inst->get_or_insert<tsr_t>(y_name);
  y_ptr->clear();
  y_ptr->resize(labels_list.size(), label_num);
  for (size_t i = 0; i < labels_list.size(); ++i) {
    auto& labels = labels_list[i];
    if (labels.size() != (size_t)label_num) {
      DXTHROW_INVALID_ARGUMENT(
          "Invalid labels, the size of labels: %zu must be %d.", labels.size(),
          label_num);
    }
    for (size_t j = 0; j < labels.size(); ++j) {
      if (labels[j] > max_label) {
        DXTHROW_INVALID_ARGUMENT(
            "Invalid labels, labels[%zu]: %d must be less than or equal to %d.",
            j, labels[j], max_label);
      }
      y_ptr->data(i * label_num + j) = labels[j];
    }
  }
}

std::unique_ptr<NeighborAggregationFlow> NewNeighborAggregationFlow(
    const GraphClient* graph_client) {
  std::unique_ptr<NeighborAggregationFlow> flow;
  flow.reset(new NeighborAggregationFlow(graph_client));
  return flow;
}

}  // namespace embedx
