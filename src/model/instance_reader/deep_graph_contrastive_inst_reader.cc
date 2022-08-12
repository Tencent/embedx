// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhitao Wang (wztzenk@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>

#include <vector>

#include "src/io/indexing_wrapper.h"
#include "src/io/value.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {

class DeepGraphContrastiveInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  bool use_neigh_feat_ = false;
  std::vector<int> num_neighbors_;
  double left_edge_drop_prob_ = 0.2;
  double right_edge_drop_prob_ = 0.4;
  double left_feat_mask_prob_ = 0.3;
  double right_feat_mask_prob_ = 0.4;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;
  vec_int_t src_nodes_;

  uint16_t ns_id_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;

 public:
  DEFINE_INSTANCE_READER_LIKE(DeepGraphContrastiveInstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    flow_ = NewNeighborAggregationFlow(graph_client);
    return true;
  }

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      DXCHECK(val == 0 || val == 1);
      is_train_ = val;
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 0 || val == 1);
      use_neigh_feat_ = val;
    } else if (k == "left_edge_drop_prob") {
      left_edge_drop_prob_ = std::stod(v);
      DXCHECK(left_edge_drop_prob_ >= 0);
    } else if (k == "right_edge_drop_prob") {
      right_edge_drop_prob_ = std::stod(v);
      DXCHECK(right_edge_drop_prob_ >= 0);
    } else if (k == "left_feat_mask_prob") {
      left_feat_mask_prob_ = std::stod(v);
      DXCHECK(left_feat_mask_prob_ >= 0);
    } else if (k == "right_feat_mask_prob") {
      right_feat_mask_prob_ = std::stod(v);
      DXCHECK(right_feat_mask_prob_ >= 0);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

  void PostInit(const std::string& /*node_config*/) override {
    ns_id_ = 0;
    indexing_wrapper_ = IndexingWrapper::Create("");
  }

  bool GetBatch(Instance* inst) override {
    return is_train_ ? GetTrainBatch(inst) : GetPredictBatch(inst);
  }

  /************************************************************************/
  /* Read batch data from file for training */
  /************************************************************************/
  bool GetTrainBatch(Instance* inst) {
    std::vector<NodeValue> values;
    if (!NextInstanceBatch<NodeValue>(inst, batch_, &values)) {
      return false;
    }

    src_nodes_ = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // build two augment graphs, which we name as left graph and right graph
    // X_NODE_LEFT_MASKED_FEATURE_NAME: masked node features in left graph
    // X_NODE_RIGHT_MASKED_FEATURE_NAME: masked node features in right
    // X_NEIGH_LEFT_MASKED_FEATURE_NAME: masked neigh features in left graph
    // X_NEIGH_RIGHT_MASKED_FEATURE_NAME: masked neigh features in right
    // X_SELF_LEFT_DROPPED_BLOCK_NAME: blocks of left graph nodes
    // X_SELF_RIGHT_DROPPED_BLOCK_NAME: blocks of right graph nodes
    // X_NEIGH_LEFT_DROPPED_BLOCK_NAME: edge dropped blocks of left neighs
    // X_NEIGH_RIGHT_DROPPED_BLOCK_NAME: edge dropped blocks of right neighbors

    FillTrainGraphNodeFeatureAndBlocks(
        inst, src_nodes_, num_neighbors_,
        instance_name::X_NODE_LEFT_MASKED_FEATURE_NAME,
        instance_name::X_NODE_RIGHT_MASKED_FEATURE_NAME,
        instance_name::X_NEIGH_LEFT_MASKED_FEATURE_NAME,
        instance_name::X_NEIGH_RIGHT_MASKED_FEATURE_NAME,
        instance_name::X_SELF_LEFT_DROPPED_BLOCK_NAME,
        instance_name::X_SELF_RIGHT_DROPPED_BLOCK_NAME,
        instance_name::X_NEIGH_LEFT_DROPPED_BLOCK_NAME,
        instance_name::X_NEIGH_RIGHT_DROPPED_BLOCK_NAME);

    inst->set_batch(src_nodes_.size());
    return true;
  }

  /************************************************************************/
  /* Read batch data from file for prediction */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    std::vector<NodeValue> values;
    if (!NextInstanceBatch<NodeValue>(inst, batch_, &values)) {
      return false;
    }
    src_nodes_ = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // Only use original graph for inference with trained gnn encoder
    FillPredictGraphNodeFeatureAndBlocks(
        inst, src_nodes_, num_neighbors_, instance_name::X_NODE_FEATURE_NAME,
        instance_name::X_NEIGH_FEATURE_NAME, instance_name::X_SELF_BLOCK_NAME,
        instance_name::X_NEIGH_BLOCK_NAME);

    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = src_nodes_;
    inst->set_batch((int)src_nodes_.size());
    return true;
  }

  // FillTrainGraphNodeFeatureAndBlock
  void FillTrainGraphNodeFeatureAndBlocks(
      Instance* inst, const vec_int_t& nodes,
      const std::vector<int>& num_neighbors,
      const std::string& node_left_masked_feature_name,
      const std::string& node_right_masked_feature_name,
      const std::string& neigh_left_masked_feature_name,
      const std::string& neigh_right_masked_feature_name,
      const std::string& node_left_dropped_block_name,
      const std::string& node_right_dropped_block_name,
      const std::string& neigh_left_dropped_block_name,
      const std::string& neigh_right_dropped_block_name) const {
    vec_set_t level_nodes;
    vec_map_neigh_t level_neighs;

    // sample subgraph
    flow_->SampleSubGraph(nodes, num_neighbors, &level_nodes, &level_neighs);

    // Fill node and neighbor masked feature for left graph
    flow_->set_feature_mask_prob(left_feat_mask_prob_);
    flow_->FillLevelNodeFeature(inst, node_left_masked_feature_name,
                                level_nodes);
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(inst, neigh_left_masked_feature_name,
                                   level_nodes);
    }

    // Fill node and neighbor masked feature for right graph
    flow_->set_feature_mask_prob(right_feat_mask_prob_);
    flow_->FillLevelNodeFeature(inst, node_right_masked_feature_name,
                                level_nodes);
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(inst, neigh_right_masked_feature_name,
                                   level_nodes);
    }

    indexing_wrapper_->Clear();
    indexing_wrapper_->BuildFrom(level_nodes);
    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);

    // Fill self and neighbor block for left graph with random edge drop
    flow_->set_edge_drop_prob(left_edge_drop_prob_);
    flow_->FillSelfAndNeighGraphBlock(
        inst, node_left_dropped_block_name, neigh_left_dropped_block_name,
        level_nodes, level_neighs, indexings, false);

    // Fill self and neighbor block for right graph with random edge drop
    flow_->set_edge_drop_prob(right_edge_drop_prob_);
    flow_->FillSelfAndNeighGraphBlock(
        inst, node_right_dropped_block_name, neigh_right_dropped_block_name,
        level_nodes, level_neighs, indexings, false);

    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, nodes,
                           &indexings[0]);
  }

  // FillPredictGraphNodeFeatureAndBlock
  void FillPredictGraphNodeFeatureAndBlocks(
      Instance* inst, const vec_int_t& nodes,
      const std::vector<int>& num_neighbors,
      const std::string& node_feature_name,
      const std::string& neigh_feature_name, const std::string& node_block_name,
      const std::string& neigh_block_name) const {
    vec_set_t level_nodes;
    vec_map_neigh_t level_neighs;

    // sample subgraph
    flow_->SampleSubGraph(nodes, num_neighbors, &level_nodes, &level_neighs);

    // Fill node and neighbor feature
    flow_->set_feature_mask_prob(0);
    flow_->FillLevelNodeFeature(inst, node_feature_name, level_nodes);
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(inst, neigh_feature_name, level_nodes);
    }

    // Fill self and neighbor block
    indexing_wrapper_->Clear();
    indexing_wrapper_->BuildFrom(level_nodes);
    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->set_edge_drop_prob(0);
    flow_->FillSelfAndNeighGraphBlock(inst, node_block_name, neigh_block_name,
                                      level_nodes, level_neighs, indexings,
                                      false);

    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, nodes,
                           &indexings[0]);
  }
};

INSTANCE_READER_REGISTER(DeepGraphContrastiveInstReader,
                         "DeepGraphContrastiveInstReader");
INSTANCE_READER_REGISTER(DeepGraphContrastiveInstReader,
                         "deep_graph_contrastive_inst_reader");
}  // namespace embedx
