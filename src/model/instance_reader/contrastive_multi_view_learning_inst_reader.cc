// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Litao Hong (Lthong.brian@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

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

class ContrastiveMultiViewInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;
  bool use_neigh_feat_ = false;
  std::vector<int> num_neighbors_;
  std::vector<int> enhance_num_neighbors_;
  int shuffle_type_ = 0;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;
  vec_int_t src_nodes_;

  uint16_t ns_id_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;
  std::unique_ptr<IndexingWrapper> enhance_indexing_wrapper_;

 public:
  DEFINE_INSTANCE_READER_LIKE(ContrastiveMultiViewInstReader);

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
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      DXCHECK(num_neg_ > 0);
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
    } else if (k == "enhance_num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &enhance_num_neighbors_));
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 0 || val == 1);
      use_neigh_feat_ = val;
    } else if (k == "shuffle_type") {
      auto val = std::stoi(v);
      DXCHECK(val == 0 || val == 1 || val == 2);
      shuffle_type_ = val;
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
    enhance_indexing_wrapper_ = IndexingWrapper::Create("");
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

    // build src graph
    BuildGraphNodeFeatureAndBlocks(
        inst, src_nodes_, num_neighbors_, indexing_wrapper_.get(),
        instance_name::X_NODE_FEATURE_NAME,
        instance_name::X_NODE_SHUFFLED_FEATURE_NAME,
        instance_name::X_SELF_BLOCK_NAME, instance_name::X_NEIGH_BLOCK_NAME);

    // build contrastive graph
    BuildGraphNodeFeatureAndBlocks(
        inst, src_nodes_, enhance_num_neighbors_,
        enhance_indexing_wrapper_.get(),
        instance_name::X_ENHANCE_NODE_FEATURE_NAME,
        instance_name::X_ENHANCE_NODE_SHUFFLED_FEATURE_NAME,
        instance_name::X_SELF_ENHANCE_BLOCK_NAME,
        instance_name::X_NEIGH_ENHANCE_BLOCK_NAME);

    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, src_nodes_,
                           &indexings[0]);

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

    // build src graph
    BuildGraphNodeFeatureAndBlocks(
        inst, src_nodes_, num_neighbors_, indexing_wrapper_.get(),
        instance_name::X_NODE_FEATURE_NAME,
        instance_name::X_NODE_SHUFFLED_FEATURE_NAME,
        instance_name::X_SELF_BLOCK_NAME, instance_name::X_NEIGH_BLOCK_NAME);

    // build contrastive graph
    BuildGraphNodeFeatureAndBlocks(
        inst, src_nodes_, enhance_num_neighbors_,
        enhance_indexing_wrapper_.get(),
        instance_name::X_ENHANCE_NODE_FEATURE_NAME,
        instance_name::X_ENHANCE_NODE_SHUFFLED_FEATURE_NAME,
        instance_name::X_SELF_ENHANCE_BLOCK_NAME,
        instance_name::X_NEIGH_ENHANCE_BLOCK_NAME);

    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, src_nodes_,
                           &indexings[0]);

    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = src_nodes_;
    inst->set_batch((int)src_nodes_.size());
    return true;
  }

  // BuildGraphNodeFeatureAndBlock
  void BuildGraphNodeFeatureAndBlocks(Instance* inst, const vec_int_t& nodes,
                                      const std::vector<int>& num_neighbors,
                                      IndexingWrapper* indexing_wrapper,
                                      const std::string& node_feature_name,
                                      const std::string& node_shuf_feature_name,
                                      const std::string& node_block_name,
                                      const std::string& neigh_block_name) {
    vec_set_t level_nodes;
    vec_map_neigh_t level_neighs;
    vec_int_t shuffled_nodes;

    // sample subgraph
    flow_->SampleSubGraph(nodes, num_neighbors, &level_nodes, &level_neighs);

    // Fill node and neighbor feature
    flow_->FillLevelNodeFeature(inst, node_feature_name, level_nodes);

    // build shuffled node feature
    if (is_train_) {
      if (shuffle_type_ == 0) {
        flow_->ShuffleNodesInBatch(level_nodes, &shuffled_nodes);
      } else {
        flow_->ShuffleNodesInGlobal(level_nodes, &shuffled_nodes);
      }

      // Fill node and neighbor feature
      flow_->FillNodeFeature(inst, node_shuf_feature_name, shuffled_nodes,
                             false);
    }

    // Fill self and neighbor block
    indexing_wrapper->Clear();
    indexing_wrapper->BuildFrom(level_nodes);
    const auto& indexings = indexing_wrapper->subgraph_indexing(ns_id_);
    flow_->FillSelfAndNeighGraphBlock(inst, node_block_name, neigh_block_name,
                                      level_nodes, level_neighs, indexings,
                                      false);
  }
};

INSTANCE_READER_REGISTER(ContrastiveMultiViewInstReader,
                         "ContrastiveMultiViewInstReader");
INSTANCE_READER_REGISTER(ContrastiveMultiViewInstReader,
                         "contrastive_multi_view_learning_inst_reader");
}  // namespace embedx
