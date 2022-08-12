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

/************************************************************************/
/* SupGraphsageInstReader */
/************************************************************************/
class SupGraphsageInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  std::vector<int> num_neighbors_;
  bool use_neigh_feat_ = false;
  int num_label_ = 1;
  int max_label_ = 1;
  bool multi_label_ = false;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;

  vec_int_t nodes_;
  std::vector<vecl_t> labels_list_;

  vec_set_t level_nodes_;
  vec_map_neigh_t level_neighs_;

  uint16_t ns_id_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;

 public:
  DEFINE_INSTANCE_READER_LIKE(SupGraphsageInstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    flow_ = NewNeighborAggregationFlow(graph_client);
    return true;
  }

  void PostInit(const std::string& /*node_config*/) override {
    ns_id_ = 0;
    indexing_wrapper_ = IndexingWrapper::Create("");
  }

  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      is_train_ = val;
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      use_neigh_feat_ = val;
    } else if (k == "num_label") {
      num_label_ = std::stoi(v);
      DXCHECK(num_label_ >= 1);
    } else if (k == "multi_label") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      multi_label_ = val;
    } else if (k == "max_label") {
      max_label_ = std::stoi(v);
      DXCHECK(max_label_ >= 1);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

  bool PostInitConfig() override {
    if (!multi_label_) {
      if (num_label_ != 1) {
        DXERROR("num_label != 1 when doing multi classification task.");
        return false;
      }
    }

    return true;
  }

 protected:
  bool GetBatch(Instance* inst) override {
    return is_train_ ? GetTrainBatch(inst) : GetPredictBatch(inst);
  }

  /************************************************************************/
  /* Read batch data from file for training */
  /************************************************************************/
  bool GetTrainBatch(Instance* inst) {
    std::vector<NodeAndLabelValue> values;
    if (!line_parser_.NextBatch<NodeAndLabelValue>(batch_, &values)) {
      line_parser_.Close();
      inst->clear_batch();
      return false;
    }
    nodes_ =
        Collect<NodeAndLabelValue, int_t>(values, &NodeAndLabelValue::node);
    labels_list_ =
        Collect<NodeAndLabelValue, vecl_t>(values, &NodeAndLabelValue::labels);

    // Sample subgraph
    flow_->SampleSubGraph(nodes_, num_neighbors_, &level_nodes_,
                          &level_neighs_);

    // Fill Instance
    // 1. Fill node feature
    flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                level_nodes_);

    // 2. Fill neighbor feature
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(inst, instance_name::X_NEIGH_FEATURE_NAME,
                                   level_nodes_);
    }

    // 3. Fill self And neigbor block
    indexing_wrapper_->Clear();
    indexing_wrapper_->BuildFrom(level_nodes_);
    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                      instance_name::X_NEIGH_BLOCK_NAME,
                                      level_nodes_, level_neighs_, indexings,
                                      false);

    // 4. Fill index
    flow_->FillNodeOrIndex(inst, instance_name::X_NODE_ID_NAME, nodes_,
                           &indexings[0]);

    // 5. Fill label
    flow_->FillLabelAndCheck(inst, deepx_core::Y_NAME, labels_list_, num_label_,
                             max_label_);

    inst->set_batch(nodes_.size());
    return true;
  }

  /************************************************************************/
  /* Read batch data from file for predicting */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    // RAW data
    std::vector<NodeValue> values;
    if (!line_parser_.NextBatch<NodeValue>(batch_, &values)) {
      line_parser_.Close();
      inst->clear_batch();
      return false;
    }
    nodes_ = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // Sample subgraph
    flow_->SampleSubGraph(nodes_, num_neighbors_, &level_nodes_,
                          &level_neighs_);

    // Fill Instance
    // 1. Fill node feature
    flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                level_nodes_);

    // 2. Fill neighbor feature
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(inst, instance_name::X_NEIGH_FEATURE_NAME,
                                   level_nodes_);
    }

    // 3. Fill self And neigbor block
    indexing_wrapper_->Clear();
    indexing_wrapper_->BuildFrom(level_nodes_);
    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                      instance_name::X_NEIGH_BLOCK_NAME,
                                      level_nodes_, level_neighs_, indexings,
                                      false);

    // 4. Fill index
    flow_->FillNodeOrIndex(inst, instance_name::X_NODE_ID_NAME, nodes_,
                           &indexings[0]);

    // input : predict_node
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = nodes_;

    inst->set_batch(nodes_.size());
    return true;
  }
};

INSTANCE_READER_REGISTER(SupGraphsageInstReader, "SupGraphsageInstReader");
INSTANCE_READER_REGISTER(SupGraphsageInstReader, "sup_graphsage");

}  // namespace embedx
