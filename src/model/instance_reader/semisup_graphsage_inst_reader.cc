// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>

#include <algorithm>  // std::shuffle, std::max, std::min
#include <random>     // std::random_device, std::default_random_engine
#include <vector>

#include "src/io/indexing_wrapper.h"
#include "src/io/value.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {
namespace {

void DiscardNodeAndLabel(vec_int_t* nodes, std::vector<vecl_t>* labels_list,
                         int num_remain) {
  static thread_local std::random_device rd;
  auto seed = (uint64_t)rd();
  static thread_local std::default_random_engine rng(seed);
  std::shuffle(nodes->begin(), nodes->end(), rng);
  rng.seed(seed);
  std::shuffle(labels_list->begin(), labels_list->end(), rng);
  nodes->erase(nodes->begin() + num_remain, nodes->end());
  labels_list->erase(labels_list->begin() + num_remain, labels_list->end());
}

}  // namespace

/************************************************************************/
/* SemisupGraphsageInstReader */
/************************************************************************/
class SemisupGraphsageInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  bool use_neigh_feat_ = false;
  int num_neg_ = 5;
  std::vector<int> num_neighbors_;
  int min_batch_ = 16;
  int num_label_ = 1;
  int max_label_ = 1;
  bool multi_label_ = false;
  double discard_prob_ = 0.0;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;

  vec_int_t src_nodes_;
  vec_int_t dst_nodes_;
  vec_int_t nodes_;
  std::vector<vecl_t> labels_list_;

  std::vector<vec_int_t> neg_nodes_list_;
  vec_int_t merged_nodes_;
  vec_set_t level_nodes_;
  vec_map_neigh_t level_neighbors_;

  uint16_t ns_id_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;

 public:
  DEFINE_INSTANCE_READER_LIKE(SemisupGraphsageInstReader);

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

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      is_train_ = val;
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      use_neigh_feat_ = val;
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      DXCHECK(num_neg_ >= 1);
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
    } else if (k == "min_batch_") {
      min_batch_ = std::stoi(v);
      DXCHECK(min_batch_ >= 1);
    } else if (k == "num_label") {
      num_label_ = std::stoi(v);
      DXCHECK(num_label_ >= 1);
    } else if (k == "max_label") {
      max_label_ = std::stoi(v);
      DXCHECK(max_label_ >= 1);
    } else if (k == "multi_label") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      multi_label_ = val;
    } else if (k == "discard_prob") {
      discard_prob_ = std::stod(v);
      DXCHECK(0.0 <= discard_prob_ && discard_prob_ <= 1.0);
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
    // batch train nodes
    std::vector<EdgeAndLabelValue> values;
    if (!NextInstanceBatch<EdgeAndLabelValue>(inst, batch_, &values)) {
      return false;
    }
    src_nodes_ =
        Collect<EdgeAndLabelValue, int_t>(values, &EdgeAndLabelValue::src_node);
    dst_nodes_ =
        Collect<EdgeAndLabelValue, int_t>(values, &EdgeAndLabelValue::dst_node);
    nodes_ =
        Collect<EdgeAndLabelValue, int_t>(values, &EdgeAndLabelValue::node);
    labels_list_ =
        Collect<EdgeAndLabelValue, vecl_t>(values, &EdgeAndLabelValue::labels);

    // negative sampling
    DXCHECK(graph_client_->SharedSampleNegative(num_neg_, dst_nodes_,
                                                dst_nodes_, &neg_nodes_list_));

    // discard 'nodes_' and 'labels_list_'
    int num_remain =
        std::max(min_batch_, (int)((1.0 - discard_prob_) * nodes_.size()));
    num_remain = std::min(num_remain, (int)nodes_.size());
    DiscardNodeAndLabel(&nodes_, &labels_list_, num_remain);

    // merge nodes to avoid repeated construction of node computation graph.
    merged_nodes_.clear();
    flow_->MergeTo(src_nodes_, &merged_nodes_);
    flow_->MergeTo(dst_nodes_, &merged_nodes_);
    flow_->MergeTo(neg_nodes_list_, &merged_nodes_);
    flow_->MergeTo(nodes_, &merged_nodes_);

    // Sample subgraph
    flow_->SampleSubGraph(merged_nodes_, num_neighbors_, &level_nodes_,
                          &level_neighbors_);

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
                                      level_nodes_, level_neighbors_, indexings,
                                      false);

    // 4. Fill index
    flow_->FillNodeOrIndex(inst, instance_name::X_NODE_ID_NAME, nodes_,
                           &indexings[0]);

    // 5. Fill label
    flow_->FillLabelAndCheck(inst, deepx_core::Y_NAME, labels_list_, num_label_,
                             max_label_);

    // 6. Fill edge and label
    auto indexing_func = [this](int_t node) {
      return indexing_wrapper_->GlobalGet(node);
    };

    flow_->FillEdgeAndLabel(
        inst, instance_name::X_SRC_ID_NAME, instance_name::X_DST_ID_NAME,
        instance_name::Y_UNSUPVISED_NAME, src_nodes_, dst_nodes_,
        neg_nodes_list_, indexing_func, indexing_func);

    inst->set_batch((int)src_nodes_.size());
    return true;
  }

  /************************************************************************/
  /* Read batch data from file for predicting */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    // batch predict nodes
    std::vector<NodeValue> values;
    if (!NextInstanceBatch<NodeValue>(inst, batch_, &values)) {
      return false;
    }
    nodes_ = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // Sample subgraph
    flow_->SampleSubGraph(nodes_, num_neighbors_, &level_nodes_,
                          &level_neighbors_);

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
                                      level_nodes_, level_neighbors_, indexings,
                                      false);

    // 4. Fill index
    flow_->FillNodeOrIndex(inst, instance_name::X_NODE_ID_NAME, nodes_,
                           &indexings[0]);

    // 5. Fill node
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = nodes_;
    inst->set_batch((int)nodes_.size());
    return true;
  }
};

INSTANCE_READER_REGISTER(SemisupGraphsageInstReader,
                         "SemisupGraphsageInstReader");
INSTANCE_READER_REGISTER(SemisupGraphsageInstReader, "semisup_graphsage");

}  // namespace embedx
