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

#include "src/io/indexing.h"
#include "src/io/value.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {

class UnsupGraphsageInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;
  std::vector<int> num_neighbors_;
  bool use_neigh_feat_ = false;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;

  vec_int_t src_nodes_;
  vec_int_t dst_nodes_;
  std::vector<vec_int_t> neg_nodes_list_;

  vec_int_t merged_nodes_;
  vec_set_t level_nodes_;
  vec_map_neigh_t level_neighbors_;
  std::vector<Indexing> indexings_;

 public:
  DEFINE_INSTANCE_READER_LIKE(UnsupGraphsageInstReader);

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
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 0 || val == 1);
      use_neigh_feat_ = val;
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
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
    // batch edges
    std::vector<EdgeValue> values;
    if (!NextInstanceBatch<EdgeValue>(inst, batch_, &values)) {
      return false;
    }
    src_nodes_ = Collect<EdgeValue, int_t>(values, &EdgeValue::src_node);
    dst_nodes_ = Collect<EdgeValue, int_t>(values, &EdgeValue::dst_node);

    // negative sampling
    DXCHECK(graph_client_->SharedSampleNegative(num_neg_, dst_nodes_,
                                                dst_nodes_, &neg_nodes_list_));

    // merge nodes to avoid repeated construction of node computation graph.
    merged_nodes_.clear();
    flow_->MergeTo(src_nodes_, &merged_nodes_);
    flow_->MergeTo(dst_nodes_, &merged_nodes_);
    flow_->MergeTo(neg_nodes_list_, &merged_nodes_);

    // sample subgraph
    flow_->SampleSubGraph(merged_nodes_, num_neighbors_, &level_nodes_,
                          &level_neighbors_);

    // Fill instance
    // 1. Fill node and neighbor feature
    flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                level_nodes_);
    flow_->FillLevelNeighFeature(inst, instance_name::X_NEIGH_FEATURE_NAME,
                                 level_nodes_);

    // 2. Fill self and neighbor block
    inst_util::CreateIndexings(level_nodes_, &indexings_);
    flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                      instance_name::X_NEIGH_BLOCK_NAME,
                                      level_nodes_, level_neighbors_,
                                      indexings_, false);

    // 3. Fill edge and label
    auto indexing_func = [this](int_t node) {
      int index = indexings_[0].Get(node);
      DXCHECK(index >= 0);
      return (int_t)index;
    };

    flow_->FillEdgeAndLabel(inst, instance_name::X_SRC_ID_NAME,
                            instance_name::X_DST_ID_NAME, deepx_core::Y_NAME,
                            src_nodes_, dst_nodes_, neg_nodes_list_,
                            indexing_func, indexing_func);

    inst->set_batch(src_nodes_.size());
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
    src_nodes_ = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // Sample subgraph
    flow_->SampleSubGraph(src_nodes_, num_neighbors_, &level_nodes_,
                          &level_neighbors_);

    // Fill Instance
    // 1. Fill node and neighbor feature
    flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                level_nodes_);
    flow_->FillLevelNeighFeature(inst, instance_name::X_NEIGH_FEATURE_NAME,
                                 level_nodes_);

    // 2. Fill self and neighbor block
    inst_util::CreateIndexings(level_nodes_, &indexings_);
    flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                      instance_name::X_NEIGH_BLOCK_NAME,
                                      level_nodes_, level_neighbors_,
                                      indexings_, false);

    // 3. Fill index
    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, src_nodes_,
                           &indexings_[0]);

    // 4. Fill node
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = src_nodes_;
    inst->set_batch((int)src_nodes_.size());
    return true;
  }
};

INSTANCE_READER_REGISTER(UnsupGraphsageInstReader, "UnsupGraphsageInstReader");
INSTANCE_READER_REGISTER(UnsupGraphsageInstReader, "unsup_graphsage");
}  // namespace embedx
