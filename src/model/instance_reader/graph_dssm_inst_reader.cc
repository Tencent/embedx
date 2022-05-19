// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>

#include <vector>

#include "src/io/indexing.h"
#include "src/io/io_util.h"
#include "src/io/value.h"
#include "src/model/data_flow/deep_flow.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {
namespace {

void ParseUserFrom(const std::vector<vec_pair_t>& user_feats,
                   uint16_t user_group, vec_int_t* user_nodes) {
  user_nodes->clear();
  for (const auto& user_feat : user_feats) {
    for (const auto& feat : user_feat) {
      auto feat_id = feat.first;
      auto group = io_util::GetNodeType(feat_id);
      if (group == user_group) {
        user_nodes->emplace_back(feat_id);
      }
    }
  }
  DXCHECK(user_nodes->size() == user_feats.size());
}

}  // namespace

class GraphDSSMInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;
  std::vector<int> num_neighbors_;
  bool add_node_ = true;
  uint16_t user_group_ = 0;

 private:
  DeepFlow deep_flow_;
  std::unique_ptr<NeighborAggregationFlow> na_flow_;

  // for training/predicting data
  vec_int_t pos_items_;
  std::vector<vec_pair_t> feats_list_;
  std::vector<vec_int_t> neg_items_list_;

  // for item feature
  vec_int_t unique_items_;
  Indexing item_indexing_;

  vec_set_t level_nodes_;
  vec_map_neigh_t level_neighbors_;
  std::vector<Indexing> indexings_;

 public:
  DEFINE_INSTANCE_READER_LIKE(GraphDSSMInstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    na_flow_ = NewNeighborAggregationFlow(graph_client);
    return true;
  }

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      is_train_ = val;
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      DXCHECK(num_neg_ > 0);
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
    } else if (k == "add_node") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      add_node_ = val;
    } else if (k == "user_group_id") {
      user_group_ = std::stoi(v);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

 public:
  bool GetBatch(Instance* inst) override {
    return is_train_ ? GetTrainBatch(inst) : GetPredictBatch(inst);
  }

  bool GetTrainBatch(Instance* inst) {
    // pos_item user_feat1:val1 user_feat2:val2 ...
    std::vector<AdjValue> values;
    if (!NextInstanceBatch<AdjValue>(inst, batch_, &values)) {
      return false;
    }
    pos_items_ = Collect<AdjValue, int_t>(values, &AdjValue::node);
    feats_list_ = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);

    // Parse user nodes from feats list
    vec_int_t user_nodes;
    ParseUserFrom(feats_list_, user_group_, &user_nodes);

    // negative sampling
    DXCHECK(deep_client_->SharedSampleNegative(num_neg_, pos_items_, pos_items_,
                                               &neg_items_list_));

    na_flow_->SampleSubGraph(user_nodes, num_neighbors_, &level_nodes_,
                             &level_neighbors_);

    // Fill Instance
    // 1. Fill node and neighbor feature
    na_flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                   level_nodes_);
    na_flow_->FillLevelNeighFeature(inst, instance_name::X_NEIGH_FEATURE_NAME,
                                    level_nodes_);

    // 2. Fill self and neighbor block
    inst_util::CreateIndexings(level_nodes_, &indexings_);
    na_flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                         instance_name::X_NEIGH_BLOCK_NAME,
                                         level_nodes_, level_neighbors_,
                                         indexings_, false);

    // 3. Fill index
    na_flow_->FillNodeOrIndex(inst, instance_name::X_SRC_ID_NAME, user_nodes,
                              &indexings_[0]);

    // 4. Fill user feature
    vec_int_t* user_nodes_ptr = nullptr;
    deep_flow_.FillNodeOrIndex(inst, instance_name::X_USER_NODE_NAME,
                               user_nodes, nullptr);
    deep_flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                               user_nodes_ptr, feats_list_);

    // 5. Fill item feature
    inst_util::RemoveDuplicateItems(pos_items_, neg_items_list_,
                                    &unique_items_);
    na_flow_->FillNodeFeature(inst, instance_name::X_ITEM_FEATURE_NAME,
                              unique_items_, add_node_);

    // 6. Fill edge and label
    inst_util::CreateIndexing(unique_items_, &item_indexing_);
    auto indexing_func = [this](int_t node) {
      int index = item_indexing_.Get(node);
      DXCHECK(index >= 0);
      return (int_t)index;
    };
    deep_flow_.FillEdgeAndLabel(inst, instance_name::X_USER_ID_NAME,
                                instance_name::X_ITEM_ID_NAME,
                                deepx_core::Y_NAME, pos_items_, neg_items_list_,
                                indexing_func, indexing_func);

    inst->set_batch(pos_items_.size());
    return true;
  }

  bool GetPredictBatch(Instance* inst) {
    std::vector<AdjValue> values;
    if (!NextInstanceBatch<AdjValue>(inst, batch_, &values)) {
      return false;
    }
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = Collect<AdjValue, int_t>(values, &AdjValue::node);
    feats_list_ = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);

    // Fill Instance
    // 1. Fill user feature
    // When predicting user embeddings, the input feature is user feature
    vec_int_t* user_nodes_ptr = nullptr;
    deep_flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                               user_nodes_ptr, feats_list_);

    // 2. Fill item feature
    // When predicting item embeddings, the input feature is item feature
    auto* item_nodes_ptr = add_node_ ? predict_node_ptr : nullptr;
    deep_flow_.FillNodeFeature(inst, instance_name::X_ITEM_FEATURE_NAME,
                               item_nodes_ptr, feats_list_);

    inst->set_batch(predict_node_ptr->size());
    return true;
  }
};

INSTANCE_READER_REGISTER(GraphDSSMInstReader, "GraphDSSMInstReader");
INSTANCE_READER_REGISTER(GraphDSSMInstReader, "GraphDSSM");
INSTANCE_READER_REGISTER(GraphDSSMInstReader, "graph_dssm");

}  // namespace embedx
