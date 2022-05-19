// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yong Zhou (zhouyongnju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include <vector>

#include "src/io/indexing.h"
#include "src/io/value.h"
#include "src/model/data_flow/deep_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {

class SelfTrainingDSSMInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;
  bool add_node_ = true;

 private:
  // for training/predicting data
  DeepFlow flow_;
  vec_int_t pos_items_;
  std::vector<vec_pair_t> feats_list_;
  std::vector<vec_int_t> neg_items_list_;

  // for item feature
  vec_int_t unique_items_;
  Indexing item_indexing_;
  std::vector<vec_pair_t> item_feats_list_;

  // for self training
  int num_self_training_ = 500;
  std::vector<vec_int_t> self_training_items_list_;

 public:
  DEFINE_INSTANCE_READER_LIKE(SelfTrainingDSSMInstReader);

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
    } else if (k == "add_node") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      add_node_ = val;
    } else if (k == "num_self_training") {
      num_self_training_ = std::stoi(v);
      DXCHECK(num_self_training_ > 0);
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
    std::vector<AdjValue> values;
    if (!NextInstanceBatch<AdjValue>(inst, batch_, &values)) {
      return false;
    }
    pos_items_ = Collect<AdjValue, int_t>(values, &AdjValue::node);

    DXCHECK(deep_client_->SharedSampleNegative(num_neg_, pos_items_, pos_items_,
                                               &neg_items_list_));
    DXCHECK(deep_client_->SharedSampleNegative(num_self_training_, pos_items_,
                                               pos_items_,
                                               &self_training_items_list_));

    // Fill Instance
    // 1. Fill user feature
    vec_int_t* user_nodes_ptr = nullptr;
    feats_list_ = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);
    flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                          user_nodes_ptr, feats_list_);

    // 2. Fill item feature
    inst_util::RemoveDuplicateItems(pos_items_, neg_items_list_,
                                    &unique_items_);
    inst_util::RemoveDuplicateItems(unique_items_, self_training_items_list_,
                                    &unique_items_);
    vec_int_t* item_nodes_ptr = add_node_ ? &unique_items_ : nullptr;
    DXCHECK(deep_client_->LookupItemFeature(unique_items_, &item_feats_list_));
    flow_.FillNodeFeature(inst, instance_name::X_ITEM_FEATURE_NAME,
                          item_nodes_ptr, item_feats_list_);

    // 3. Fill edge and label
    inst_util::CreateIndexing(unique_items_, &item_indexing_);
    auto indexing_func = [this](int_t node) {
      int index = item_indexing_.Get(node);
      DXCHECK(index >= 0);
      return (int_t)index;
    };
    flow_.FillEdgeAndLabel(inst, instance_name::X_USER_ID_NAME,
                           instance_name::X_ITEM_ID_NAME, deepx_core::Y_NAME,
                           pos_items_, neg_items_list_, indexing_func,
                           indexing_func);

    // 4. Fill self training edge
    flow_.FillEdge(inst, instance_name::X_SRC_ID_NAME,
                   instance_name::X_DST_ID_NAME, pos_items_,
                   self_training_items_list_, indexing_func);

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
    flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                          user_nodes_ptr, feats_list_);

    // 2. Fill item feature
    // When predicting item embeddings, the input feature is item feature
    vec_int_t* item_nodes_ptr = add_node_ ? predict_node_ptr : nullptr;
    flow_.FillNodeFeature(inst, instance_name::X_ITEM_FEATURE_NAME,
                          item_nodes_ptr, feats_list_);

    inst->set_batch(predict_node_ptr->size());
    return true;
  }
};

INSTANCE_READER_REGISTER(SelfTrainingDSSMInstReader,
                         "SelfTrainingDSSMInstReader");
INSTANCE_READER_REGISTER(SelfTrainingDSSMInstReader, "SelfTrainingDSSM");
INSTANCE_READER_REGISTER(SelfTrainingDSSMInstReader, "self_training_dssm");

}  // namespace embedx
