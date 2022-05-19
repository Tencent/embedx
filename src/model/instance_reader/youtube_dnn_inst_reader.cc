// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yong Zhou (zhouyongnju@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include <vector>

#include "src/io/value.h"
#include "src/model/data_flow/deep_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"

namespace embedx {

class YoutubeDNNInstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;

 private:
  DeepFlow flow_;
  // for training/predicting data
  vec_int_t pos_items_;
  std::vector<vec_pair_t> feats_list_;
  std::vector<vec_int_t> neg_items_list_;

 public:
  DEFINE_INSTANCE_READER_LIKE(YoutubeDNNInstReader);

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

  /************************************************************************/
  /* Read batch data from file for training */
  /************************************************************************/
  bool GetTrainBatch(Instance* inst) {
    std::vector<AdjValue> values;
    if (!NextInstanceBatch<AdjValue>(inst, batch_, &values)) {
      return false;
    }
    pos_items_ = Collect<AdjValue, int_t>(values, &AdjValue::node);

    // shared sampling
    DXCHECK(deep_client_->SharedSampleNegative(num_neg_, pos_items_, pos_items_,
                                               &neg_items_list_));
    // Fill Instance
    // 1. Fill user feature
    vec_int_t* user_nodes_ptr = nullptr;
    feats_list_ = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);
    flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                          user_nodes_ptr, feats_list_);

    // 2. Fill edge and label
    auto indexing_func = [](int_t node) { return node; };
    flow_.FillEdgeAndLabel(inst, instance_name::X_USER_ID_NAME,
                           instance_name::X_ITEM_NODE_NAME, deepx_core::Y_NAME,
                           pos_items_, neg_items_list_, indexing_func,
                           indexing_func);

    inst->set_batch(pos_items_.size());
    return true;
  }

  /************************************************************************/
  /* Read batch data from file for predicting */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    std::vector<AdjValue> values;
    if (!NextInstanceBatch<AdjValue>(inst, batch_, &values)) {
      return false;
    }
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = Collect<AdjValue, int_t>(values, &AdjValue::node);

    // Fill Instance
    // 1. Fill user feature
    // When predicting user embeddings, only the features are used and the label
    // is a placeholder which can be just set as 0 or 1.
    vec_int_t* user_nodes_ptr = nullptr;
    feats_list_ = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);
    flow_.FillNodeFeature(inst, instance_name::X_USER_FEATURE_NAME,
                          user_nodes_ptr, feats_list_);

    // 2. Fill item node
    // When predicting item embeddings, the predict_nodes repesents which item
    // to dump and the features are placeholders which can be just set as 1:1.
    flow_.FillNodeOrIndex(inst, instance_name::X_ITEM_NODE_NAME,
                          *predict_node_ptr, nullptr);

    inst->set_batch(predict_node_ptr->size());
    return true;
  }
};

INSTANCE_READER_REGISTER(YoutubeDNNInstReader, "YoutubeDNNInstReader");
INSTANCE_READER_REGISTER(YoutubeDNNInstReader, "youtube_dnn");

}  // namespace embedx
