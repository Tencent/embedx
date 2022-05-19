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

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class YoutubeDNN : public ModelZooImpl {
  using group_config = GroupConfigItem3;
  using vec_group_config = std::vector<group_config>;

 private:
  std::vector<int> dims_ = {64, 32};
  double relu_alpha_ = 0;
  int num_neg_ = 5;

  // label_group_id refers to the clicked items
  uint16_t label_group_id_ = 0;

  // user: multi feature groups
  // item: one feature group
  vec_group_config user_config_;
  group_config item_config_;

 public:
  DEFINE_MODEL_ZOO_LIKE(YoutubeDNN);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "alpha" || k == "relu_alpha") {
      relu_alpha_ = std::stod(v);
      if (relu_alpha_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "dim") {
      if (!deepx_core::ParseDeepDims(v, &dims_, k.c_str())) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      if (num_neg_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "label_group_id") {
      label_group_id_ = std::stoi(v);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Model argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

  bool PostInitConfig() override {
    if (items_.empty()) {
      DXERROR("Please specify feature group config.");
      return false;
    }

    for (const auto& config : items_) {
      if (config.group_id == label_group_id_) {
        item_config_ = std::move(config);
      } else {
        user_config_.emplace_back(std::move(config));
      }
    }

    // user embedding dim = dims_.back()
    if (item_config_.embedding_col != dims_.back()) {
      DXERROR(
          "The dim of item embed (%d) should be the same as user embed (%d).",
          item_config_.embedding_col, dims_.back());
      return false;
    }

    return !user_config_.empty();
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    auto* Xuser_feat = GetXInput(instance_name::X_USER_FEATURE_NAME);
    auto* user_embed = XInputGroupEmbeddingLookup("usr_embed", Xuser_feat,
                                                  user_config_, sparse_);
    for (size_t i = 0; i < dims_.size(); ++i) {
      user_embed = FullyConnect("fc" + std::to_string(i), user_embed, dims_[i]);
      user_embed = LeakyRelu("ac" + std::to_string(i), user_embed, relu_alpha_);
    }

    int tensor_type = sparse_ ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
    auto* item_embed = GetVariable(
        "item_embed",
        Shape(item_config_.embedding_row, item_config_.embedding_col),
        tensor_type, TENSOR_INITIALIZER_TYPE_RAND,
        -1.0 / item_config_.embedding_col, 1.0 / item_config_.embedding_col);

    auto* Xuser_id = GetXInput(instance_name::X_USER_ID_NAME);
    auto* Xitem_node = GetXInput(instance_name::X_ITEM_NODE_NAME);
    auto* dot = BatchLookupDot("batch_dot", Xuser_id, Xitem_node, user_embed,
                               item_embed);
    // [pos1, neg1, ..., neg_(num_neg),
    //  pos2, neg1, ..., neg_(num_neg),
    //  ...]
    dot = Reshape("", dot, Shape(-1, num_neg_ + 1));
    // sampled softmax loss
    auto Z = MultiClassificationTarget("loss", dot, has_w_);

    Z.emplace_back(user_embed);
    auto* dump_item_embed =
        EmbeddingLookup("dump_item_embed", Xitem_node, item_embed);
    Z.emplace_back(dump_item_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: dump classification prob
    // Z[2]: dump user embedding
    // Z[3]: dump item embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(YoutubeDNN, "YoutubeDNN");
MODEL_ZOO_REGISTER(YoutubeDNN, "youtube_dnn");

}  // namespace embedx
