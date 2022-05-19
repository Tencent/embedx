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

class DSSM : public ModelZooImpl {
  using vec_group_config = std::vector<GroupConfigItem3>;

 private:
  std::vector<int> dims_ = {64, 32};
  double relu_alpha_ = 0;
  int num_neg_ = 5;

  vec_group_config user_config_;
  vec_group_config item_config_;

 public:
  DEFINE_MODEL_ZOO_LIKE(DSSM);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "user_config") {
      if (!GuessGroupConfig(v, &user_config_, nullptr)) {
        return false;
      }
    } else if (k == "item_config") {
      if (!GuessGroupConfig(v, &item_config_, nullptr)) {
        return false;
      }
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
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Model argument: %s = %s", k.c_str(), v.c_str());
    return true;
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    // user tower
    auto* Xuser_feat = GetXInput(instance_name::X_USER_FEATURE_NAME);
    // TODO(Chowy): Analyze the difference of using XInputGroupEmbeddingLookup
    // and XInputGroupEmbeddingLookup2 initialization.
    auto* user_embed = XInputGroupEmbeddingLookup("user_embed", Xuser_feat,
                                                  user_config_, sparse_);
    for (size_t i = 0; i < dims_.size(); ++i) {
      user_embed =
          FullyConnect("user_fc" + std::to_string(i), user_embed, dims_[i]);
      user_embed =
          LeakyRelu("user_ac" + std::to_string(i), user_embed, relu_alpha_);
    }

    // item tower
    auto* Xitem_feat = GetXInput(instance_name::X_ITEM_FEATURE_NAME);
    auto* item_embed = XInputGroupEmbeddingLookup2("item_embed", Xitem_feat,
                                                   item_config_, sparse_);
    for (size_t i = 0; i < dims_.size(); ++i) {
      item_embed =
          FullyConnect("item_fc" + std::to_string(i), item_embed, dims_[i]);
      item_embed =
          LeakyRelu("item_ac" + std::to_string(i), item_embed, relu_alpha_);
    }

    auto* Xuser_id = GetXInput(instance_name::X_USER_ID_NAME);
    auto* Xitem_id = GetXInput(instance_name::X_ITEM_ID_NAME);
    auto* dot =
        BatchLookupDot("batch_dot", Xuser_id, Xitem_id, user_embed, item_embed);
    // [pos1, neg1, ..., neg_(num_neg),
    //  pos2, neg1, ..., neg_(num_neg),
    //  ...]
    dot = Reshape("", dot, Shape(-1, num_neg_ + 1));
    // sampled softmax loss
    auto Z = MultiClassificationTarget("loss", dot, has_w_);

    Z.emplace_back(user_embed);
    Z.emplace_back(item_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: dump classification prob
    // Z[2]: dump user embedding
    // Z[3]: dump item embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(DSSM, "DSSM");
MODEL_ZOO_REGISTER(DSSM, "dssm");

}  // namespace embedx
