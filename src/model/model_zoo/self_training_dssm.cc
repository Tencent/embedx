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
#include <deepx_core/graph/graph_node.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {
namespace {

GraphNode* BuildSelfTrainingLoss(GraphNode* user_embed, GraphNode* item_embed,
                                 double upper_bound, double lower_bound) {
  // compute user-item pair score p
  auto* X_user_id = GetXInput(instance_name::X_SRC_ID_NAME);
  auto* X_item_id = GetXInput(instance_name::X_DST_ID_NAME);
  auto* dot = BatchLookupDot("batch_dot2", X_user_id, X_item_id, user_embed,
                             item_embed);
  auto* prob = Sigmoid("", dot);

  // compute loss : - p * log(p) for all user-item pairs
  auto* entropy = Negate("", Mul("", prob, Log("", prob)));

  // filter out pseudo positive sample (p > upper_bound)
  auto* upper_bounds = ConstantLike("", prob, upper_bound);
  auto* pseudo_positive_sample = GreaterEqual("", prob, upper_bounds);
  auto* positive_loss = Mul("", entropy, pseudo_positive_sample);

  // filter out pseudo negative sample (p < lower_bound)
  auto* lower_bounds = ConstantLike("", prob, lower_bound);
  auto* pseudo_negative_sample = LessEqual("", prob, lower_bounds);
  auto* negative_loss = Mul("", entropy, pseudo_negative_sample);

  auto* loss =
      Add("", ReduceMean("", positive_loss), ReduceMean("", negative_loss));

  return loss;
}

}  // namespace

class SelfTrainingDSSM : public ModelZooImpl {
  using vec_group_config = std::vector<GroupConfigItem3>;

 private:
  std::vector<int> dims_ = {64, 32};
  double relu_alpha_ = 0;
  double tau_ = 0.1;
  int num_neg_ = 5;

  vec_group_config user_config_;
  vec_group_config item_config_;

  // self training (empirical value)
  double lower_bound_ = 0.31;
  double upper_bound_ = 0.69;

 public:
  DEFINE_MODEL_ZOO_LIKE(SelfTrainingDSSM);

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
    } else if (k == "dim") {
      if (!deepx_core::ParseDeepDims(v, &dims_, k.c_str())) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "alpha" || k == "relu_alpha") {
      relu_alpha_ = std::stod(v);
      if (relu_alpha_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      if (num_neg_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "lower_bound") {
      lower_bound_ = std::stod(v);
      if (lower_bound_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "upper_bound") {
      upper_bound_ = std::stod(v);
      if (upper_bound_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "tau") {
      tau_ = std::stod(v);
      if (tau_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Model argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    // user tower
    auto* Xuser_feat = GetXInput(instance_name::X_USER_FEATURE_NAME);
    auto* user_embed = XInputGroupEmbeddingLookup("user_embed", Xuser_feat,
                                                  user_config_, sparse_);
    for (size_t i = 0; i < dims_.size(); ++i) {
      user_embed =
          FullyConnect("user_fc" + std::to_string(i), user_embed, dims_[i]);
      user_embed =
          LeakyRelu("user_ac" + std::to_string(i), user_embed, relu_alpha_);
    }
    user_embed = Normalize2("user_l2_norm", user_embed);

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
    item_embed = Normalize2("item_l2_norm", item_embed);

    auto* Xuser_id = GetXInput(instance_name::X_USER_ID_NAME);
    auto* Xitem_id = GetXInput(instance_name::X_ITEM_ID_NAME);
    auto* dot =
        BatchLookupDot("batch_dot", Xuser_id, Xitem_id, user_embed, item_embed);

    // dot = dot / tau
    // Rescale the dot value to accelarate training with l2 norm embedding.
    auto* tau = deepx_core::ConstantScalar("tau", 1 / tau_);
    dot = BroadcastMul("", dot, tau);

    // [pos1, neg1, ..., neg_(num_neg),
    //  pos2, neg1, ..., neg_(num_neg),
    //  ...]
    dot = Reshape("", dot, Shape(-1, num_neg_ + 1));
    // sampled softmax loss
    auto Z = MultiClassificationTarget("loss", dot, has_w_);

    auto* loss = BuildSelfTrainingLoss(user_embed, item_embed, upper_bound_,
                                       lower_bound_);
    Z[0] = deepx_core::Add("", Z[0], loss);

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

MODEL_ZOO_REGISTER(SelfTrainingDSSM, "SelfTrainingDSSM");
MODEL_ZOO_REGISTER(SelfTrainingDSSM, "self_training_dssm");

}  // namespace embedx
