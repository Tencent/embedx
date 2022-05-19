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

#include <deepx_core/common/misc.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor_type.h>

#include <cmath>  // std::sqrt
#include <string>
#include <vector>

#include "src/model/model_zoo_impl.h"

namespace embedx {

class DIN : public ModelZooImpl {
 private:
  std::vector<deepx_core::GroupConfigItem3> user_config_;
  std::vector<deepx_core::GroupConfigItem3> item_config_;
  std::vector<int> deep_dims_;
  int att_hidden_dim_ = 0;
  int hist_size_ = 0;

 private:
  int user_dim_ = 0;
  int item_dim_ = 0;
  int embed_dim_ = 0;
  std::vector<uint16_t> user_group_ids_;
  std::vector<uint16_t> item_group_ids_;

 public:
  DEFINE_MODEL_ZOO_LIKE(DIN);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
      DXINFO("Default model argument %s = %s", k.c_str(), v.c_str());
      return true;
    } else if (k == "user_config") {
      if (!deepx_core::GuessGroupConfig(v, &user_config_, nullptr)) {
        return false;
      }
      if (!deepx_core::CheckFMGroupConfig(user_config_)) {
        return false;
      }
    } else if (k == "item_config") {
      if (!deepx_core::GuessGroupConfig(v, &item_config_, nullptr)) {
        return false;
      }
      if (!deepx_core::CheckFMGroupConfig(item_config_)) {
        return false;
      }
    } else if (k == "deep_dims") {
      if (!deepx_core::ParseDeepDimsAppendOne(v, &deep_dims_, k.c_str())) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "att_hidden_dim") {
      att_hidden_dim_ = std::stod(v);
      if (att_hidden_dim_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "hist_size") {
      hist_size_ = std::stod(v);
      if (hist_size_ <= 0) {
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

  bool PostInitConfig() override {
    embed_dim_ = user_config_.front().embedding_col;
    user_dim_ = (int)user_config_.size() * embed_dim_;
    item_dim_ = (int)item_config_.size() * embed_dim_;

    deep_dims_.emplace(deep_dims_.begin(), user_dim_ + item_dim_ * 2);

    for (const auto& entry : user_config_) {
      user_group_ids_.emplace_back(entry.group_id);
    }
    for (const auto& entry : item_config_) {
      item_group_ids_.emplace_back(entry.group_id);
    }
    return true;
  }

  bool InitGraph(deepx_core::Graph* graph) const override {
    auto* X_user = deepx_core::GetXUser();
    auto* X_cand = deepx_core::GetXCand();
    auto* X_hist_size = deepx_core::GetXHistSize();

    auto* user_emb = deepx_core::DeepGroupEmbeddingLookup2(
        "deep", X_user, user_config_, sparse_);
    auto* cand_emb = deepx_core::DeepGroupEmbeddingLookup2(
        "deep", X_cand, item_config_, sparse_);

    // hist click sequence
    std::vector<deepx_core::GraphNode*> hist_list(hist_size_);
    for (int i = 0; i < hist_size_; ++i) {
      auto* hist_i = deepx_core::GetXHist(i);
      hist_list[i] = deepx_core::DeepGroupEmbeddingLookup2(
          "deep", hist_i, item_config_, sparse_);
    }

    // attention
    deepx_core::GraphNode* attention =
        DINAttention(cand_emb, hist_list, X_hist_size);

    auto* C = deepx_core::Concat("C", {user_emb, cand_emb, attention});
    auto* sfc = deepx_core::StackedFullyConnect("sfc", C, deep_dims_);
    auto Z = deepx_core::BinaryClassificationTarget(sfc, has_w_);
    deepx_core::ReleaseVariable();
    return graph->Compile(Z, 1);
  }

 private:
  deepx_core::GraphNode* DINAttention(
      deepx_core::GraphNode* X_item,
      std::vector<deepx_core::GraphNode*> hist_list,
      deepx_core::GraphNode* X_hist_size) const {
    int num_att_parts = 4;
    auto* att_W1 = deepx_core::GetVariableRandXavier(
        "att_W1",
        deepx_core::Shape(num_att_parts * item_dim_, att_hidden_dim_));
    auto* att_b1 = deepx_core::GetVariableZeros(
        "att_b1", deepx_core::Shape(1, att_hidden_dim_));
    auto* att_W2 = deepx_core::GetVariableRandXavier(
        "att_W2", deepx_core::Shape(att_hidden_dim_, 1));
    auto* att_b2 =
        deepx_core::GetVariableZeros("att_b2", deepx_core::Shape(1, 1));

    std::vector<deepx_core::GraphNode*> att_weights;
    for (int i = 0; i < hist_size_; ++i) {
      auto ii = std::to_string(i);
      std::vector<deepx_core::GraphNode*> att_parts(num_att_parts);
      att_parts[0] = hist_list[i];
      att_parts[1] = X_item;
      att_parts[2] =
          new deepx_core::SubNode("att_sub" + ii, hist_list[i], X_item);
      att_parts[3] =
          new deepx_core::MulNode("att_mul" + ii, hist_list[i], X_item);
      // (batch, 4 * item_dim_)
      auto* att_H0 = new deepx_core::ConcatNode("att_H0" + ii, att_parts);

      // (batch, att_hidden_dim_)
      auto* att_H1 = new deepx_core::FullyConnectNode("att_H1" + ii, att_H0,
                                                      att_W1, att_b1);
      auto* att_H2 = new deepx_core::SigmoidNode("att_H2" + ii, att_H1);
      // (batch, 1)
      auto* att_Z = new deepx_core::FullyConnectNode("att_Z" + ii, att_H2,
                                                     att_W2, att_b2);
      att_weights.emplace_back(att_Z);
    }
    // (batch, hist_size_)
    auto* att_weights_c =
        new deepx_core::ConcatNode("att_weights_c", att_weights);

    // (batch, hist_size_)
    auto* hist_masks =
        new deepx_core::SequenceMaskNode("hist_masks", X_hist_size, hist_size_);

    auto* padding =
        new deepx_core::ConstantLikeNode("padding", hist_masks, -4294967295.0);
    auto* att_weights_c_rep = new deepx_core::WhereNode(
        "att_weights_c_rep", hist_masks, att_weights_c, padding);

    auto* scale = new deepx_core::ConstantNode("scale", deepx_core::Shape(1),
                                               1 / std::sqrt((float)item_dim_));
    auto* att_weights_c_scale = new deepx_core::BroadcastMulNode(
        "att_weights_c_scale", att_weights_c_rep, scale);

    // (batch, hist_size)
    auto* att_weights_c_norm =
        new deepx_core::SoftmaxNode("att_weights_c_norm", att_weights_c_scale);
    auto* att_weights_c_mul =
        new deepx_core::Reshape2Node("att_weights_c_mul", att_weights_c_norm,
                                     deepx_core::Shape(-1, 1, hist_size_));

    // (batch, hist_size * item_dim_);
    auto* hist_list_c =
        new deepx_core::ConcatNode("hist_list_c", hist_list, -1);
    // (batch, hist_size, item_dim_)
    auto* hist_list_z = new deepx_core::Reshape2Node(
        "hist_list_z", hist_list_c,
        deepx_core::Shape(-1, hist_size_, item_dim_));

    // (batch, 1, item_dim_)
    auto* weighted_hist = new deepx_core::MatmulNode(
        "weighted_hist", att_weights_c_mul, hist_list_z);
    return new deepx_core::Reshape2Node("hist", weighted_hist,
                                        deepx_core::Shape(-1, item_dim_));
  }
};

MODEL_ZOO_REGISTER(DIN, "din");

}  // namespace embedx
