// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhitao Wang (wztzenk@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class DeepGraphContrastive : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  int num_mlp_layers_ = 2;
  double relu_alpha_ = 0;
  double tau_ = 0.5;
  bool use_neigh_feat_ = false;

 public:
  DEFINE_MODEL_ZOO_LIKE(DeepGraphContrastive);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
      DXINFO("Default model argument %s = %s", k.c_str(), v.c_str());
      return true;
    } else if (k == "depth") {
      depth_ = std::stoi(v);
      if (depth_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "alpha" || k == "relu_alpha") {
      relu_alpha_ = std::stod(v);
      if (relu_alpha_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "dim") {
      dim_ = std::stoi(v);
      if (dim_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "num_mlp_layers") {
      num_mlp_layers_ = std::stoi(v);
      if (num_mlp_layers_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "tau") {
      tau_ = std::stod(v);
      if (tau_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "use_neigh_feat") {
      use_neigh_feat_ = std::stoi(v);
      if (use_neigh_feat_ != 0 && use_neigh_feat_ != 1) {
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
    if (items_.empty()) {
      DXERROR("Please specify feature group config.");
      return false;
    }

    return true;
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    // graph nodes for predicting
    auto* Xnode_feat = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    GraphNode* Xneigh_feat = nullptr;
    if (use_neigh_feat_) {
      Xneigh_feat = GetXInput(instance_name::X_NEIGH_FEATURE_NAME);
    }
    const auto& self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth_);
    const auto& neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth_);
    auto* hidden =
        GraphSageEncoder("gnn_encoder", items_, Xnode_feat, Xneigh_feat,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);
    auto* Xsrc_id = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* src_embed = HiddenLookup("", Xsrc_id, hidden);

    // left augment graph nodes for training
    auto* Xnode_left_feat =
        GetXInput(instance_name::X_NODE_LEFT_MASKED_FEATURE_NAME);
    GraphNode* Xneigh_left_feat = nullptr;
    if (use_neigh_feat_) {
      Xneigh_left_feat =
          GetXInput(instance_name::X_NEIGH_LEFT_MASKED_FEATURE_NAME);
    }
    const auto& self_left_blocks =
        GetXBlockInputs(instance_name::X_SELF_LEFT_DROPPED_BLOCK_NAME, depth_);
    const auto& neigh_left_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_LEFT_DROPPED_BLOCK_NAME, depth_);
    auto* left_hidden = GraphSageEncoder(
        "gnn_encoder", items_, Xnode_left_feat, Xneigh_left_feat,
        self_left_blocks, neigh_left_blocks, sparse_, relu_alpha_, dim_);

    // right augment graph nodes for training
    auto* Xnode_right_feat =
        GetXInput(instance_name::X_NODE_RIGHT_MASKED_FEATURE_NAME);
    GraphNode* Xneigh_right_feat = nullptr;
    if (use_neigh_feat_) {
      Xneigh_right_feat =
          GetXInput(instance_name::X_NEIGH_RIGHT_MASKED_FEATURE_NAME);
    }
    const auto& self_right_blocks =
        GetXBlockInputs(instance_name::X_SELF_RIGHT_DROPPED_BLOCK_NAME, depth_);
    const auto& neigh_right_blocks = GetXBlockInputs(
        instance_name::X_NEIGH_RIGHT_DROPPED_BLOCK_NAME, depth_);
    auto* right_hidden = GraphSageEncoder(
        "gnn_encoder", items_, Xnode_right_feat, Xneigh_right_feat,
        self_right_blocks, neigh_right_blocks, sparse_, relu_alpha_, dim_);

    // get gnn embedding in left and right augment graphs
    auto* left_embed = HiddenLookup("", Xsrc_id, left_hidden);
    auto* right_embed = HiddenLookup("", Xsrc_id, right_hidden);

    // mlps
    for (int i = 1; i <= num_mlp_layers_; ++i) {
      left_embed = deepx_core::FullyConnect("mlp_" + std::to_string(i),
                                            left_embed, dim_ * 2);
      right_embed = deepx_core::FullyConnect("mlp_" + std::to_string(i),
                                             right_embed, dim_ * 2);
      if (i != num_mlp_layers_) {
        left_embed = deepx_core::Elu("", left_embed, 1.0);
        right_embed = deepx_core::Elu("", right_embed, 1.0);
      }
    }

    // normalize
    auto* left_normed_embed = deepx_core::Normalize2("", left_embed, 1);
    auto* right_normed_embed = deepx_core::Normalize2("", right_embed, 1);

    // similarity matrix, matrix size: batch_size x batch_size
    auto* between_sim_matrix =
        deepx_core::GEMM("", left_normed_embed, right_normed_embed, 0, 1);
    auto* left_own_sim_matrix =
        deepx_core::GEMM("", left_normed_embed, left_normed_embed, 0, 1);
    auto* right_own_sim_matrix =
        deepx_core::GEMM("", right_normed_embed, right_normed_embed, 0, 1);

    // tau: temperature of exponential
    GraphNode* tau_matrix =
        deepx_core::ConstantLike("", between_sim_matrix, tau_);

    // exponential operation on similarity matrix with temperature tau
    between_sim_matrix = deepx_core::Exp(
        "", deepx_core::Div("", between_sim_matrix, tau_matrix));
    left_own_sim_matrix = deepx_core::Exp(
        "", deepx_core::Div("", left_own_sim_matrix, tau_matrix));
    right_own_sim_matrix = deepx_core::Exp(
        "", deepx_core::Div("", right_own_sim_matrix, tau_matrix));

    // diag of similarity matrix
    auto* between_sim_diag =
        deepx_core::MatrixBandPart("", between_sim_matrix, 0, 0);
    auto* left_own_sim_diag =
        deepx_core::MatrixBandPart("", left_own_sim_matrix, 0, 0);
    auto* right_own_sim_diag =
        deepx_core::MatrixBandPart("", right_own_sim_matrix, 0, 0);

    // Similarity values of positive samples, which are pairs of nodes
    // with same ids across left and right augment graphs.
    auto* pos_exp = deepx_core::ReduceSum("", between_sim_diag, 1, 1);

    // Sum of between similarity matrix, which equals to
    // sum of positive samples + sum of between negative samples.
    // Between negative samples are pairs of nodes with different ids
    // across left and right augment graphs.
    auto* between_sim_sum = deepx_core::ReduceSum("", between_sim_matrix, 1, 0);

    // Similarity sum of left negative samples, which are pairs of nodes
    // with different ids in left graph.
    auto* left_neg_sim_sum = deepx_core::ReduceSum(
        "", deepx_core::Sub("", left_own_sim_matrix, left_own_sim_diag), 1, 0);

    // Similarity sum of right negative samples, which are pairs of nodes
    // with different ids in right graph.
    auto* right_neg_sim_sum = deepx_core::ReduceSum(
        "", deepx_core::Sub("", right_own_sim_matrix, right_own_sim_diag), 1,
        0);

    // Logits
    auto* left_logits = deepx_core::BroadcastDiv(
        "", pos_exp, deepx_core::Add("", between_sim_sum, left_neg_sim_sum));
    auto* right_logtis = deepx_core::BroadcastDiv(
        "", pos_exp, deepx_core::Add("", between_sim_sum, right_neg_sim_sum));
    auto* logits = deepx_core::Add("", left_logits, right_logtis);
    auto* labels = deepx_core::OnesLike("", logits);

    // loss
    auto* loss =
        deepx_core::ReduceMean("", deepx_core::BCELoss2("", logits, labels));
    std::vector<GraphNode*> Z;
    Z = {loss, logits};
    Z.emplace_back(src_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: src embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(DeepGraphContrastive, "DeepGraphContrastive");
MODEL_ZOO_REGISTER(DeepGraphContrastive, "deep_graph_contrastive");

}  // namespace embedx
