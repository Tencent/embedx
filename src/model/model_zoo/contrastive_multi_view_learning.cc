// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Litao Hong (Lthong.brian@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class ContrastiveMultiViewLearning : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  double relu_alpha_ = 0;
  bool use_neigh_feat_ = false;

 public:
  DEFINE_MODEL_ZOO_LIKE(ContrastiveMultiViewLearning);

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
    // src node encoder
    auto* node_feat_ptr = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    GraphNode* neigh_feat_ptr = nullptr;
    const auto& self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth_);
    const auto& neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth_);
    auto* hidden =
        GraphSageEncoder("src_encoder", items_, node_feat_ptr, neigh_feat_ptr,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);

    // src shuffled node encoder
    auto* node_shuffled_feat_ptr =
        GetXInput(instance_name::X_NODE_SHUFFLED_FEATURE_NAME);
    auto* shuffled_hidden = GraphSageEncoder(
        "src_encoder", items_, node_shuffled_feat_ptr, neigh_feat_ptr,
        self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);

    auto* src_id_ptr = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* src_embed = HiddenLookup("", src_id_ptr, hidden);
    auto* shuffled_embed = HiddenLookup("", src_id_ptr, shuffled_hidden);

    // enhance node encoder
    auto* enhance_node_feat_ptr =
        GetXInput(instance_name::X_ENHANCE_NODE_FEATURE_NAME);
    const auto& enhance_self_blocks =
        GetXBlockInputs(instance_name::X_SELF_ENHANCE_BLOCK_NAME, depth_);
    const auto& enhance_neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_ENHANCE_BLOCK_NAME, depth_);
    auto* enhance_hidden = GraphSageEncoder(
        "enhance_encoder", items_, enhance_node_feat_ptr, neigh_feat_ptr,
        enhance_self_blocks, enhance_neigh_blocks, sparse_, relu_alpha_, dim_);

    // enhance shuffled node encoder
    auto* enhance_node_shuffled_feat_ptr =
        GetXInput(instance_name::X_ENHANCE_NODE_SHUFFLED_FEATURE_NAME);
    auto* enhance_shuffled_hidden = GraphSageEncoder(
        "enhance_encoder", items_, enhance_node_shuffled_feat_ptr,
        neigh_feat_ptr, enhance_self_blocks, enhance_neigh_blocks, sparse_,
        relu_alpha_, dim_);
    auto* enhance_embed = HiddenLookup("", src_id_ptr, enhance_hidden);
    auto* enhance_shuffled_embed =
        HiddenLookup("", src_id_ptr, enhance_shuffled_hidden);

    // src graph embed
    auto* src_embed_mean = deepx_core::ReduceMean("", src_embed, 0, 1);
    auto* summary = deepx_core::Sigmoid("", src_embed_mean);
    auto* readout = deepx_core::FullyConnect("readout", summary, dim_ * 2);

    // enhance graph embed
    auto* enhance_embed_mean = deepx_core::ReduceMean("", enhance_embed, 0, 1);
    auto* enhance_summary = deepx_core::Sigmoid("", enhance_embed_mean);
    auto* enhance_readout =
        deepx_core::FullyConnect("enhance_readout", enhance_summary, dim_ * 2);

    // pos
    auto* pos_dot =
        deepx_core::GEMM("pos_dot", src_embed, enhance_readout, 0, 1);
    auto* enhance_pos_dot =
        deepx_core::GEMM("enhance_pos_dot", enhance_embed, readout, 0, 1);
    auto* ones = deepx_core::OnesLike("", pos_dot);
    auto* enhance_ones = deepx_core::OnesLike("", enhance_pos_dot);

    // neg
    auto* neg_dot =
        deepx_core::GEMM("neg_dot", shuffled_embed, enhance_readout, 0, 1);
    auto* enhance_neg_dot = deepx_core::GEMM(
        "enhance_neg_dot", enhance_shuffled_embed, readout, 0, 1);
    auto* zeros = deepx_core::ZerosLike("", neg_dot);
    auto* enhance_zeros = deepx_core::ZerosLike("", enhance_neg_dot);

    // build X, Y
    auto* X = deepx_core::Concat(
        "", {pos_dot, enhance_pos_dot, neg_dot, enhance_neg_dot}, 0);
    auto* Y =
        deepx_core::Concat("", {ones, enhance_ones, zeros, enhance_zeros}, 0);

    // node embed
    auto* node_embed = deepx_core::Add("", src_embed, enhance_embed);

    // Binary Classification
    auto Z = BinaryClassificationTarget("", X, Y, has_w_);
    Z.emplace_back(node_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: src embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(ContrastiveMultiViewLearning, "CMVL");
MODEL_ZOO_REGISTER(ContrastiveMultiViewLearning,
                   "contrastive_multi_view_learning");

}  // namespace embedx
