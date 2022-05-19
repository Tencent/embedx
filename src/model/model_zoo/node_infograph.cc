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

class NodeInfoGraph : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  double relu_alpha_ = 0;
  bool use_neigh_feat_ = false;
  int num_label_ = 1;
  bool multi_label_ = false;
  int max_label_ = 1;

 public:
  DEFINE_MODEL_ZOO_LIKE(NodeInfoGraph);

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
    } else if (k == "num_label") {
      num_label_ = std::stoi(v);
      if (num_label_ < 1) {
        DXERROR("invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "max_label") {
      max_label_ = std::stoi(v);
      if (max_label_ < 1) {
        DXERROR("invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "multi_label") {
      auto val = std::stoi(v);
      if (val == 1) {
        multi_label_ = true;
      } else if (val == 0) {
        multi_label_ = false;
      } else {
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
    // get node feature and block
    auto* node_feat_ptr = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    GraphNode* neigh_feat_ptr = nullptr;
    auto* node_shuffled_feat_ptr =
        GetXInput(instance_name::X_NODE_SHUFFLED_FEATURE_NAME);
    const auto& self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth_);
    const auto& neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth_);

    // sup part
    auto* sup_hidden =
        GraphSageEncoder("sup_encoder", items_, node_feat_ptr, neigh_feat_ptr,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);

    auto* node_id_ptr = GetXInput(instance_name::X_NODE_ID_NAME);
    auto* node_embed = HiddenLookup("node_embed", node_id_ptr, sup_hidden);

    int output_dim = multi_label_ ? num_label_ : max_label_ + 1;
    auto* output = deepx_core::FullyConnect("output", node_embed, output_dim);

    std::vector<GraphNode*> Zsup;
    if (multi_label_) {
      Zsup = MultiLabelClassificationTarget(output, has_w_);
    } else {
      Zsup = MultiClassificationTarget(output, has_w_);
    }
    Zsup.emplace_back(node_embed);

    // unsup part
    auto* src_id_ptr = GetXInput(instance_name::X_SRC_ID_NAME);
    // unsu= part use unsup_encoder
    auto* unsup_hidden =
        GraphSageEncoder("unsup_encoder", items_, node_feat_ptr, neigh_feat_ptr,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);
    auto* unsup_shuffled_hidden = GraphSageEncoder(
        "unsup_encoder", items_, node_shuffled_feat_ptr, neigh_feat_ptr,
        self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);

    auto* src_embed = HiddenLookup("src_embed", src_id_ptr, unsup_hidden);
    auto* shuffled_embed =
        HiddenLookup("shuffled_embed", src_id_ptr, unsup_shuffled_hidden);

    auto* src_embed_mean =
        deepx_core::ReduceMean("src_embed_mean", src_embed, 0, 1);
    auto* summary = deepx_core::Sigmoid("summary", src_embed_mean);
    auto* readout = deepx_core::FullyConnect("readout", summary, dim_ * 2);

    auto* pos_dot = deepx_core::GEMM("pos_dot", src_embed, readout, 0, 1);
    auto* ones = deepx_core::OnesLike("ones", pos_dot);
    auto* neg_dot = deepx_core::GEMM("neg_dot", shuffled_embed, readout, 0, 1);
    auto* zeros = deepx_core::ZerosLike("zeros", neg_dot);

    // unsup use sup part encoder
    auto* mix_hidden =
        GraphSageEncoder("sup_encoder", items_, node_feat_ptr, neigh_feat_ptr,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);
    auto* mix_shuffle_hidden = GraphSageEncoder(
        "sup_encoder", items_, node_shuffled_feat_ptr, neigh_feat_ptr,
        self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);
    auto* mix_src_embed = HiddenLookup("mix_src_embed", src_id_ptr, mix_hidden);
    auto* mix_src_shuffled_embed =
        HiddenLookup("con_src_shuffled_embed", src_id_ptr, mix_shuffle_hidden);

    auto* mix_pos_dot =
        deepx_core::GEMM("mix_pos_dot", mix_src_embed, readout, 0, 1);
    auto* mix_ones = deepx_core::OnesLike("mix_ones", mix_pos_dot);

    auto* mix_neg_dot =
        deepx_core::GEMM("mix_neg_dot", mix_src_shuffled_embed, readout, 0, 1);
    auto* mix_zeros = deepx_core::ZerosLike("mix_zeros", mix_neg_dot);

    // build unsub X, Y
    auto* Xunsup = deepx_core::Concat(
        "Xunsup", {pos_dot, mix_pos_dot, neg_dot, mix_neg_dot}, 0);
    auto* Yunsup =
        deepx_core::Concat("Yunsup", {ones, mix_ones, zeros, mix_zeros}, 0);
    auto Zunsup = BinaryClassificationTarget("Zunsub", Xunsup, Yunsup, has_w_);

    // loss, sup part + unsup part
    std::vector<GraphNode*> Z;
    auto* loss = deepx_core::Add("", Zsup[0], Zunsup[0]);
    Z.emplace_back(loss);

    auto* sum_embed = deepx_core::Add("", src_embed, mix_src_embed);
    Z.emplace_back(Zunsup[1]);
    Z.emplace_back(sum_embed);
    deepx_core::ReleaseVariable();
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(NodeInfoGraph, "NIG");
MODEL_ZOO_REGISTER(NodeInfoGraph, "node_infograph");

}  // namespace embedx
