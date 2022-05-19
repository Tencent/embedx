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

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class GraphDSSM : public ModelZooImpl {
 private:
  using vec_group_config = std::vector<GroupConfigItem3>;

 private:
  std::vector<int> dims_ = {64, 32};
  double relu_alpha_ = 0;
  int num_neg_ = 5;
  int depth_ = 1;

 private:
  vec_group_config user_config_;
  vec_group_config item_config_;

  uint16_t user_group_id_ = 0;

 public:
  DEFINE_MODEL_ZOO_LIKE(GraphDSSM);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
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
    } else if (k == "depth") {
      depth_ = std::stoi(v);
      if (depth_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "user_config") {
      if (!GuessGroupConfig(v, &user_config_, nullptr)) {
        return false;
      }
    } else if (k == "item_config") {
      if (!GuessGroupConfig(v, &item_config_, nullptr)) {
        return false;
      }
    } else if (k == "user_group_id") {
      user_group_id_ = (uint16_t)std::stoi(v);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Model argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

  bool PostInitConfig() override {
    if (user_config_.empty()) {
      DXERROR("user_config is empty.");
      return false;
    }

    if (item_config_.empty()) {
      DXERROR("item_config is empty.");
      return false;
    }
    return true;
  }

 public:
  static GraphNode* StackFullyConnect(const std::string& prefix, GraphNode* X,
                                      const std::vector<int> dims,
                                      double relu_alpha) {
    auto* hidden = X;
    for (size_t i = 0; i < dims.size(); ++i) {
      hidden =
          FullyConnect(prefix + "_fc" + std::to_string(i), hidden, dims[i]);
      hidden = LeakyRelu("", hidden, relu_alpha);
    }
    return hidden;
  }

  static GraphNode* GraphEncoder(const std::string& items_name,
                                 const std::vector<GroupConfigItem3>& items,
                                 int depth, bool sparse, double relu_alpha,
                                 int dim) {
    auto* Xnode_feat = GetXInput(instance_name::X_NODE_FEATURE_NAME);

    auto* next_hidden =
        XInputGroupEmbeddingLookup2(items_name, Xnode_feat, items, sparse);

    const auto& self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth);
    const auto& neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth);
    for (int i = 0; i < depth; ++i) {
      bool is_act = (i + 1) < depth ? true : false;
      next_hidden = DenseSageEncoder("DenseSageEncoder" + std::to_string(i),
                                     next_hidden, self_blocks[i],
                                     neigh_blocks[i], dim, is_act, relu_alpha);
    }
    return next_hidden;
  }

  bool InitGraph(deepx_core::Graph* graph) const override {
    // user tower
    auto* user_feat_ptr = GetXInput(instance_name::X_USER_FEATURE_NAME);
    auto* user_feat_embed = XInputGroupEmbeddingLookup(
        "user_embed", user_feat_ptr, user_config_, sparse_);
    auto* sync_user_embed =
        StackFullyConnect("user", user_feat_embed, dims_, relu_alpha_);

    auto* hidden = GraphEncoder("item_embed", item_config_, depth_, sparse_,
                                relu_alpha_, dims_.back());
    hidden = StackFullyConnect("graph", hidden, {dims_.back()}, relu_alpha_);

    auto* src_id_ptr = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* user_neigh_embed = HiddenLookup("", src_id_ptr, hidden);

    auto* user_node_ptr = GetXInput(instance_name::X_USER_NODE_NAME);
    auto* unsup_user_neigh_embed = deepx_core::GetVariable(
        "unsup_user_neigh_embed", Shape(1, dims_.back()),
        deepx_core::TENSOR_TYPE_SRM, deepx_core::TENSOR_INITIALIZER_TYPE_ZEROS,
        0, 0);
    auto* sync_user_neigh_embed =
        Assemble("", user_node_ptr, user_neigh_embed, unsup_user_neigh_embed);

    auto* sync_user_embed_graph = deepx_core::Add(
        "sync_user_embed_graph", sync_user_embed, sync_user_neigh_embed);

    // user infer: supervised prob with user neigh embedding from param
    std::vector<uint16_t> user_group_id = {user_group_id_};
    auto* async_user_neigh_embed = deepx_core::GroupEmbeddingLookup2(
        "", user_feat_ptr, unsup_user_neigh_embed, user_group_id);

    auto* async_user_embed_graph = deepx_core::Add(
        "async_user_embed_graph", sync_user_embed, async_user_neigh_embed);

    // item tower
    auto* item_feat_ptr = GetXInput(instance_name::X_ITEM_FEATURE_NAME);
    auto* item_embed = XInputGroupEmbeddingLookup2("item_embed", item_feat_ptr,
                                                   item_config_, sparse_);

    item_embed = StackFullyConnect("item", item_embed, dims_, relu_alpha_);

    auto* user_id_ptr = GetXInput(instance_name::X_USER_ID_NAME);
    auto* item_id_ptr = GetXInput(instance_name::X_ITEM_ID_NAME);
    auto* dot = BatchLookupDot("batch_dot", user_id_ptr, item_id_ptr,
                               sync_user_embed_graph, item_embed);
    // [pos1, neg1, ..., neg_(num_neg),
    //  pos2, neg1, ..., neg_(num_neg),
    //  ...]
    dot = Reshape("", dot, Shape(-1, num_neg_ + 1));
    // sampled softmax loss
    auto Z = MultiClassificationTarget("loss", dot, has_w_);
    Z.emplace_back(async_user_embed_graph);
    Z.emplace_back(item_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: dump classification prob
    // Z[2]: dump user embedding
    // Z[3]: dump item embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(GraphDSSM, "GraphDSSM");
MODEL_ZOO_REGISTER(GraphDSSM, "graph_dssm");

}  // namespace embedx
