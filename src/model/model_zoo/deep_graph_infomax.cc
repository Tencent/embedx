// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Litao Hong (Lthong.brian@gmail.com)
//         Yong Zhou (zhouyongnju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class DeepGraphInfoMax : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  double relu_alpha_ = 0;

  bool use_neigh_feat_ = false;

 public:
  DEFINE_MODEL_ZOO_LIKE(UnsupervisedGraphsage);

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
    auto* node_feat_ptr = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    GraphNode* neigh_feat_ptr = nullptr;
    const auto& self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth_);
    const auto& neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth_);
    auto* hidden =
        GraphSageEncoder("", items_, node_feat_ptr, neigh_feat_ptr, self_blocks,
                         neigh_blocks, sparse_, relu_alpha_, dim_);

    auto* node_shuffled_feat_ptr =
        GetXInput(instance_name::X_NODE_SHUFFLED_FEATURE_NAME);
    auto* shuffled_hidden =
        GraphSageEncoder("", items_, node_shuffled_feat_ptr, neigh_feat_ptr,
                         self_blocks, neigh_blocks, sparse_, relu_alpha_, dim_);

    auto* src_id_ptr = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* src_embed = HiddenLookup("", src_id_ptr, hidden);
    auto* shuffled_embed = HiddenLookup("", src_id_ptr, shuffled_hidden);

    auto* src_embed_mean = deepx_core::ReduceMean("", src_embed, 0, 1);
    auto* summary = deepx_core::Sigmoid("", src_embed_mean);
    auto* readout = deepx_core::FullyConnect("readout", summary, dim_ * 2);

    auto* pos_dot = deepx_core::GEMM("pos_dot", src_embed, readout, 0, 1);
    auto* ones = deepx_core::OnesLike("", pos_dot);
    auto* neg_dot = deepx_core::GEMM("neg_dot", shuffled_embed, readout, 0, 1);
    auto* zeros = deepx_core::ZerosLike("", neg_dot);

    auto* X = deepx_core::Concat("", {pos_dot, neg_dot}, 0);
    auto* Y = deepx_core::Concat("", {ones, zeros}, 0);
    auto Z = BinaryClassificationTarget("", X, Y, has_w_);
    Z.emplace_back(src_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: src embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(DeepGraphInfoMax, "DGI");
MODEL_ZOO_REGISTER(DeepGraphInfoMax, "deep_graph_infomax");

}  // namespace embedx
