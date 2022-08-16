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
// Bipartite PLNLP is an extended model of PLNLP on bipartite graph.
// Please refer to the paper "Pairwise Learning for Neural Link Prediction"
// (https://arxiv.org/abs/2112.02936) for more details of PLNLP.

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {
namespace {

const std::string USER_ENCODER_NAME = "USER_ENCODER_NAME";
const std::string ITEM_ENCODER_NAME = "ITEM_ENCODER_NAME";

}  // namespace

class BipartitePLNLP : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  int num_neg_ = 5;
  std::vector<int> fc_dims_;
  double relu_alpha_ = 0;
  bool use_neigh_feat_ = false;
  std::string decoder_name_ = "DOT";

 public:
  DEFINE_MODEL_ZOO_LIKE(BipartitePLNLP);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
      DXINFO("Default model argument %s = %s.", k.c_str(), v.c_str());
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
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      if (num_neg_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "decoder_name") {
      decoder_name_ = v;
      if (v != "DOT" && v != "FC") {
        DXERROR("Invalid %s: %s. Should be DOT or FC.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "fc_dims") {
      if (!deepx_core::ParseDeepDims(v, &fc_dims_, k.c_str())) {
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
    auto* user_hidden =
        GraphSageEncoder(USER_ENCODER_NAME, items_, depth_, use_neigh_feat_,
                         sparse_, relu_alpha_, dim_);
    auto* item_hidden =
        GraphSageEncoder(ITEM_ENCODER_NAME, items_, depth_, use_neigh_feat_,
                         sparse_, relu_alpha_, dim_);
    auto* hidden = deepx_core::Concat("", {user_hidden, item_hidden}, 0);

    auto* Xsrc_id = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* Xdst_id = GetXInput(instance_name::X_DST_ID_NAME);
    auto* src_embed = HiddenLookup("", Xsrc_id, hidden);
    auto* dst_embed = HiddenLookup("", Xdst_id, hidden);

    GraphNode* score;
    if (decoder_name_ == "DOT") {
      score = BatchDot("", src_embed, dst_embed);
    } else {
      auto* ele = deepx_core::Concat("", {src_embed, dst_embed});
      for (size_t i = 0; i < fc_dims_.size(); ++i) {
        ele = FullyConnect("fc" + std::to_string(i), ele, fc_dims_[i]);
        ele = LeakyRelu("fc_act" + std::to_string(i), ele, relu_alpha_);
      }
      score = FullyConnect("fc_to_1", ele, 1);
    }

    // Format of reshaped_score:
    // [
    // [pos_1, neg_11, ..., neg_1(num_neg)],
    // [pos_2, neg_21, ..., neg_2(num_neg)],
    // [...],
    // [pos_(batch_size), neg_(batch_size)1, ..., neg_(batch_size)(num_neg)]
    // ]
    auto* reshaped_score = Reshape("", score, Shape(-1, num_neg_ + 1));

    // Get scores of positive samples:
    auto* pos_score = SubscriptRange("", reshaped_score, 1, 0, 1);

    // Get scores of negative samples:
    auto* neg_score =
        Reshape("", SubscriptRange("", reshaped_score, 1, 1, num_neg_ + 1),
                Shape(1, -1));

    auto* pos_sub_neg_score = BroadcastSub("", pos_score, neg_score);
    auto* ones = Ones("", Shape(1, 1));
    auto* loss =
        ReduceMean("", Square("", BroadcastSub("", ones, pos_sub_neg_score)));

    std::vector<GraphNode*> Z;
    Z.emplace_back(loss);
    Z.emplace_back(score);
    Z.emplace_back(src_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: src embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(BipartitePLNLP, "BipartitePLNLP");
MODEL_ZOO_REGISTER(BipartitePLNLP, "bipartite_plnlp");

}  // namespace embedx
