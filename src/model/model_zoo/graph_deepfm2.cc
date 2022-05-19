// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/dx_log.h>
#include <deepx_core/graph/instance_reader.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {
namespace {

// TODO(tinkle1129): encoder name used in model and instance reader must be the
// consistent!!!
const std::string USER_ENCODER_NAME = "USER_ENCODER_NAME";

}  // namespace

class GraphDeepFM2Model : public ModelZooImpl {
 private:
  // gnn
  int depth_ = 2;
  int sage_dim_ = 128;
  double relu_alpha_ = 0;
  uint16_t user_group_id_ = 0;

  // 0 for pinsage encoder, 1 for graphsage encoder
  int sage_encoder_type_ = 0;

  // deepfm2
  std::vector<int> dfm_dims_{64, 32};

  // loss
  double unsup_weight_ = 0;

 public:
  DEFINE_MODEL_ZOO_LIKE(GraphDeepFM2Model);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
      DXINFO("Default model argument %s = %s", k.c_str(), v.c_str());
      return true;
    } else if (k == "depth") {
      depth_ = std::stoi(v);
      if (depth_ < 1) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "alpha" || k == "relu_alpha") {
      relu_alpha_ = std::stod(v);
      if (relu_alpha_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "user_group_id") {
      user_group_id_ = (uint16_t)std::stoi(v);
    } else if (k == "sage_dim") {
      sage_dim_ = std::stoi(v);
      if (sage_dim_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "dfm_dims") {
      if (!deepx_core::ParseDeepDims(v, &dfm_dims_, k.c_str())) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "weight") {
      unsup_weight_ = std::stod(v);
      if (unsup_weight_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "sage_encoder_type") {
      sage_encoder_type_ = std::stod(v);
      if (sage_encoder_type_ != 0 && sage_encoder_type_ != 1) {
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
    if ((int)items_.size() < 2) {
      DXERROR(
          "Feature group config items should be greater than or equal to 2, at "
          "least user and item.");
      return false;
    }

    if (sage_encoder_type_ == 0) {
      if (sage_dim_ != items_[0].embedding_col) {
        DXERROR("Pinsage model, sage_dim != dst_embed_dim, %d != %d.",
                sage_dim_, items_[0].embedding_col);
        return false;
      }
    } else if (sage_encoder_type_ == 1) {
      if (2 * sage_dim_ != items_[0].embedding_col) {
        DXERROR("Graphsage model, 2 * sage_dim != dst_embed_dim, %d != %d.",
                2 * sage_dim_, items_[0].embedding_col);
        return false;
      }
    } else {
      DXERROR("Currently only support sage_encoder_type equals 0 and 1.");
      return false;
    }
    return true;
  }

 public:
  GraphNode* DeepFM2(GraphNode* X, GraphNode* user_embed) const {
    auto* lin = deepx_core::WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* quad1 =
        deepx_core::DeepGroupEmbeddingLookup2("feature", X, items_, sparse_);
    auto* quad2 = deepx_core::Reshape("", quad1, Shape(-1, item_m_, item_k_));
    auto* quad3 = deepx_core::BatchGroupFMQuadratic2("", quad2);
    auto* quad4 = deepx_core::Concat("", {quad1, user_embed});
    auto* deep =
        deepx_core::StackedFullyConnect("deep", quad4, dfm_dims_, "relu");
    auto* concat = deepx_core::Concat("", {lin, quad3, deep});
    auto* logit = deepx_core::FullyConnect("", concat, 1);
    return logit;
  }

  bool InitGraph(deepx_core::Graph* graph) const override {
    // Unsupervised Part
    // input: edges provided by random walk
    auto* Xuser_node_feat = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    auto* user_feat_embed =
        XInputEmbeddingLookup2("feature", Xuser_node_feat, items_[0], sparse_);

    auto user_self_blocks =
        GetXBlockInputs(instance_name::X_SELF_BLOCK_NAME, depth_);
    auto user_neigh_blocks =
        GetXBlockInputs(instance_name::X_NEIGH_BLOCK_NAME, depth_);
    auto* user_hidden = SageEncoder("user", user_feat_embed, user_self_blocks,
                                    user_neigh_blocks, sage_encoder_type_,
                                    depth_, sage_dim_, relu_alpha_);

    auto* Xsrc_id = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* Xdst_node = GetXInput(instance_name::X_DST_NODE_NAME);
    auto* src_embed = HiddenLookup("", Xsrc_id, user_hidden);
    auto* dst_embed =
        XInputEmbeddingLookup("feature", Xdst_node, items_[0], sparse_);

    auto* dot = deepx_core::BatchDot("", src_embed, dst_embed);
    auto* Y_unsup = GetYUnsup(instance_name::Y_UNSUPVISED_NAME, 1);
    auto Z1 = BinaryClassificationTarget(dot, Y_unsup, has_w_);

    // CTR Part
    // DeepFM2 model + concat pinsage generated user emb at deep part

    // ctr related user emb from pinsage encoder
    auto* Xuser_id =
        GetXInput(instance_name::X_NODE_ID_NAME + USER_ENCODER_NAME);
    auto* user_embed = HiddenLookup("", Xuser_id, user_hidden);

    // user embedding generated from pinsage encoder is saved in
    // "unsup_user_embed" according to Assemble Node.
    auto* Xuser_node = GetXInput(instance_name::X_USER_NODE_NAME);
    auto* unsup_user_embed = deepx_core::GetVariable(
        "unsup_user_embed", Shape(1, items_[0].embedding_col),
        deepx_core::TENSOR_TYPE_SRM, deepx_core::TENSOR_INITIALIZER_TYPE_ZEROS,
        0, 0);
    auto* sync_user_embed =
        Assemble("", Xuser_node, user_embed, unsup_user_embed);

    // deepfm2
    // TODO(tinkle1129): move the instance node names defined in deepx to
    // embedx
    auto* X = GetXInput(deepx_core::X_NAME);
    auto* sync_logit = DeepFM2(X, sync_user_embed);

    auto* Y = deepx_core::GetY(1);
    auto Z2 = BinaryClassificationTarget(sync_logit, Y, has_w_);
    // loss = weight * unsup_loss + ctr_loss
    auto* unsup_weight = deepx_core::ConstantScalar("", unsup_weight_);
    auto* unsup_loss = deepx_core::Mul("", unsup_weight, Z1[0]);
    auto* loss = deepx_core::Add("", unsup_loss, Z2[0]);

    // infer: ctr related user emb from unsup_user_embed
    std::vector<uint16_t> user_group_id = {user_group_id_};
    auto* async_user_embed = deepx_core::GroupEmbeddingLookup2(
        "async_user_embed", X, unsup_user_embed, user_group_id);
    auto* async_logits = DeepFM2(X, async_user_embed);
    auto* async_prob = deepx_core::Sigmoid("", async_logits);

    std::vector<GraphNode*> Z;
    Z.emplace_back(loss);
    Z.emplace_back(async_prob);
    Z.emplace_back(user_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: user_embed
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(GraphDeepFM2Model, "GraphDeepFM2Model");
MODEL_ZOO_REGISTER(GraphDeepFM2Model, "graph_deepfm2");

}  // namespace embedx
