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

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class SemisupGraphsage : public ModelZooImpl {
 private:
  int depth_ = 1;
  int dim_ = 128;
  double relu_alpha_ = 0;

  int num_label_ = 1;
  int max_label_ = 1;
  bool multi_label_ = false;

  bool use_neigh_feat_ = false;
  double unsup_weight_ = 0;

 public:
  DEFINE_MODEL_ZOO_LIKE(SemisupGraphsage);

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
        DXERROR("invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "num_label") {
      num_label_ = std::stoi(v);
      if (num_label_ < 1) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "max_label") {
      max_label_ = std::stoi(v);
      if (max_label_ < 1) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
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
    } else if (k == "weight" || k == "unsup_weight") {
      unsup_weight_ = std::stod(v);
      if (unsup_weight_ < 0) {
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
    auto* hidden = GraphSageEncoder("", items_, depth_, use_neigh_feat_,
                                    sparse_, relu_alpha_, dim_);

    // sup part
    std::vector<GraphNode*> Zsup;
    {
      auto* Xnode_id = GetXInput(instance_name::X_NODE_ID_NAME);
      auto* node_embed = HiddenLookup("", Xnode_id, hidden);
      int output_dim = multi_label_ ? num_label_ : max_label_ + 1;
      auto* output = deepx_core::FullyConnect("", node_embed, output_dim);
      if (multi_label_) {
        Zsup = MultiLabelClassificationTarget("sup", output, has_w_);
      } else {
        Zsup = MultiClassificationTarget("sup", output, has_w_);
      }
      Zsup.emplace_back(node_embed);
    }

    // unsup part
    std::vector<GraphNode*> Zunsup;
    {
      auto* Xsrc_id = GetXInput(instance_name::X_SRC_ID_NAME);
      auto* Xdst_id = GetXInput(instance_name::X_DST_ID_NAME);
      auto* src_embed = HiddenLookup("", Xsrc_id, hidden);
      auto* dst_embed = HiddenLookup("", Xdst_id, hidden);
      auto* dot = deepx_core::BatchDot("", src_embed, dst_embed);
      auto* Yunsup = GetYUnsup(instance_name::Y_UNSUPVISED_NAME, 1);
      Zunsup = BinaryClassificationTarget("unsup", dot, Yunsup, has_w_);
    }

    // sup part + unsup part
    std::vector<GraphNode*> Z;
    {
      auto* unsup_weight = deepx_core::ConstantScalar("", unsup_weight_);
      auto* unsup_loss = deepx_core::Mul("", unsup_weight, Zunsup[0]);
      auto* loss = deepx_core::Add("", Zsup[0], unsup_loss);
      Z.emplace_back(loss);
    }
    Z.emplace_back(Zsup[1]);
    Z.emplace_back(Zsup[2]);
    deepx_core::ReleaseVariable();
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(SemisupGraphsage, "SemisupGraphsage");
MODEL_ZOO_REGISTER(SemisupGraphsage, "semisup_graphsage");

}  // namespace embedx
