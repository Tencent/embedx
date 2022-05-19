// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class SupGraphsage : public ModelZooImpl {
 private:
  int depth_ = 1;
  double relu_alpha_ = 0;
  int dim_ = 128;
  // multi label
  int num_label_ = 1;
  bool multi_label_ = false;
  // multi classification
  int max_label_ = 1;

  bool use_neigh_feat_ = false;

 public:
  DEFINE_MODEL_ZOO_LIKE(SupGraphsage);

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
    auto* hidden = GraphSageEncoder("", items_, depth_, use_neigh_feat_,
                                    sparse_, relu_alpha_, dim_);

    auto* Xnode_id = GetXInput(instance_name::X_NODE_ID_NAME);
    auto* node_embed = HiddenLookup("", Xnode_id, hidden);

    int output_dim = multi_label_ ? num_label_ : max_label_ + 1;
    auto* output = deepx_core::FullyConnect("output", node_embed, output_dim);

    std::vector<GraphNode*> Z;
    if (multi_label_) {
      Z = MultiLabelClassificationTarget(output, has_w_);
    } else {
      Z = MultiClassificationTarget(output, has_w_);
    }

    Z.emplace_back(node_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: node embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(SupGraphsage, "SupGraphSage");
MODEL_ZOO_REGISTER(SupGraphsage, "sup_graphsage");
}  // namespace embedx
