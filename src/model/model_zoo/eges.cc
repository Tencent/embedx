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

class Eges : public ModelZooImpl {
 public:
  DEFINE_MODEL_ZOO_LIKE(Eges);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
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
    // eges encoder
    auto* unique_node_ptr = GetXInput(instance_name::X_UNIQUE_NODE_NAME);
    auto* src_feat_ptr = GetXInput(instance_name::X_NODE_FEATURE_NAME);
    auto* src_embed =
        EgesEncoder("eges", src_feat_ptr, unique_node_ptr, items_, sparse_);

    auto* src_id_ptr = GetXInput(instance_name::X_SRC_ID_NAME);
    auto* hidden = HiddenLookup("", src_id_ptr, src_embed);

    // embedding_lookup encoder
    auto* dst_node_ptr = GetXInput(instance_name::X_DST_NODE_NAME);
    auto* dst_embed =
        XOutputEmbeddingLookup("out", dst_node_ptr, items_[0], sparse_);

    auto* dot = deepx_core::BatchDot("", hidden, dst_embed);
    auto Z = BinaryClassificationTarget(dot, has_w_);
    Z.emplace_back(hidden);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: input embedding
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(Eges, "Eges");
MODEL_ZOO_REGISTER(Eges, "eges");

}  // namespace embedx
