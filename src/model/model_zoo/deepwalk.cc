// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/encoder/gnn_encoder.h"
#include "src/model/instance_node_name.h"
#include "src/model/model_zoo_impl.h"
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

class DeepWalk : public ModelZooImpl {
 public:
  DEFINE_MODEL_ZOO_LIKE(DeepWalk);

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
    if ((int)items_.size() != 1) {
      DXERROR("Deepwalk should only have one feature group config.");
      return false;
    }

    return true;
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    auto* src_node_ptr = GetXInput(instance_name::X_SRC_NODE_NAME);
    auto* dst_node_ptr = GetXInput(instance_name::X_DST_NODE_NAME);

    // only use in prediction
    auto* src_embed =
        XInputEmbeddingLookup3("deepwalk", src_node_ptr, items_[0], sparse_);

    // BatchLookupAndDot fuses two lookup op and one batch dot op.
    // It is much faster than code below.
    // src_embed = EmbeddingLookup("src", Xsrc_node)
    // dst_embed = EmbeddingLookup("dst", Xdst_node)
    // dot = BatchDot(src_embed, dst_embed)
    auto* dot = BatchLookupAndDot("deepwalk", src_node_ptr, dst_node_ptr,
                                  items_[0], sparse_);

    auto Z = deepx_core::BinaryClassificationTarget(dot, has_w_);
    Z.emplace_back(src_embed);
    deepx_core::ReleaseVariable();
    // Z[0]: loss
    // Z[1]: prob
    // Z[2]: src embed
    return graph->Compile(Z, 1);
  }
};

MODEL_ZOO_REGISTER(DeepWalk, "DeepWalk");
MODEL_ZOO_REGISTER(DeepWalk, "deepwalk");

}  // namespace embedx
