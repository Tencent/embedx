// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <deepx_core/common/misc.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/tensor/shape.h>

#include <string>
#include <vector>

#include "src/model/model_zoo_impl.h"

namespace embedx {

class DeepFMModel : public ModelZooImpl {
 private:
  std::vector<int> deep_dims_{64, 32, 1};

 public:
  DEFINE_MODEL_ZOO_LIKE(DeepFMModel);

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (ModelZooImpl::InitConfigKV(k, v)) {
    } else if (k == "deep_dims") {
      if (!deepx_core::ParseDeepDimsAppendOne(v, &deep_dims_, k.c_str())) {
        return false;
      }
    } else {
      DXERROR("Unexpected config: %s=%s.", k.c_str(), v.c_str());
      return false;
    }
    return true;
  }

  bool PostInitConfig() override {
    if (items_.empty()) {
      DXERROR("Please specify group_config.");
      return false;
    }
    DXCHECK_THROW(item_is_fm_);
    return true;
  }

 public:
  bool InitGraph(deepx_core::Graph* graph) const override {
    auto* X = deepx_core::GetX();
    auto* lin1 =
        deepx_core::WideGroupEmbeddingLookup("lin", X, items_, sparse_);
    auto* lin2 = deepx_core::ReduceSum("", lin1, 1, 1);
    auto* quad1 =
        deepx_core::DeepGroupEmbeddingLookup("quad", X, items_, sparse_);
    auto* quad2 =
        deepx_core::Reshape("", quad1, deepx_core::Shape(-1, item_m_, item_k_));
    auto* quad3 = deepx_core::BatchGroupFMQuadratic("", quad2);
    auto* deep =
        deepx_core::StackedFullyConnect("deep", quad1, deep_dims_, "relu");
    auto* Z1 = deepx_core::AddN("", {lin2, quad3, deep});
    auto Z2 = deepx_core::BinaryClassificationTarget(Z1, has_w_);
    deepx_core::ReleaseVariable();
    return graph->Compile(Z2, 1);
  }
};

MODEL_ZOO_REGISTER(DeepFMModel, "DeepFMModel");
MODEL_ZOO_REGISTER(DeepFMModel, "deep_fm");
MODEL_ZOO_REGISTER(DeepFMModel, "deepfm");

}  // namespace embedx
