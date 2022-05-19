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

#include "src/model/data_flow/deep_flow.h"

#include <deepx_core/dx_log.h>

namespace embedx {

void DeepFlow::FillNodeOrIndex(Instance* inst, const std::string& name,
                               const vec_int_t& nodes,
                               const Indexing* indexing) const {
  auto* ptr = &inst->get_or_insert<csr_t>(name);
  ptr->clear();

  for (auto node : nodes) {
    if (indexing != nullptr) {
      int index = indexing->Get(node);
      DXCHECK(index >= 0);
      ptr->emplace(index, 1);
      ptr->add_row();
    } else {
      ptr->emplace(node, 1);
      ptr->add_row();
    }
  }
}

void DeepFlow::FillNodeFeature(
    Instance* inst, const std::string& name, const vec_int_t* nodes_ptr,
    const std::vector<vec_pair_t>& node_feats_list) const {
  auto* node_feat_ptr = &inst->get_or_insert<csr_t>(name);
  node_feat_ptr->clear();

  for (size_t i = 0; i < node_feats_list.size(); ++i) {
    for (const auto& entry : node_feats_list[i]) {
      node_feat_ptr->emplace(entry.first, entry.second);
    }

    if (nodes_ptr) {
      node_feat_ptr->emplace((*nodes_ptr)[i], 1.0);
    }

    node_feat_ptr->add_row();
  }
}

}  // namespace embedx
