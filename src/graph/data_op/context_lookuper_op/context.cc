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

#include "src/graph/data_op/context_lookuper_op/context.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

namespace embedx {
namespace graph_op {

bool Context::Lookup(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* contexts) const {
  contexts->clear();
  contexts->resize(nodes.size());

  size_t empty_count = 0;

  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto* cur_context = graph_.FindContext(nodes[i]);

    if (cur_context == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", nodes[i]);

      empty_count += 1;
      continue;
    }

    (*contexts)[i] = *cur_context;
  }

  return nodes.size() > empty_count;
}

std::unique_ptr<Context> NewContext(const InMemoryGraph* graph) {
  std::unique_ptr<Context> context;
  context.reset(new Context(graph));
  return context;
}

}  // namespace graph_op
}  // namespace embedx
