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
#include "src/model/op/gnn_graph_node.h"

namespace embedx {

GraphNode* BatchLookupAndDot(const std::string& prefix, GraphNode* Xin,
                             GraphNode* Xout, const GroupConfigItem3& item,
                             int sparse) {
  DXCHECK_THROW(Xin->shape().is_rank(2));
  DXCHECK_THROW(Xout->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;

  auto* Win =
      GetVariable(prefix + "W", Shape(item.embedding_row, item.embedding_col),
                  tensor_type, TENSOR_INITIALIZER_TYPE_RAND,
                  -1.0 / item.embedding_col, 1.0 / item.embedding_col);
  auto* Wout = GetVariable(prefix + "Wout",
                           Shape(item.embedding_row, item.embedding_col),
                           tensor_type, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);

  auto* dot = BatchLookupDot("", Xin, Xout, Win, Wout);
  return dot;
}

}  // namespace embedx
