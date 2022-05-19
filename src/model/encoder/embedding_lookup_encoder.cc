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

#include <cmath>  // std::log

#include "src/model/encoder/gnn_encoder.h"

namespace embedx {

/************************************************************************/
/* input embedding lookup functions */
/************************************************************************/
// rand initialization (-1 / log(col), 1 / log(col))
GraphNode* XInputEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                 const GroupConfigItem3& item, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  int item_k = item.embedding_row + item.embedding_col;
  auto* W =
      GetVariable(prefix + "W", Shape(item.embedding_row, item.embedding_col),
                  tensor_type, TENSOR_INITIALIZER_TYPE_RAND,
                  -1.0 / std::log(item_k), 1.0 / std::log(item_k));
  return EmbeddingLookup("", X, W);
}

// randn initialization (0, 1e-3) for graph-ctr models
GraphNode* XInputEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  auto* W =
      GetVariable(prefix + "W", Shape(item.embedding_row, item.embedding_col),
                  tensor_type, TENSOR_INITIALIZER_TYPE_RANDN, 0, 1e-3);
  return EmbeddingLookup("", X, W);
}

// rand initiialization (-1 / col, 1 / col)
GraphNode* XInputEmbeddingLookup3(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  auto* W =
      GetVariable(prefix + "W", Shape(item.embedding_row, item.embedding_col),
                  tensor_type, TENSOR_INITIALIZER_TYPE_RAND,
                  -1.0 / item.embedding_col, 1.0 / item.embedding_col);
  return EmbeddingLookup("", X, W);
}

/************************************************************************/
/* output embedding lookup functions */
/************************************************************************/
GraphNode* XOutputEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  auto* W =
      GetVariable(prefix + "W", Shape(item.embedding_row, item.embedding_col),
                  tensor_type, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
  return EmbeddingLookup("", X, W);
}

}  // namespace embedx
