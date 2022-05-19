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
/* input group embedding lookup functions */
/************************************************************************/
GraphNode* XNodeGroupWeightLookup(const std::string& prefix, GraphNode* X,
                                  const std::vector<GroupConfigItem3>& items,
                                  int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  int col = items.size();
  auto* W = GetVariable(
      prefix + "W", Shape(items[0].embedding_row, items.size()), tensor_type,
      TENSOR_INITIALIZER_TYPE_RAND, -1.0 / col, 1.0 / col);
  return EmbeddingLookup("", X, W);
}

GraphNode* XInputGroupEmbeddingLookup(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  std::vector<GraphNode*> W(items.size());
  std::vector<uint16_t> group_id(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_id[i] = items[i].group_id;
    auto ii = std::to_string(group_id[i]);
    int item_k = items[i].embedding_row + items[i].embedding_col;
    W[i] = GetVariable(prefix + "W" + ii,
                       Shape(items[i].embedding_row, items[i].embedding_col),
                       tensor_type, TENSOR_INITIALIZER_TYPE_RAND,
                       -1.0 / std::log(item_k), 1.0 / std::log(item_k));
  }
  return GroupEmbeddingLookup("", X, W, group_id);
}

GraphNode* XInputGroupEmbeddingLookup2(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  std::vector<GraphNode*> W(items.size());
  std::vector<uint16_t> group_id(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_id[i] = items[i].group_id;
    auto ii = std::to_string(group_id[i]);
    int item_k = items[i].embedding_col;
    W[i] = GetVariable(prefix + "W" + ii,
                       Shape(items[i].embedding_row, items[i].embedding_col),
                       tensor_type, TENSOR_INITIALIZER_TYPE_RAND, -1.0 / item_k,
                       1.0 / item_k);
  }
  return GroupEmbeddingLookup("", X, W, group_id);
}

/************************************************************************/
/* output group embedding lookup functions */
/************************************************************************/
GraphNode* XOutputGroupEmbeddingLookup(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse) {
  DXCHECK_THROW(X->shape().is_rank(2));
  DXCHECK_THROW(!items.empty());
  std::vector<GraphNode*> W(items.size());
  std::vector<uint16_t> group_id(items.size());
  int tensor_type = sparse ? TENSOR_TYPE_SRM : TENSOR_TYPE_TSR;
  for (size_t i = 0; i < items.size(); ++i) {
    group_id[i] = items[i].group_id;
    auto ii = std::to_string(group_id[i]);
    W[i] = GetVariable(prefix + "W" + ii,
                       Shape(items[i].embedding_row, items[i].embedding_col),
                       tensor_type, TENSOR_INITIALIZER_TYPE_ZEROS, 0, 0);
  }
  return GroupEmbeddingLookup("", X, W, group_id);
}

}  // namespace embedx
