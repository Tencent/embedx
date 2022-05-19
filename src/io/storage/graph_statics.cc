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

#include "src/io/storage/graph_statics.h"

#include <deepx_core/dx_log.h>

namespace embedx {

GraphStatics::GraphStatics(Indexing* src_indexing, Indexing* dst_indexing)
    : src_indexing_(src_indexing), dst_indexing_(dst_indexing) {}

bool GraphStatics::Add(int_t src_node, int_t dst_node) {
  int src_index = src_indexing_->Get(src_node);
  int src_list_size = (int)src_node_list_.size();
  if (src_index < src_list_size) {
    ++out_degree_list_[src_index];
  } else if (src_index == src_list_size) {
    src_node_list_.emplace_back(src_node);
    out_degree_list_.emplace_back(1);
  } else {
    DXERROR(
        "Need src_index <= src_node_list.size(), got src_index: %d vs "
        "src_node_list.size(): %d",
        src_index, src_list_size);
    return false;
  }

  int dst_index = dst_indexing_->Get(dst_node);
  int dst_list_size = (int)dst_node_list_.size();
  if (dst_index < dst_list_size) {
    ++in_degree_list_[dst_index];
  } else if (dst_index == dst_list_size) {
    dst_node_list_.emplace_back(dst_node);
    in_degree_list_.emplace_back(1);
  } else {
    DXERROR(
        "Need dst_index <= dst_node_list.size(), got dst_index :%d vs "
        "dst_node_list.size(): %d",
        dst_index, dst_list_size);
    return false;
  }

  return true;
}

int GraphStatics::GetInDegree(int_t dst_node) const {
  int dst_index = dst_indexing_->Get(dst_node);
  if (dst_index >= 0 && dst_index < (int)in_degree_list_.size()) {
    return in_degree_list_[dst_index];
  } else {
    return 0;
  }
}

int GraphStatics::GetOutDegree(int_t src_node) const {
  int src_index = src_indexing_->Get(src_node);
  if (src_index >= 0 && src_index < (int)out_degree_list_.size()) {
    return out_degree_list_[src_index];
  } else {
    return 0;
  }
}

std::unique_ptr<GraphStatics> NewGraphStatics(Indexing* src_indexing,
                                              Indexing* dst_indexing) {
  std::unique_ptr<GraphStatics> graph_statics;
  graph_statics.reset(new GraphStatics(src_indexing, dst_indexing));
  return graph_statics;
}

}  // namespace embedx
