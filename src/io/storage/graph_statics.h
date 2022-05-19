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

#pragma once
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/io/indexing.h"

namespace embedx {

class GraphStatics {
 private:
  Indexing* src_indexing_;
  Indexing* dst_indexing_;
  vec_int_t src_node_list_;
  vec_int_t dst_node_list_;
  std::vector<int> in_degree_list_;
  std::vector<int> out_degree_list_;

 public:
  GraphStatics(Indexing* src_indexing, Indexing* dst_indexing);
  ~GraphStatics() = default;

 public:
  bool Add(int_t src_node, int_t dst_node);
  int GetInDegree(int_t dst_node) const;
  int GetOutDegree(int_t src_node) const;

 public:
  const Indexing* src_indexing() const noexcept { return src_indexing_; }
  const Indexing* dst_indexing() const noexcept { return dst_indexing_; }
  const vec_int_t* src_node_list() const noexcept { return &src_node_list_; }
  const vec_int_t* dst_node_list() const noexcept { return &dst_node_list_; }

  const std::vector<int>* in_degree_list() const noexcept {
    return &in_degree_list_;
  }
  const std::vector<int>* out_degree_list() const noexcept {
    return &out_degree_list_;
  }
};

std::unique_ptr<GraphStatics> NewGraphStatics(Indexing* src_indexing,
                                              Indexing* dst_indexing);

}  // namespace embedx
