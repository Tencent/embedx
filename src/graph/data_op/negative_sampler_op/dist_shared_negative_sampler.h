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
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/data_op/gs_op.h"

namespace embedx {
namespace graph_op {

class DistSharedNegativeSampler : public DistGSOp {
 public:
  ~DistSharedNegativeSampler() override = default;

 public:
  bool Run(int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
           std::vector<vec_int_t>* sampled_nodes_list) const;
};

}  // namespace graph_op
}  // namespace embedx
