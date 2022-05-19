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

#include <gtest/gtest.h>

#include "src/model/op/gnn_graph_node.h"
#include "src/model/op/op_test.h"

namespace embedx {

class WeightedAverageOpForwardTest : public testing::Test,
                                     public deepx_core::DataType {};
class WeightedAverageOpBackwardTest : public testing::Test,
                                      public deepx_core::DataType {};

TEST_F(WeightedAverageOpForwardTest, WeightedAverageOpForward) {
  deepx_core::VariableNode X("X", Shape(2, 4),
                             deepx_core::TENSOR_INITIALIZER_TYPE_ONES, 1, 1);
  deepx_core::VariableNode W("W", Shape(2, 2),
                             deepx_core::TENSOR_INITIALIZER_TYPE_ONES, 1, 1);
  WeightedAverageNode Z("Z", &X, &W);
  tsr_t expected_Z{{1, 1}, {1, 1}};
  CheckOpForward(&Z, 0, expected_Z);
}

TEST_F(WeightedAverageOpBackwardTest, WeightedAverageOpBackward) {
  deepx_core::VariableNode X("X", Shape(2, 12),
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  deepx_core::VariableNode W("W", Shape(2, 3),
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  WeightedAverageNode Z("Z", &X, &W);
  CheckOpBackward(&Z, 0);
}
}  // namespace embedx
