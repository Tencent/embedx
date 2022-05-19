// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//

#include <gtest/gtest.h>

#include "src/model/op/gnn_graph_node.h"
#include "src/model/op/op_test.h"

namespace embedx {

class GnnOpForwardTest : public testing::Test, public deepx_core::DataType {
 protected:
  // csr: row_offset, col, val
  const csr_t X_{
      {0, 3, 7, 9}, {0, 1, 3, 2, 3, 4, 6, 1, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
};

class GnnOpBackwardTest : public testing::Test, public deepx_core::DataType {
 protected:
  const csr_t X_{
      {0, 3, 7, 9}, {0, 1, 3, 2, 3, 4, 6, 1, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
};

TEST_F(GnnOpForwardTest, MeanAggregatorOp) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::ConstantNode W("W", Shape(7, 2),
                             {1, 1, 1, 2, 2, 3, 1, 3, 5, 6, 3, 4, 4, 6});
  MeanAggregatorNode Z("Z", &X, &W);
  tsr_t expected_Z{{1, 2}, {3, 4.5}, {2, 3}};
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
}

TEST_F(GnnOpBackwardTest, MeanAggregatorOp) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::VariableNode W("W", Shape(7, 10),
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  MeanAggregatorNode Z("Z", &X, &W);
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
}

TEST_F(GnnOpForwardTest, SumAggregatorOp) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::ConstantNode W("W", Shape(7, 2),
                             {1, 1, 1, 2, 2, 3, 1, 3, 5, 6, 3, 4, 4, 6});
  SumAggregatorNode Z("Z", &X, &W);
  tsr_t expected_Z{{3, 6}, {12, 18}, {4, 6}};
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
}

TEST_F(GnnOpBackwardTest, SumAggregatorOp) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::VariableNode W("W", Shape(7, 10),
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  SumAggregatorNode Z("Z", &X, &W);
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
}
}  // namespace embedx
