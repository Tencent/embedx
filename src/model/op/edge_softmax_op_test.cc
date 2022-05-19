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

class EdgeSoftmaxOpForwardTest : public testing::Test,
                                 public deepx_core::DataType {
 protected:
  // row 0: 0=1 1=1 3=1
  // row 1: 2=1 3=1 4=1 6=1
  // row 2: 1=1 5=1
  const csr_t X_{
      {0, 3, 7, 9}, {0, 1, 3, 2, 3, 4, 6, 1, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
};

class EdgeSoftmaxOpBackwardTest : public testing::Test,
                                  public deepx_core::DataType {
 protected:
  // row 0: 0=1 1=1 3=1
  // row 1: 2=1 3=1 4=1 6=1
  // row 2: 1=1 5=1
  const csr_t X_{
      {0, 3, 7, 9}, {0, 1, 3, 2, 3, 4, 6, 1, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};

 protected:
  void TestNumHead(int num_heads) {
    deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
    deepx_core::VariableNode W("W", Shape(num_heads, 9),
                               deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 5);
    EdgeSoftmaxNode Z("Z", &X, &W);
    auto inst_initializer = [this](deepx_core::Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }
};

TEST_F(EdgeSoftmaxOpForwardTest, NumHead1) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::ConstantNode W("W", Shape(1, 9), {1, 2, 3, 6, 5, 4, 7, 8, 9});
  EdgeSoftmaxNode Z("Z", &X, &W);

  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  tsr_t expected_Z{0.09003, 0.24472, 0.66524, 0.23688, 0.08714,
                   0.03205, 0.64391, 0.26894, 0.73105};
  expected_Z.reshape(1, 9);
  CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
}

TEST_F(EdgeSoftmaxOpForwardTest, NumHead2) {
  deepx_core::InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::ConstantNode W(
      "W", Shape(2, 9), {1, 2, 4, 6, 5, 3, 7, 8, 9, 3, 2, 1, 4, 5, 6, 9, 8, 7});
  EdgeSoftmaxNode Z("Z", &X, &W);

  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  tsr_t expected_Z{0.04201, 0.11419, 0.84379, 0.24178, 0.08894, 0.01203,
                   0.65723, 0.26894, 0.73105, 0.66524, 0.24472, 0.09003,
                   0.00626, 0.01704, 0.04632, 0.93037, 0.73105, 0.26894};
  expected_Z.reshape(2, 9);
  CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
}

TEST_F(EdgeSoftmaxOpBackwardTest, NumHead1) { TestNumHead(1); }

TEST_F(EdgeSoftmaxOpBackwardTest, NumHead5) { TestNumHead(5); }

}  // namespace embedx
