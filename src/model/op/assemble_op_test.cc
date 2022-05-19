// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//

#include <gtest/gtest.h>

#include "src/model/op/gnn_graph_node.h"
#include "src/model/op/op_test.h"

namespace embedx {

class AssembleOpBaseTest : public testing::Test, public deepx_core::DataType {
 protected:
  const csr_t X_{{0, 1, 2, 3}, {1, 2, 3}, {1, 1, 1}};
};

TEST_F(AssembleOpBaseTest, AssembleOpForward) {
  deepx_core::InstanceNode X("X", Shape(3, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::ConstantNode Y("Y", Shape(3, 2), {0, 1, 2, 3, 4, 5});
  deepx_core::VariableNode W("W", Shape(3, 2), deepx_core::TENSOR_TYPE_SRM,
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  AssembleNode Z("Z", &X, &Y, &W);
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  tsr_t expected_Z{{0, 1}, {2, 3}, {4, 5}};
  CheckOpForward(&Z, 0, expected_Z, nullptr, nullptr, inst_initializer);
}

TEST_F(AssembleOpBaseTest, AssembleOpBackward) {
  deepx_core::InstanceNode X("X", Shape(3, 0), deepx_core::TENSOR_TYPE_CSR);
  deepx_core::VariableNode Y("Y", Shape(3, 2),
                             deepx_core::TENSOR_INITIALIZER_TYPE_ONES, 1, 1);
  deepx_core::VariableNode W("W", Shape(3, 2), deepx_core::TENSOR_TYPE_SRM,
                             deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0, 1);
  AssembleNode Z("Z", &X, &Y, &W);
  auto inst_initializer = [this](deepx_core::Instance* inst) {
    inst->insert<csr_t>("X") = X_;
  };
  CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  const srm_t OW{{1, 2, 3}, {{1, 1}, {1, 1}, {1, 1}}};
  CheckOpOverwrittenParam(&Z, "W", 0, OW, nullptr, nullptr, inst_initializer);
}

}  // namespace embedx
