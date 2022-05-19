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

class GraphHiddenLookupOpBackwardTest : public testing::Test,
                                        public deepx_core::DataType {
 protected:
  const csr_t X_{{0, 1, 4, 6, 7, 10, 14},
                 {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7},
                 {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3}};

 protected:
  void TestTSR(const Shape& Wshape) {
    InstanceNode X("X", Shape(-1, 0), deepx_core::TENSOR_TYPE_CSR);
    VariableNode W("W", Wshape, deepx_core::TENSOR_INITIALIZER_TYPE_RANDN, 0,
                   1);
    HiddenLookupNode Z("Z", &X, &W);
    auto inst_initializer = [this](deepx_core::Instance* inst) {
      inst->insert<csr_t>("X") = X_;
    };
    CheckOpBackward(&Z, 0, nullptr, nullptr, inst_initializer);
  }
};

TEST_F(GraphHiddenLookupOpBackwardTest, EmbeddingLookupTSRWcol1) {
  TestTSR(Shape(10, 1));
}

TEST_F(GraphHiddenLookupOpBackwardTest, EmbeddingLookupTSRWcol4) {
  TestTSR(Shape(10, 4));
}

}  // namespace embedx
