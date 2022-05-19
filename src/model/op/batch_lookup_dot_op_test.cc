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

#include "src/common/data_types.h"
#include "src/model/op/gnn_graph_node.h"
#include "src/model/op/op_test.h"

namespace embedx {

class GraphBatchLookupDotOpForwardTest : public testing::Test,
                                         public deepx_core::DataType {
 protected:
  // input ids {0,1,2,3}
  csr_t Xin_{{0, 1, 2, 3, 4}, {0, 1, 2, 3}, {1, 1, 1, 1}};
  // output ids {1,3,2,4}
  csr_t Xout_{{0, 1, 2, 3, 4}, {1, 3, 2, 4}, {1, 1, 1, 1}};

 protected:
  void TestInputTensorType(int Win_tensor_type, int Wout_tensor_type) {
    deepx_core::InstanceNode Xin("Xin", Shape(-1, 0),
                                 deepx_core::TENSOR_TYPE_CSR);
    deepx_core::InstanceNode Xout("Xout", Shape(-1, 0),
                                  deepx_core::TENSOR_TYPE_CSR);
    deepx_core::VariableNode Win("Win", Shape(5, 3), Win_tensor_type);
    deepx_core::VariableNode Wout("Wout", Shape(5, 3), Wout_tensor_type);
    BatchLookupDotNode Z("Z", &Xin, &Xout, &Win, &Wout);

    auto post_param_initializer = [this, Win_tensor_type, Wout_tensor_type](
                                      std::default_random_engine& /*engine*/,
                                      deepx_core::TensorMap* param) {
      InitParam("Win", Win_tensor_type, param);
      InitParam("Wout", Wout_tensor_type, param);
    };

    auto inst_initializer = [this](deepx_core::Instance* inst) {
      inst->insert<csr_t>("Xin") = Xin_;
      inst->insert<csr_t>("Xout") = Xout_;
    };

    tsr_t expected_Z{14, 122, 149, 392};
    expected_Z.reshape(4, 1);
    CheckOpForward(&Z, 0, expected_Z, nullptr, post_param_initializer,
                   inst_initializer);
  }

  void InitParam(const std::string& name, int tensor_type,
                 deepx_core::TensorMap* param) {
    if (tensor_type == deepx_core::TENSOR_TYPE_TSR) {
      auto& W = param->get<tsr_t>(name);
      W.arange();
    } else {
      auto& W = param->get<srm_t>(name);
      vec_float_t values;
      values = {0, 1, 2};
      W.assign(0, values.data());
      values = {3, 4, 5};
      W.assign(1, values.data());
      values = {6, 7, 8};
      W.assign(2, values.data());
      values = {9, 10, 11};
      W.assign(3, values.data());
      values = {12, 13, 14};
      W.assign(4, values.data());
    }
  }
};

TEST_F(GraphBatchLookupDotOpForwardTest, BatchLookupDot_TSR_TSR) {
  TestInputTensorType(deepx_core::TENSOR_TYPE_TSR, deepx_core::TENSOR_TYPE_TSR);
}

TEST_F(GraphBatchLookupDotOpForwardTest, BatchLookupDot_TSR_SRM) {
  TestInputTensorType(deepx_core::TENSOR_TYPE_TSR, deepx_core::TENSOR_TYPE_SRM);
}

TEST_F(GraphBatchLookupDotOpForwardTest, BatchLookupDot_SRM_TSR) {
  TestInputTensorType(deepx_core::TENSOR_TYPE_TSR, deepx_core::TENSOR_TYPE_SRM);
}

TEST_F(GraphBatchLookupDotOpForwardTest, BatchLookupDot_SRM_SRM) {
  TestInputTensorType(deepx_core::TENSOR_TYPE_SRM, deepx_core::TENSOR_TYPE_SRM);
}

class GraphBatchLookupDotOpBackwardTest : public testing::Test,
                                          public deepx_core::DataType {
 protected:
  // input ids {0,1,2,4,3,1}
  csr_t Xin_{{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 4, 3, 1}, {1, 1, 1, 1, 1, 1}};
  // output ids {0,3,2,4,1,2}
  csr_t Xout_{{0, 1, 2, 3, 4, 5, 6}, {0, 3, 2, 4, 1, 2}, {1, 1, 1, 1, 1, 1}};

  // for copy param to hidden
  csr_t in_{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}, {1, 1, 1, 1, 1}};
  csr_t out_{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}, {1, 1, 1, 1, 1}};

 protected:
  void TestInputTensorTypeAndNodeType(int in_tensor_type, int in_node_type,
                                      int out_tensor_type, int out_node_type) {
    deepx_core::InstanceNode Xin("Xin", Shape(-1, 0),
                                 deepx_core::TENSOR_TYPE_CSR);
    deepx_core::InstanceNode Xout("Xout", Shape(-1, 0),
                                  deepx_core::TENSOR_TYPE_CSR);
    // graph_node: param
    deepx_core::VariableNode Win("Win", Shape(5, 3), in_tensor_type,
                                 deepx_core::TENSOR_INITIALIZER_TYPE_RAND, 0,
                                 1);
    deepx_core::VariableNode Wout("Wout", Shape(5, 3), out_tensor_type,
                                  deepx_core::TENSOR_INITIALIZER_TYPE_RAND, 0,
                                  1);

    // copy param to hidden
    deepx_core::InstanceNode in("in", Shape(-1, 0),
                                deepx_core::TENSOR_TYPE_CSR);
    deepx_core::InstanceNode out("out", Shape(-1, 0),
                                 deepx_core::TENSOR_TYPE_CSR);
    deepx_core::EmbeddingLookupNode in_embed("in_embed", &in, &Win);
    deepx_core::EmbeddingLookupNode out_embed("out_embed", &out, &Wout);

    auto post_param_initializer = [this, in_tensor_type, out_tensor_type](
                                      std::default_random_engine& engine,
                                      deepx_core::TensorMap* param) {
      if (in_tensor_type == deepx_core::TENSOR_TYPE_SRM) {
        auto& Win = param->get<srm_t>("Win");
        for (int i = 0; i < Xin_.row(); ++i) {
          Win.get_row(engine, Xin_.col(i));
        }
      }
      if (out_tensor_type == deepx_core::TENSOR_TYPE_SRM) {
        auto& Wout = param->get<srm_t>("Wout");
        for (int i = 0; i < Xout_.row(); ++i) {
          Wout.get_row(engine, Xout_.col(i));
        }
      }
    };

    auto inst_initializer = [this, in_node_type,
                             out_node_type](deepx_core::Instance* inst) {
      inst->insert<csr_t>("Xin") = Xin_;
      inst->insert<csr_t>("Xout") = Xout_;
      if (in_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
        inst->insert<csr_t>("in") = in_;
      }
      if (out_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
        inst->insert<csr_t>("out") = out_;
      }
    };

    if (in_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
        out_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      BatchLookupDotNode Z("Z", &Xin, &Xout, &in_embed, &out_embed);
      CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
    } else if (in_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      BatchLookupDotNode Z("Z", &Xin, &Xout, &in_embed, &Wout);
      CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
    } else if (out_node_type == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      BatchLookupDotNode Z("Z", &Xin, &Xout, &Win, &out_embed);
      CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
    } else {
      BatchLookupDotNode Z("Z", &Xin, &Xout, &Win, &Wout);
      CheckOpBackward(&Z, 0, nullptr, post_param_initializer, inst_initializer);
    }
  }
};

// tsr_hidden,tsr_param,srm, 3 x 3 = 9 cases
TEST_F(GraphBatchLookupDotOpBackwardTest,
       BatchLookupDot_TSR_HIDDEN_TSR_HIDDEN) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_TSR_HIDDEN_TSR_PARM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_TSR_PARAM_TSR_HIDDEN) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_TSR_HIDDEN_SRM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN,
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_SRM_TSR_HIDDEN) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_HIDDEN);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_TSR_PARAM_TSR_PARAM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_TSR_PARAM_SRM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_SRM_TSR_PARAM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_TSR, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

TEST_F(GraphBatchLookupDotOpBackwardTest, BatchLookupDot_SRM_SRM) {
  TestInputTensorTypeAndNodeType(
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM,
      deepx_core::TENSOR_TYPE_SRM, deepx_core::GRAPH_NODE_TYPE_PARAM);
}

}  // namespace embedx
