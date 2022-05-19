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

#include "src/graph/data_op/context_lookuper_op/context.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"

namespace embedx {
namespace graph_op {

class ContextTest : public ::testing::Test {
 protected:
  std::unique_ptr<InMemoryGraph> graph_;
  std::unique_ptr<Context> context_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";

 protected:
  void SetUp() override { config_.set_node_graph(CONTEXT); }
};

TEST_F(ContextTest, Lookup) {
  graph_ = InMemoryGraph::Create(config_);
  EXPECT_TRUE(graph_ != nullptr);

  context_ = NewContext(graph_.get());
  EXPECT_TRUE(context_ != nullptr);

  vec_int_t nodes = {0, 1, 2};
  std::vector<vec_pair_t> contexts;

  EXPECT_TRUE(context_->Lookup(nodes, &contexts));
  EXPECT_EQ(nodes.size(), contexts.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto* real_context = graph_->FindContext(nodes[i]);

    EXPECT_TRUE(real_context != nullptr);
    EXPECT_EQ(contexts[i].size(), real_context->size());

    for (size_t j = 0; j < contexts[i].size(); ++j) {
      EXPECT_EQ(contexts[i][j], (*real_context)[j]);
    }
  }
}

}  // namespace graph_op
}  // namespace embedx
