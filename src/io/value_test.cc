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

#include "src/io/value.h"

#include <gtest/gtest.h>

#include <vector>

#include "src/common/data_types.h"
#include "src/io/value.h"

namespace embedx {

class ValueTest : public ::testing::Test {
 protected:
  int batch = 2;
};

TEST_F(ValueTest, EdgeValue) {
  std::vector<EdgeValue> edge_values(batch);
  for (int i = 0; i < batch; ++i) {
    edge_values[i].src_node = i;
    edge_values[i].dst_node = i;
    edge_values[i].weight = i;
  }

  auto src_nodes = Collect<EdgeValue, int_t>(edge_values, &EdgeValue::src_node);
  auto dst_nodes = Collect<EdgeValue, int_t>(edge_values, &EdgeValue::dst_node);
  auto weights = Collect<EdgeValue, float_t>(edge_values, &EdgeValue::weight);

  for (int i = 0; i < batch; ++i) {
    EXPECT_EQ(src_nodes[i], (int_t)i);
    EXPECT_EQ(dst_nodes[i], (int_t)i);
    EXPECT_EQ(weights[i], (float_t)i);
  }
}

TEST_F(ValueTest, AdjValue) {
  std::vector<AdjValue> feat_values(batch);
  for (int i = 0; i < batch; ++i) {
    feat_values[i].node = i;
    feat_values[i].pairs = {{i, 1.0}, {i + 1, 1.0}};
  }

  auto nodes = Collect<AdjValue, int_t>(feat_values, &AdjValue::node);

  for (int i = 0; i < batch; ++i) {
    EXPECT_EQ(nodes[i], (int_t)i);
  }
}

}  // namespace embedx
