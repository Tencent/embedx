// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//

#include "src/tools/graph/swing/swing.h"

#include <deepx_core/dx_log.h>
#include <gtest/gtest.h>

#include <algorithm>  // std::find_if
#include <memory>     // std::unique_ptr
#include <string>
#include <utility>  // std::make_pair

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"

namespace embedx {

class SwingTest : public ::testing::Test {
 protected:
  std::unique_ptr<Swing> swing_;

 protected:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;
  SwingConfig swing_config_;

 protected:
  const std::string USER_CONTEXT = "testdata/swing_user_context";

 protected:
  void SetUp() override {
    graph_config_.set_node_graph(USER_CONTEXT);
    graph_client_ = NewGraphClient(graph_config_, GraphClientEnum::LOCAL);
    DXCHECK(graph_client_ != nullptr);

    swing_config_.alpha = 1.0;
    swing_config_.cache_thld = 10;
    swing_config_.sample_thld = 5000;
    swing_.reset(new Swing(graph_client_.get(), swing_config_));
  }
};

TEST_F(SwingTest, ComputeItemScore) {
  // user context
  // 0 5:1.0 6:1.0 7:1.0 8:1.0 9:1.0
  // 1 5:1.0 7:1.0 8:1.0 9:1.0
  // 2 5:1.0 9:1.0 10:1.0 11:1.0
  // 3 5:1.0 11:1.0
  // 4 5:1.0 11:1.0 12:1.0 13:1.0

  // item context
  // 5 0:1.0 1:1.0 2:1.0 3:1.0 4:1.0

  vec_int_t item_nodes = {5};
  std::vector<vec_pair_t> item_contexts = {
      {std::make_pair((int_t)0, 1.0), std::make_pair((int_t)1, 1.0),
       std::make_pair((int_t)2, 1.0), std::make_pair((int_t)3, 1.0),
       std::make_pair((int_t)4, 1.0)}};
  std::vector<vec_pair_t> item_scores;
  (void)swing_->ComputeItemScore(item_nodes, item_contexts, &item_scores);

  // check item score
  auto find_item_score = [&item_scores](int_t item) {
    auto it =
        std::find_if(item_scores[0].begin(), item_scores[0].end(),
                     [item](const pair_t& pair) { return pair.first == item; });
    DXCHECK(it != item_scores[0].end());
    return it->second;
  };

  EXPECT_EQ(item_scores[0].size(), 4u);
  EXPECT_NEAR(find_item_score((int_t)7), (float_t)0.0721, 1e-4);
  EXPECT_NEAR(find_item_score((int_t)8), (float_t)0.0721, 1e-4);
  EXPECT_NEAR(find_item_score((int_t)9), (float_t)0.3831, 1e-4);
  EXPECT_NEAR(find_item_score((int_t)11), (float_t)0.744, 1e-4);
}

}  // namespace embedx
