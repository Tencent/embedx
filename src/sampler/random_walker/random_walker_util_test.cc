// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/sampler/random_walker/random_walker_util.h"

#include <deepx_core/tensor/ll_tensor.h>
#include <gtest/gtest.h>

#include <utility>  // std::unique_ptr, std::pair

#include "src/common/data_types.h"

namespace embedx {
namespace random_walker_util {

class RandomWalkerUtilTest : public ::testing::Test {
 protected:
  using ll_sparse_tensor_t = ::deepx_core::LLSparseTensor<float_t, int_t>;

 protected:
  vec_pair_t context_;

 protected:
  void SetUp() override {
    context_ = {{ll_sparse_tensor_t::make_feature_id(1, 1), 1},
                {ll_sparse_tensor_t::make_feature_id(1, 2), 1},
                {ll_sparse_tensor_t::make_feature_id(3, 1), 1},
                {ll_sparse_tensor_t::make_feature_id(3, 2), 1},
                {ll_sparse_tensor_t::make_feature_id(4, 1), 1},
                {ll_sparse_tensor_t::make_feature_id(4, 2), 1},
                {ll_sparse_tensor_t::make_feature_id(5, 1), 1}};
  }
};

TEST_F(RandomWalkerUtilTest, FindBound) {
  std::pair<int, int> bound;
  EXPECT_FALSE(FindBound(context_, 0, &bound));
  EXPECT_FALSE(FindBound(context_, 2, &bound));
  EXPECT_FALSE(FindBound(context_, 6, &bound));

  EXPECT_TRUE(FindBound(context_, 1, &bound));
  EXPECT_EQ(bound.first, 0);
  EXPECT_EQ(bound.second, 2);

  EXPECT_TRUE(FindBound(context_, 3, &bound));
  EXPECT_EQ(bound.first, 2);
  EXPECT_EQ(bound.second, 4);

  EXPECT_TRUE(FindBound(context_, 5, &bound));
  EXPECT_EQ(bound.first, 6);
  EXPECT_EQ(bound.second, 7);
}

TEST_F(RandomWalkerUtilTest, ContainsNode) {
  EXPECT_FALSE(
      ContainsNode(context_, ll_sparse_tensor_t::make_feature_id(2, 1)));
  EXPECT_FALSE(
      ContainsNode(context_, ll_sparse_tensor_t::make_feature_id(1, 3)));

  EXPECT_TRUE(
      ContainsNode(context_, ll_sparse_tensor_t::make_feature_id(1, 1)));
  EXPECT_TRUE(
      ContainsNode(context_, ll_sparse_tensor_t::make_feature_id(3, 2)));
}

}  // namespace random_walker_util
}  // namespace embedx
