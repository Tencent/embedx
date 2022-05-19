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

#include "src/graph/post_builder.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/io/loader/loader.h"

namespace embedx {

class PostBuilderTest : public ::testing::Test {
 protected:
  std::unique_ptr<PostBuilder> post_builder_;
  std::unique_ptr<Loader> loader_;
  GraphConfig config_;

 protected:
  const std::string CONTEXT = "testdata/context";
  const std::string USER_ITEM_CONTEXT = "testdata/user_item_context";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";

  const int THREAD_NUM = 3;

 protected:
  void SetUp() override { config_.set_thread_num(THREAD_NUM); }
};

TEST_F(PostBuilderTest, Build_OneNameSapce) {
  // loader
  loader_ = NewContextLoader();
  EXPECT_TRUE(loader_->Load(CONTEXT, THREAD_NUM));

  // post_builder
  post_builder_ = PostBuilder::Create(loader_->storage(), config_);
  EXPECT_TRUE(post_builder_ != nullptr);

  EXPECT_EQ(post_builder_->uniq_nodes_list()[0].size(), 13u);
  EXPECT_EQ(post_builder_->total_freqs()[0], (int_t)52);
}

TEST_F(PostBuilderTest, Build_TwoNameSpace) {
  // loader
  loader_ = NewContextLoader();
  EXPECT_TRUE(loader_->Load(USER_ITEM_CONTEXT, THREAD_NUM));

  // post builder
  config_.set_node_config(USER_ITEM_CONFIG);
  post_builder_ = PostBuilder::Create(loader_->storage(), config_);
  EXPECT_TRUE(post_builder_ != nullptr);

  // namespace : user
  EXPECT_EQ(post_builder_->uniq_nodes_list()[0].size(), 13u);
  EXPECT_EQ(post_builder_->total_freqs()[0], (int_t)52);

  // namepspace : item
  EXPECT_EQ(post_builder_->uniq_nodes_list()[1].size(), 13u);
  EXPECT_EQ(post_builder_->total_freqs()[1], (int_t)52);
}

}  // namespace embedx
