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

#include "src/io/loader/freq_file_loader.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>

namespace embedx {

class FreqFileLoaderTest : public ::testing::Test {
 protected:
  std::unique_ptr<FreqFileLoader> loader_;

 protected:
  const int THREAD_NUM = 3;
  const std::string FREQ_FILE = "testdata/freq";
  const std::string USER_ITEM_FREQ_FILE = "testdata/user_item_freq";
  const std::string USER_ITEM_CONFIG = "testdata/user_item_config";
};

TEST_F(FreqFileLoaderTest, LoadFreqFile_OneNameSpace) {
  loader_ = FreqFileLoader::Create("", FREQ_FILE, THREAD_NUM);
  EXPECT_TRUE(loader_ != nullptr);

  EXPECT_EQ(loader_->ns_size(), 1);
  EXPECT_EQ(loader_->id_name_map().size(), 1u);

  EXPECT_EQ(loader_->nodes_list()[0].size(), 9u);
  EXPECT_EQ(loader_->freqs_list()[0].size(), 9u);
}

TEST_F(FreqFileLoaderTest, LoadFreqFile_TwoNameSpace) {
  loader_ =
      FreqFileLoader::Create(USER_ITEM_CONFIG, USER_ITEM_FREQ_FILE, THREAD_NUM);
  EXPECT_TRUE(loader_ != nullptr);

  EXPECT_EQ(loader_->ns_size(), 2);
  EXPECT_EQ(loader_->id_name_map().size(), 2u);

  // ns 0
  EXPECT_EQ(loader_->nodes_list()[0].size(), 13u);
  EXPECT_EQ(loader_->freqs_list()[0].size(), 13u);

  // ns 1
  EXPECT_EQ(loader_->nodes_list()[1].size(), 13u);
  EXPECT_EQ(loader_->freqs_list()[1].size(), 13u);
}

}  // namespace embedx
