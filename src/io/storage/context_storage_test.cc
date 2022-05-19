// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chunchen Su (chunchen.scut@gmail.com)
//

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <vector>

#include "src/io/storage/adjacency.h"
#include "src/io/storage/storage.h"
#include "src/io/value.h"

namespace embedx {

class ContextStorageTest : public ::testing::Test {
 protected:
  std::unique_ptr<Storage> context_store_;
  std::vector<AdjValue> context_values_;

 protected:
  const uint64_t ESTIMATED_SIZE = 10;

 protected:
  void SetUp() override {
    AdjValue context_value;
    context_values_.clear();
    for (int i = 0; i < 5; ++i) {
      context_value.node = i;
      context_value.pairs.emplace_back(i, i);
      context_values_.emplace_back(context_value);
    }
  }
};

TEST_F(ContextStorageTest, Insert_AdjList) {
  context_store_ = NewContextStorage((int)AdjacencyEnum::ADJ_LIST);
  context_store_->Clear();
  context_store_->Reserve(ESTIMATED_SIZE);

  for (auto value : context_values_) {
    EXPECT_TRUE(context_store_->InsertContext(&value));
  }

  EXPECT_EQ(context_store_->Size(), 5u);
  EXPECT_TRUE(!context_store_->Empty());

  for (size_t i = 0; i < context_store_->Keys().size(); ++i) {
    auto node = context_store_->Keys()[i];
    EXPECT_EQ(context_store_->FindNeighbor(node)->size(), i + 1);
    EXPECT_EQ(context_store_->GetInDegree(node), 5 - (int)i);
    EXPECT_EQ(context_store_->GetOutDegree(node), (int)i + 1);
  }
}

TEST_F(ContextStorageTest, Insert_AdjMatrix) {
  context_store_ = NewContextStorage((int)AdjacencyEnum::ADJ_MATRIX);
  context_store_->Clear();
  context_store_->Reserve(ESTIMATED_SIZE);

  for (auto value : context_values_) {
    EXPECT_TRUE(context_store_->InsertContext(&value));
  }

  EXPECT_EQ(context_store_->Size(), 5u);
  EXPECT_TRUE(!context_store_->Empty());

  for (size_t i = 0; i < context_store_->Keys().size(); ++i) {
    auto node = context_store_->Keys()[i];
    EXPECT_EQ(context_store_->FindNeighbor(node)->size(), i + 1);
    EXPECT_EQ(context_store_->GetInDegree(node), 5 - (int)i);
    EXPECT_EQ(context_store_->GetOutDegree(node), (int)i + 1);
  }
}

}  // namespace embedx
