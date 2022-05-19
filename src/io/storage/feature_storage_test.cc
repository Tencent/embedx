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

#include "src/common/data_types.h"
#include "src/io/storage/adjacency.h"
#include "src/io/storage/storage.h"
#include "src/io/value.h"

namespace embedx {

class FeatureStorageTest : public ::testing::Test {
 protected:
  std::unique_ptr<Storage> feature_store_;
  std::vector<AdjValue> feature_values_;

 protected:
  const uint64_t ESTIMATED_SIZE = 10;

 protected:
  void SetUp() override {
    feature_values_.clear();
    for (int i = 0; i < 5; ++i) {
      AdjValue feature_value;
      feature_value.node = i;
      feature_value.pairs.emplace_back(i, i);
      feature_values_.emplace_back(feature_value);
    }
  }
};

TEST_F(FeatureStorageTest, Insert_AdjList) {
  feature_store_ = NewFeatureStorage((int)AdjacencyEnum::ADJ_LIST);
  feature_store_->Clear();
  feature_store_->Reserve(ESTIMATED_SIZE);

  for (auto value : feature_values_) {
    EXPECT_TRUE(feature_store_->InsertFeature(&value));
  }

  EXPECT_EQ(feature_store_->Size(), 5u);
  EXPECT_TRUE(!feature_store_->Empty());

  for (size_t i = 0; i < feature_store_->Keys().size(); ++i) {
    auto node = feature_store_->Keys()[i];
    auto* feature = feature_store_->FindNeighbor(node);
    EXPECT_EQ(feature->size(), 1u);
    for (auto& pair : *feature) {
      EXPECT_EQ(pair.first, (int_t)i);
      EXPECT_EQ(pair.second, (float_t)i);
    }
  }

  for (size_t i = 0; i < feature_store_->Keys().size(); ++i) {
    auto node = feature_store_->Keys()[i];
    EXPECT_EQ(feature_store_->GetInDegree(node), 0);
    EXPECT_EQ(feature_store_->GetOutDegree(node), 0);
  }
}

TEST_F(FeatureStorageTest, Insert_AdjMatrix) {
  feature_store_ = NewFeatureStorage((int)AdjacencyEnum::ADJ_MATRIX);
  feature_store_->Clear();
  feature_store_->Reserve(ESTIMATED_SIZE);

  for (auto value : feature_values_) {
    EXPECT_TRUE(feature_store_->InsertFeature(&value));
  }

  EXPECT_EQ(feature_store_->Size(), 5u);
  EXPECT_TRUE(!feature_store_->Empty());

  for (size_t i = 0; i < feature_store_->Keys().size(); ++i) {
    auto node = feature_store_->Keys()[i];
    auto* feature = feature_store_->FindNeighbor(node);
    EXPECT_EQ(feature->size(), 1u);
    for (auto& pair : *feature) {
      EXPECT_EQ(pair.first, (int_t)i);
      EXPECT_EQ(pair.second, (float_t)i);
    }
  }

  for (size_t i = 0; i < feature_store_->Keys().size(); ++i) {
    auto node = feature_store_->Keys()[i];
    EXPECT_EQ(feature_store_->GetInDegree(node), 0);
    EXPECT_EQ(feature_store_->GetOutDegree(node), 0);
  }
}

}  // namespace embedx
