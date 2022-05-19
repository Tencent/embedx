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

#include "src/io/line_parser.h"

#include <gtest/gtest.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/io/value.h"

namespace embedx {

class LineParserTest : public ::testing::Test {
 protected:
  std::unique_ptr<LineParser> parser_;

 protected:
  const std::string CONTEXT = "testdata/context/context-0";
  const std::string FEATURE_FILE = "testdata/node_feature/feature-0";
  const std::string WALK_FILE = "testdata/walk_file";
  const std::string LABEL_FILE = "testdata/label_file";
  const int BATCH = 2;

 protected:
  void SetUp() override { parser_.reset(new LineParser); }
};

TEST_F(LineParserTest, NextBatch_Node) {
  EXPECT_TRUE(parser_->Open(CONTEXT));

  // next batch
  std::vector<NodeValue> values;
  EXPECT_TRUE(parser_->NextBatch<NodeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "0 1");
  EXPECT_EQ(values[1].ToString(), "3 1");

  // next batch
  EXPECT_TRUE(parser_->NextBatch<NodeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "6 1");
  EXPECT_EQ(values[1].ToString(), "9 1");

  // last batch
  EXPECT_TRUE(parser_->NextBatch<NodeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0].ToString(), "12 1");

  // end batch
  EXPECT_FALSE(parser_->NextBatch<NodeValue>(BATCH, &values));
}

TEST_F(LineParserTest, NextBatch_Edge) {
  EXPECT_TRUE(parser_->Open(WALK_FILE));

  // next batch
  std::vector<EdgeValue> values;
  EXPECT_TRUE(parser_->NextBatch<EdgeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "1 2 1.1");
  EXPECT_EQ(values[1].ToString(), "2 4 1.2");

  // next batch
  EXPECT_TRUE(parser_->NextBatch<EdgeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "3 5 2.1");
  EXPECT_EQ(values[1].ToString(), "4 5 2.2");

  // last batch
  EXPECT_TRUE(parser_->NextBatch<EdgeValue>(BATCH, &values));
  EXPECT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0].ToString(), "5 6 3.1");

  // end batch
  EXPECT_FALSE(parser_->NextBatch<EdgeValue>(BATCH, &values));
}

TEST_F(LineParserTest, NextBatch_Context) {
  EXPECT_TRUE(parser_->Open(CONTEXT));

  // next batch
  std::vector<AdjValue> values;
  EXPECT_TRUE(parser_->NextBatch<AdjValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "0 12:1.1 11:1.2 10:1.3");
  EXPECT_EQ(values[1].ToString(), "3 2:1.1 1:1.2 0:1.3");

  // next batch
  EXPECT_TRUE(parser_->NextBatch<AdjValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "6 5:1.1 4:1.2 3:1.3");
  EXPECT_EQ(values[1].ToString(), "9 8:1.1 7:1.2 6:1.3");

  // last batch
  EXPECT_TRUE(parser_->NextBatch<AdjValue>(BATCH, &values));
  EXPECT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0].ToString(), "12 11:1.1 10:1.2 9:1.3");

  // end batch
  EXPECT_FALSE(parser_->NextBatch<AdjValue>(BATCH, &values));
}

TEST_F(LineParserTest, NextBatch_Feature) {
  EXPECT_TRUE(parser_->Open(FEATURE_FILE));

  // next batch
  std::vector<AdjValue> values;
  EXPECT_TRUE(parser_->NextBatch<AdjValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "0 10:1.1 20:1.2");
  EXPECT_EQ(values[1].ToString(), "3 31:1.1 32:1.2");

  // last batch
  EXPECT_TRUE(parser_->NextBatch<AdjValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "6 61:1.1 62:1.2");
  EXPECT_EQ(values[1].ToString(), "9 91:1.1 92:1.2");

  // end batch
  EXPECT_FALSE(parser_->NextBatch<AdjValue>(BATCH, &values));
}

TEST_F(LineParserTest, NextBatch_NodeAndLabel) {
  EXPECT_TRUE(parser_->Open(LABEL_FILE));

  // next batch
  std::vector<NodeAndLabelValue> values;
  EXPECT_TRUE(parser_->NextBatch<NodeAndLabelValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "1 0");
  EXPECT_EQ(values[1].ToString(), "2 0");

  // next batch
  EXPECT_TRUE(parser_->NextBatch<NodeAndLabelValue>(BATCH, &values));
  EXPECT_EQ(values.size(), (size_t)BATCH);
  EXPECT_EQ(values[0].ToString(), "3 1");
  EXPECT_EQ(values[1].ToString(), "4 2");

  // last batch
  EXPECT_TRUE(parser_->NextBatch<NodeAndLabelValue>(BATCH, &values));
  EXPECT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0].ToString(), "5 0");

  // end batch
  EXPECT_FALSE(parser_->NextBatch<NodeAndLabelValue>(BATCH, &values));
}

}  // namespace embedx
