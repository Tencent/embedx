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

#pragma once
#include <memory>  //std::unique_ptr
#include <vector>

#include "src/deep/data_op/deep_op.h"
#include "src/deep/data_op/deep_op_resource.h"
#include "src/deep/data_op/feature_lookuper_op/item_feature.h"

namespace embedx {
namespace deep_op {

class ItemFeatureLookuper : public LocalDeepOp {
 private:
  std::unique_ptr<ItemFeature> feature_;

 public:
  ~ItemFeatureLookuper() override = default;

 public:
  bool Run(const vec_int_t& items, std::vector<vec_pair_t>* item_feats) const;

 private:
  bool Init(const LocalDeepOpResource* resource) override {
    feature_ = NewItemFeature(resource->deep_data());
    return feature_ != nullptr;
  }
};

}  // namespace deep_op
}  // namespace embedx
