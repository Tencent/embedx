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
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/deep_config.h"
#include "src/deep/deep_data.h"

namespace embedx {
namespace deep_op {

class ItemFeature {
 private:
  const DeepData& deep_data_;

 public:
  explicit ItemFeature(const DeepData* deep_data) : deep_data_(*deep_data) {}

 public:
  bool LookupFeature(const vec_int_t& nodes,
                     std::vector<vec_pair_t>* item_feats) const;
};

std::unique_ptr<ItemFeature> NewItemFeature(const DeepData* deep_data);

}  // namespace deep_op
}  // namespace embedx
