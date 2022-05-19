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

#include "src/deep/data_op/feature_lookuper_op/item_feature.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

namespace embedx {
namespace deep_op {

bool ItemFeature::LookupFeature(const vec_int_t& items,
                                std::vector<vec_pair_t>* item_feats) const {
  item_feats->clear();
  for (auto item : items) {
    const auto* feat = deep_data_.FindItemFeature(item);
    if (feat == nullptr) {
      DXERROR("Couldn't find item: %." PRIu64, item);
      return false;
    } else {
      item_feats->emplace_back(*feat);
    }
  }

  return !item_feats->empty();
}

std::unique_ptr<ItemFeature> NewItemFeature(const DeepData* deep_data) {
  std::unique_ptr<ItemFeature> feature;
  feature.reset(new ItemFeature(deep_data));
  return feature;
}

}  // namespace deep_op
}  // namespace embedx
