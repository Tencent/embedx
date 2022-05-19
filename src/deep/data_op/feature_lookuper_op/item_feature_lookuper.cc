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

#include "src/deep/data_op/feature_lookuper_op/item_feature_lookuper.h"

#include <deepx_core/dx_log.h>

#include "src/deep/data_op/deep_op_registry.h"

namespace embedx {
namespace deep_op {

bool ItemFeatureLookuper::Run(const vec_int_t& items,
                              std::vector<vec_pair_t>* item_feats) const {
  if (!feature_->LookupFeature(items, item_feats)) {
    DXERROR("Failed to lookup item feature.");
    return false;
  }

  return true;
}

REGISTER_LOCAL_DEEP_OP("ItemFeatureLookuper", ItemFeatureLookuper);

}  // namespace deep_op
}  // namespace embedx
