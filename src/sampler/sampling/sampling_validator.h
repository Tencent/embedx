// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yong Zhou (zhouyongnju@gmail.com)
//

#pragma once
#include "src/common/data_types.h"

namespace embedx {

class SamplingValidator {
 public:
  static bool Test(const vec_pair_t& ground_truth_distribution,
                   const vec_int_t& sampled_result);
};

}  // namespace embedx
