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
#include <utility>  // std::pair

#include "src/common/data_types.h"

namespace embedx {
namespace random_walker_util {

bool FindRange(const vec_pair_t& context, uint16_t node_type,
               std::pair<int, int>* range);

bool ContainsNode(const vec_pair_t& context, int_t node);

}  // namespace random_walker_util
}  // namespace embedx
