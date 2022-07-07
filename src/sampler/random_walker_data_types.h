// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//

#pragma once
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

using meta_path_t = std::vector<uint16_t>;

struct WalkerConfig {
  int thread_num = 1;

  // for dynamic random walker
  double dynamic_p = 1;
  double dynamic_q = 1;
};

struct PrevInfo {
  vec_int_t nodes;
  std::vector<vec_pair_t> contexts;
};

struct WalkerInfo {
  // metapath
  meta_path_t meta_path;
  int walker_length;

  // dynamic
  PrevInfo prev_info;
};

}  // namespace embedx
