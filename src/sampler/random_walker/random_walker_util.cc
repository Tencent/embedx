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

#include "src/sampler/random_walker/random_walker_util.h"

#include <algorithm>  // std::lower_bound, std::upper_bound
#include <utility>    // std::unique_ptr

#include "src/io/io_util.h"

namespace embedx {
namespace random_walker_util {

bool FindRange(const vec_pair_t& context, uint16_t node_type,
               std::pair<int, int>* range) {
  auto l = std::lower_bound(context.begin(), context.end(), node_type,
                            [](const pair_t& p, uint16_t node_type) {
                              return io_util::GetNodeType(p.first) < node_type;
                            });
  auto h = std::upper_bound(context.begin(), context.end(), node_type,
                            [](uint16_t node_type, const pair_t& p) {
                              return node_type < io_util::GetNodeType(p.first);
                            });
  if (l == h) {
    return false;
  } else {
    range->first = (int)(l - context.begin());
    range->second = (int)(h - context.begin());
    return true;
  }
}

bool ContainsNode(const vec_pair_t& context, int_t node) {
  uint16_t node_type = io_util::GetNodeType(node);
  std::pair<int, int> range;
  if (!FindRange(context, node_type, &range)) {
    return false;
  }
  int low = range.first;
  int high = range.second;
  int mid;
  while (low < high) {
    mid = low + (high - low) / 2;
    context[mid].first < node ? low = mid + 1 : high = mid;
  }
  return low < range.second && context[low].first == node;
}

}  // namespace random_walker_util
}  // namespace embedx
