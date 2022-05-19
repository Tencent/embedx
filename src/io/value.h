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

#pragma once
#include <sstream>  // std::stringstream
#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

struct NodeValue {
  int_t node;
  float_t weight = 1;  // optional

  std::string ToString() const {
    std::stringstream ss;
    ss << node << " " << weight;
    return ss.str();
  }
};

struct EdgeValue {
  int_t src_node;
  int_t dst_node;
  float_t weight = 1;  // optional

  std::string ToString() const {
    std::stringstream ss;
    ss << src_node << " " << dst_node << " " << weight;
    return ss.str();
  }
};

struct SeqValue {
  vec_int_t nodes;

  std::string ToString() const {
    std::stringstream ss;
    for (auto& v : nodes) {
      ss << " " << v;
    }
    return ss.str();
  }
};

struct AdjValue {
  int_t node;
  vec_pair_t pairs;

  std::string ToString() const {
    std::stringstream ss;
    ss << node;
    for (auto& pair : pairs) {
      ss << " " << pair.first << ":" << pair.second;
    }
    return ss.str();
  }
};

struct NodeAndLabelValue {
  int_t node;
  vecl_t labels;

  std::string ToString() const {
    std::stringstream ss;
    ss << node;
    for (auto& l : labels) {
      ss << " " << l;
    }
    return ss.str();
  }
};

struct EdgeAndLabelValue {
  int_t src_node;
  int_t dst_node;
  int_t node;
  vecl_t labels;

  std::string ToString() const {
    std::stringstream ss;
    ss << src_node << " " << dst_node << " " << node;
    for (auto& l : labels) {
      ss << " " << l;
    }
    return ss.str();
  }
};

template <typename ValueType, typename T>
std::vector<T> Collect(const std::vector<ValueType>& v, T ValueType::*field) {
  std::vector<T> output;
  for (auto& elem : v) {
    output.emplace_back(elem.*field);
  }
  return output;
}

}  // namespace embedx
