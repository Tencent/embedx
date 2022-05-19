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
#include <deepx_core/tensor/data_type.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>  // std::pair
#include <vector>

namespace embedx {

using int_t = ::deepx_core::DataType::int_t;
using float_t = ::deepx_core::DataType::float_t;
using csr_t = ::deepx_core::DataType::csr_t;
using tsr_t = ::deepx_core::DataType::tsr_t;
using pair_t = std::pair<int_t, float_t>;
using vec_int_t = std::vector<int_t>;
using vec_float_t = std::vector<float_t>;
using vecl_t = std::vector<int>;
using vec_str_t = std::vector<std::string>;
using vec_pair_t = std::vector<pair_t>;
using set_int_t = std::unordered_set<int_t>;

using vec_set_t = std::vector<set_int_t>;
using vec_map_neigh_t = std::vector<std::unordered_map<int_t, vec_int_t>>;

using id_name_t = std::unordered_map<uint16_t, std::string>;
using adj_list_t = std::unordered_map<int_t, vec_pair_t>;
using degree_list_t = std::unordered_map<int_t, std::pair<int_t, int_t>>;

using index_map_t = std::unordered_map<int_t, int>;
using vec_index_map_t = std::vector<std::unordered_map<int_t, int>>;
using weight_map_t = std::unordered_map<int_t, float_t>;

}  // namespace embedx
