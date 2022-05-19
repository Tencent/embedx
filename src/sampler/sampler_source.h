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

namespace embedx {

class InMemoryGraph;
class DeepData;

class SamplerSource {
 public:
  virtual ~SamplerSource() = default;

 public:
  virtual int ns_size() const noexcept = 0;
  virtual const id_name_t& id_name_map() const noexcept = 0;
  virtual const std::vector<vec_int_t>& nodes_list() const noexcept = 0;
  virtual const std::vector<vec_float_t>& freqs_list() const noexcept = 0;
  virtual const vec_int_t& node_keys() const noexcept = 0;
  virtual const vec_pair_t* FindContext(int_t node) const = 0;
};

std::unique_ptr<SamplerSource> NewGraphSamplerSource(
    const InMemoryGraph* graph);

std::unique_ptr<SamplerSource> NewDeepSamplerSource(const DeepData* deep_data);

// for test
std::unique_ptr<SamplerSource> NewMockSamplerSource(
    const std::string& node_graph, const std::string& node_config,
    int thread_num);

}  // namespace embedx
