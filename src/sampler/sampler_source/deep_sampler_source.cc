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

#include <deepx_core/dx_log.h>

#include <memory>   // std::unique_ptr
#include <utility>  // std::move
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/deep_data.h"
#include "src/sampler/sampler_source.h"

namespace embedx {
namespace {

const vec_int_t EMPTY_NODE_KEYS = {0};

}  // namespace

class DeepSamplerSource : public SamplerSource {
 private:
  const DeepData& deep_data_;

 public:
  explicit DeepSamplerSource(const DeepData* deep_data)
      : deep_data_(*deep_data) {}
  ~DeepSamplerSource() override = default;

 public:
  int ns_size() const noexcept override { return deep_data_.ns_size(); }
  const id_name_t& id_name_map() const noexcept override {
    return deep_data_.id_name_map();
  }
  const std::vector<vec_int_t>& nodes_list() const noexcept override {
    return deep_data_.nodes_list();
  }
  const std::vector<vec_float_t>& freqs_list() const noexcept override {
    return deep_data_.freqs_list();
  }
  const vec_int_t& node_keys() const noexcept override {
    DXERROR("Node_keys was not implemented in DeepSamplerSource.");
    return EMPTY_NODE_KEYS;
  }
  const vec_pair_t* FindContext(int_t /*node*/) const override {
    DXERROR("Find_context was not implemented in DeepSamplerSource.");
    return nullptr;
  }
};

std::unique_ptr<SamplerSource> NewDeepSamplerSource(const DeepData* deep_data) {
  std::unique_ptr<SamplerSource> sampler_source(
      new DeepSamplerSource(deep_data));
  return sampler_source;
}

}  // namespace embedx
