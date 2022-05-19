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

#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/sampler/negative_sampler.h"

namespace embedx {

class SharedNegativeSampler : public NegativeSampler {
 public:
  explicit SharedNegativeSampler(const SamplerBuilder* sampler_builder)
      : NegativeSampler(sampler_builder) {}
  ~SharedNegativeSampler() override = default;

 public:
  bool Sample(int count, const vec_int_t& nodes,
              const vec_int_t& excluded_nodes,
              std::vector<vec_int_t>* sampled_nodes_list) const override {
    const auto& sampler_source = sampler_builder_.sampler_source();
    std::unordered_set<uint16_t> ns_id_set;
    io_util::ParseMaxNodeType(sampler_source.ns_size(), nodes, &ns_id_set);
    sampled_nodes_list->clear();
    sampled_nodes_list->resize(sampler_source.ns_size());

    const auto& id_name_map = sampler_source.id_name_map();
    // sample per namespace
    for (auto ns_id : ns_id_set) {
      if (id_name_map.find(ns_id) == id_name_map.end()) {
        DXERROR("Invalid ns_id: %d !", (int)ns_id);
        return false;
      }

      auto& uniq_nodes = sampler_source.nodes_list()[ns_id];
      auto& sampled_nodes = (*sampled_nodes_list)[ns_id];
      if (!DoSampling(count, uniq_nodes, excluded_nodes, &sampled_nodes)) {
        return false;
      }
    }

    return true;
  }
};

std::unique_ptr<NegativeSampler> NewSharedNegativeSampler(
    const SamplerBuilder* sampler_builder) {
  std::unique_ptr<NegativeSampler> sampler;
  sampler.reset(new SharedNegativeSampler(sampler_builder));
  return sampler;
}

}  // namespace embedx
