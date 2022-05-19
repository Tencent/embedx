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

#include <cinttypes>  // PRIu64
#include <memory>     // std::unique_ptr

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/sampler/negative_sampler.h"

namespace embedx {

class IndepNegativeSampler : public NegativeSampler {
 public:
  explicit IndepNegativeSampler(const SamplerBuilder* sampler_builder)
      : NegativeSampler(sampler_builder) {}
  ~IndepNegativeSampler() override = default;

 public:
  bool Sample(int count, const vec_int_t& nodes,
              const vec_int_t& excluded_nodes,
              std::vector<vec_int_t>* sampled_nodes_list) const override {
    sampled_nodes_list->clear();
    sampled_nodes_list->resize(nodes.size());

    const auto& sampler_source = sampler_builder_.sampler_source();
    const auto& id_name_map = sampler_source.id_name_map();
    // sample per node
    for (size_t i = 0; i < nodes.size(); ++i) {
      auto ns_id = io_util::GetNodeType(nodes[i]);
      if (id_name_map.find(ns_id) == id_name_map.end()) {
        DXERROR("Couldn't find node: %" PRIu64
                " namespace id: %d in the config file.",
                nodes[i], (int)ns_id);
        return false;
      }

      auto& uniq_nodes = sampler_source.nodes_list()[ns_id];
      auto& sampled_nodes = (*sampled_nodes_list)[i];
      if (!DoSampling(count, uniq_nodes, excluded_nodes, &sampled_nodes)) {
        return false;
      }
    }

    return true;
  }
};

std::unique_ptr<NegativeSampler> NewIndepNegativeSampler(
    const SamplerBuilder* sampler_builder) {
  std::unique_ptr<NegativeSampler> sampler;
  sampler.reset(new IndepNegativeSampler(sampler_builder));
  return sampler;
}

}  // namespace embedx
