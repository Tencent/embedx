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

#include <vector>

#include "src/common/data_types.h"
#include "src/graph/in_memory_graph.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class GraphSamplerSource : public SamplerSource {
 private:
  const InMemoryGraph& graph_;

 public:
  explicit GraphSamplerSource(const InMemoryGraph* graph) : graph_(*graph) {}
  ~GraphSamplerSource() override = default;

 public:
  int ns_size() const noexcept override { return graph_.ns_size(); }
  const id_name_t& id_name_map() const noexcept override {
    return graph_.id_name_map();
  }
  const std::vector<vec_int_t>& nodes_list() const noexcept override {
    return graph_.uniq_nodes_list();
  }
  const std::vector<vec_float_t>& freqs_list() const noexcept override {
    return graph_.uniq_freqs_list();
  }
  const vec_int_t& node_keys() const noexcept override {
    return graph_.node_keys();
  }
  const vec_pair_t* FindContext(int_t node) const override {
    return graph_.FindContext(node);
  }
};

std::unique_ptr<SamplerSource> NewGraphSamplerSource(
    const InMemoryGraph* graph) {
  std::unique_ptr<SamplerSource> sampler_source(new GraphSamplerSource(graph));
  return sampler_source;
}

}  // namespace embedx
