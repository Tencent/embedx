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

#include <algorithm>  //std::find
#include <cinttypes>  // PRIu64
#include <memory>     // std::unique_ptr
#include <string>
#include <unordered_set>
#include <vector>

#include "src/common/data_types.h"
#include "src/io/io_util.h"
#include "src/io/loader/loader.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class MockSamplerSource : public SamplerSource {
 private:
  static constexpr uint64_t ESTIMATED_SIZE = 1000000;  // magic number

 private:
  std::unique_ptr<Loader> context_loader_;
  uint16_t ns_size_ = 1;
  id_name_t id_name_map_;
  std::vector<vec_int_t> uniq_nodes_list_;
  std::vector<vec_float_t> uniq_freqs_list_;

  std::unordered_set<int_t> node_set_;
  std::vector<weight_map_t> node_weight_maps_;

 public:
  static std::unique_ptr<SamplerSource> Create(const std::string& node_graph,
                                               const std::string& node_config,
                                               int thread_num);

 public:
  int ns_size() const noexcept override { return ns_size_; }
  const id_name_t& id_name_map() const noexcept override {
    return id_name_map_;
  }
  const std::vector<vec_int_t>& nodes_list() const noexcept override {
    return uniq_nodes_list_;
  }
  const std::vector<vec_float_t>& freqs_list() const noexcept override {
    return uniq_freqs_list_;
  }
  const vec_int_t& node_keys() const noexcept override {
    return context_loader_->storage()->Keys();
  }
  const vec_pair_t* FindContext(int_t node) const override {
    return context_loader_->storage()->FindNeighbor(node);
  }

 private:
  void Clear();
  bool Insert(int_t node);
  void AccumulateFreqs();

 private:
  MockSamplerSource() = default;
  bool Init(const std::string& node_graph, const std::string& node_config,
            int thread_num);
};

std::unique_ptr<SamplerSource> MockSamplerSource::Create(
    const std::string& node_graph, const std::string& node_config,
    int thread_num) {
  std::unique_ptr<SamplerSource> sampler_source;
  sampler_source.reset(new MockSamplerSource);
  if (!dynamic_cast<MockSamplerSource*>(sampler_source.get())
           ->Init(node_graph, node_config, thread_num)) {
    DXERROR("Failed to create MockSamplerSource.");
    sampler_source.reset();
  }
  return sampler_source;
}

void MockSamplerSource::Clear() {
  context_loader_.reset();
  ns_size_ = 1;
  id_name_map_.clear();
  uniq_nodes_list_.clear();
  uniq_freqs_list_.clear();
  node_set_.clear();
  node_weight_maps_.clear();
}

bool MockSamplerSource::Insert(int_t node) {
  auto ns_id = io_util::GetNodeType(node);
  if (id_name_map_.find(ns_id) == id_name_map_.end()) {
    DXERROR("Couldn't find node: %" PRIu64
            " namespace id: %d in the config file.",
            node, (int)ns_id);
    return false;
  }

  if (node_set_.find(node) == node_set_.end()) {
    node_set_.insert(node);
    uniq_nodes_list_[ns_id].emplace_back(node);
  }
  auto& node_weight_map = node_weight_maps_[ns_id];
  auto it = node_weight_map.find(node);
  if (it == node_weight_map.end()) {
    node_weight_map.emplace(node, 1.0);
  } else {
    it->second += 1.0;
  }
  return true;
}

void MockSamplerSource::AccumulateFreqs() {
  for (size_t i = 0; i < uniq_freqs_list_.size(); ++i) {
    const auto& nodes = uniq_nodes_list_[i];
    auto& freqs = uniq_freqs_list_[i];
    freqs.reserve(nodes.size());

    for (auto node : nodes) {
      auto ns_id = io_util::GetNodeType(node);
      auto it = node_weight_maps_[ns_id].find(node);
      DXCHECK(it != node_weight_maps_[ns_id].end());
      freqs.emplace_back(it->second);
    }
  }
}

bool MockSamplerSource::Init(const std::string& node_graph,
                             const std::string& node_config, int thread_num) {
  Clear();

  context_loader_ = NewContextLoader(1, 0, 0);
  context_loader_->Reserve(ESTIMATED_SIZE);
  if (!context_loader_->Load(node_graph, thread_num)) {
    return false;
  }

  if (!io_util::LoadConfig(node_config, &ns_size_, &id_name_map_)) {
    return false;
  }

  uniq_nodes_list_.resize(ns_size_);
  uniq_freqs_list_.resize(ns_size_);
  node_set_.reserve(ESTIMATED_SIZE);
  node_weight_maps_.resize(ns_size_);
  for (auto node : context_loader_->storage()->Keys()) {
    if (!Insert(node)) {
      return false;
    }
    const auto* context = context_loader_->storage()->FindNeighbor(node);
    if (context == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", node);
      return false;
    }
    for (auto& entry : *context) {
      if (!Insert(entry.first)) {
        return false;
      }
    }
  }

  AccumulateFreqs();
  return true;
}

std::unique_ptr<SamplerSource> NewMockSamplerSource(
    const std::string& node_graph, const std::string& node_config,
    int thread_num) {
  return MockSamplerSource::Create(node_graph, node_config, thread_num);
}

}  // namespace embedx
