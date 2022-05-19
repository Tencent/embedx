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
#include <deepx_core/ps/rpc_client.h>

#include <memory>   // std::unique_ptr
#include <utility>  // std::move

#include "src/graph/cache/cache_storage.h"
#include "src/graph/client/rpc_connector.h"
#include "src/graph/graph_config.h"
#include "src/graph/in_memory_graph.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"
#include "src/sampler/sampling.h"

namespace embedx {
namespace graph_op {

class LocalGSOpResource {
 private:
  GraphConfig graph_config_;
  std::unique_ptr<InMemoryGraph> graph_;
  std::unique_ptr<SamplerSource> sampler_source_;
  std::unique_ptr<SamplerBuilder> negative_sampler_builder_;
  std::unique_ptr<SamplerBuilder> neighbor_sampler_builder_;

 public:
  const GraphConfig& graph_config() const noexcept { return graph_config_; }
  const InMemoryGraph* graph() const noexcept { return graph_.get(); }
  const SamplerSource* sampler_source() const noexcept {
    return sampler_source_.get();
  }
  const SamplerBuilder* negative_sampler_builder() const noexcept {
    return negative_sampler_builder_.get();
  }
  const SamplerBuilder* neighbor_sampler_builder() const noexcept {
    return neighbor_sampler_builder_.get();
  }

 public:
  void set_graph_config(const GraphConfig& graph_config) {
    graph_config_ = graph_config;
  }
  void set_graph(std::unique_ptr<InMemoryGraph> graph) {
    graph_ = std::move(graph);
  }
  void set_sampler_source(std::unique_ptr<SamplerSource> sampler_source) {
    sampler_source_ = std::move(sampler_source);
  }
  void set_negative_sampler_builder(
      std::unique_ptr<SamplerBuilder> sampler_builder) {
    negative_sampler_builder_ = std::move(sampler_builder);
  }
  void set_neighbor_sampler_builder(
      std::unique_ptr<SamplerBuilder> sampler_builder) {
    neighbor_sampler_builder_ = std::move(sampler_builder);
  }
};

class DistGSOpResource {
 private:
  mutable std::unique_ptr<RpcConnector> rpc_connector_;
  int ns_size_ = 1;
  std::unique_ptr<Sampling> sampling_;
  std::unique_ptr<CacheStorage> cache_storage_;

 public:
  ~DistGSOpResource() { rpc_connector_->Close(); }

 public:
  RpcConnector* rpc_connector() const noexcept { return rpc_connector_.get(); }
  int ns_size() const noexcept { return ns_size_; }
  const Sampling* sampling() const noexcept { return sampling_.get(); }
  const CacheStorage* cache_storage() const noexcept {
    return cache_storage_.get();
  }

 public:
  void set_rpc_connector(std::unique_ptr<RpcConnector> rpc_connector) noexcept {
    rpc_connector_ = std::move(rpc_connector);
  }
  void set_ns_size(int ns_size) noexcept { ns_size_ = ns_size; }
  void set_sampling(std::unique_ptr<Sampling> sampling) noexcept {
    sampling_ = std::move(sampling);
  }
  void set_cache_storage(std::unique_ptr<CacheStorage> cache_storage) noexcept {
    cache_storage_ = std::move(cache_storage);
  }
};

}  // namespace graph_op
}  // namespace embedx
