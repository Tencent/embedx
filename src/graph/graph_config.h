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
#include <string>

#include "src/common/data_types.h"

namespace embedx {

class GraphConfig {
 private:
  static constexpr uint64_t ESTIMATED_SIZE = 1000000;  // magic number

 private:
  std::string node_graph_;
  std::string node_config_;
  std::string node_feat_;
  std::string neigh_feat_;
  int store_type_ = 0;

  int negative_sampler_type_ = 0;
  int neighbor_sampler_type_ = 0;
  int random_walker_type_ = 0;

  int shard_num_ = 1;
  int shard_id_ = 0;

  int thread_num_ = 1;
  std::string ip_ports_;

  int cache_type_ = 0;
  double cache_thld_ = 0.0;
  int max_node_per_rpc_ = 2000;

  std::string success_out_;

 public:
  // data
  const std::string& node_graph() const noexcept { return node_graph_; }
  const std::string& node_config() const noexcept { return node_config_; }
  const std::string& node_feature() const noexcept { return node_feat_; }
  const std::string& neighbor_feature() const noexcept { return neigh_feat_; }
  int store_type() const noexcept { return store_type_; }

  // sampler type
  int negative_sampler_type() const noexcept { return negative_sampler_type_; }
  int neighbor_sampler_type() const noexcept { return neighbor_sampler_type_; }
  int random_walker_type() const noexcept { return random_walker_type_; }

  // dist
  int shard_num() const noexcept { return shard_num_; }
  int shard_id() const noexcept { return shard_id_; }

  // performance
  int thread_num() const noexcept { return thread_num_; }
  const std::string& ip_ports() const noexcept { return ip_ports_; }
  uint64_t estimated_size() const noexcept { return ESTIMATED_SIZE; }

  // cache
  int cache_type() const noexcept { return cache_type_; }
  double cache_thld() const noexcept { return cache_thld_; }
  int max_node_per_rpc() const noexcept { return max_node_per_rpc_; }

  // output
  const std::string& success_out() const noexcept { return success_out_; }

 public:
  // data
  void set_node_graph(const std::string& path) noexcept { node_graph_ = path; }
  void set_node_config(const std::string& path) noexcept {
    node_config_ = path;
  }
  void set_node_feature(const std::string& path) noexcept { node_feat_ = path; }
  void set_neighbor_feature(const std::string& path) noexcept {
    neigh_feat_ = path;
  }
  void set_store_type(int store_type) noexcept { store_type_ = store_type; }

  // sampler type
  void set_negative_sampler_type(int type) noexcept {
    negative_sampler_type_ = type;
  }
  void set_neighbor_sampler_type(int type) noexcept {
    neighbor_sampler_type_ = type;
  }
  void set_random_walker_type(int type) noexcept { random_walker_type_ = type; }

  // dist
  void set_shard_num(int shard_num) noexcept { shard_num_ = shard_num; }
  void set_shard_id(int shard_id) noexcept { shard_id_ = shard_id; }

  // performance
  void set_thread_num(int thread_num) noexcept { thread_num_ = thread_num; }
  void set_ip_ports(const std::string& ip_ports) noexcept {
    ip_ports_ = ip_ports;
  }

  // cache
  void set_cache_type(int cache_type) noexcept { cache_type_ = cache_type; }
  void set_cache_thld(double cache_thld) noexcept { cache_thld_ = cache_thld; }
  void set_max_node_per_rpc(int max_node_per_rpc) noexcept {
    max_node_per_rpc_ = max_node_per_rpc;
  }

  // output
  void set_success_out(const std::string& success_out) noexcept {
    success_out_ = success_out;
  }
};

}  // namespace embedx
