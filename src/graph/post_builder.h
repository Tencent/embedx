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
#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/graph_builder.h"
#include "src/graph/graph_config.h"
#include "src/io/storage/storage.h"

namespace embedx {

class PostBuilder {
 private:
  uint16_t ns_size_ = 1;
  id_name_t id_name_map_;
  std::vector<vec_int_t> uniq_nodes_list_;
  std::vector<vec_float_t> uniq_freqs_list_;
  vec_int_t total_freqs_;

  const Storage* store_ = nullptr;
  uint64_t estimated_size_ = 1000000;  // magic number

 public:
  static std::unique_ptr<PostBuilder> Create(const Storage* store,
                                             const GraphConfig& config);

 public:
  uint16_t ns_size() const noexcept { return ns_size_; }
  const id_name_t& id_name_map() const noexcept { return id_name_map_; }

  const std::vector<vec_int_t>& uniq_nodes_list() const noexcept {
    return uniq_nodes_list_;
  }
  const std::vector<vec_float_t>& uniq_freqs_list() const noexcept {
    return uniq_freqs_list_;
  }
  const vec_int_t& total_freqs() const noexcept { return total_freqs_; }

 private:
  void set_store(const Storage* store) noexcept { store_ = store; }
  void set_estimated_size(uint64_t size) noexcept { estimated_size_ = size; }
  bool Build(const std::string& config, int thread_num);

 private:
  PostBuilder() = default;
};

}  // namespace embedx
