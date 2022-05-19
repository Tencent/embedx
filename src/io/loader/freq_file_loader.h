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
#include <mutex>
#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

class FreqFileLoader {
 private:
  uint16_t ns_size_;
  id_name_t id_name_map_;
  std::vector<vec_int_t> nodes_list_;
  std::vector<vec_float_t> freqs_list_;
  std::mutex mtx_;

 public:
  uint16_t ns_size() const noexcept { return ns_size_; }
  const id_name_t& id_name_map() const noexcept { return id_name_map_; }
  const std::vector<vec_int_t>& nodes_list() const noexcept {
    return nodes_list_;
  }
  const std::vector<vec_float_t>& freqs_list() const noexcept {
    return freqs_list_;
  }

 private:
  bool LoadConfig(const std::string& config_file);
  bool LoadFreq(const std::string& dir, int thread_num);
  bool LoadFreqEntry(const vec_str_t& freq_files, int thread_id);

 private:
  FreqFileLoader() = default;

 public:
  static std::unique_ptr<FreqFileLoader> Create(const std::string& config,
                                                const std::string& dir,
                                                int thread_num);
};

}  // namespace embedx
