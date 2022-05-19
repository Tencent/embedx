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
#include <mutex>
#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

class InstFileLoader {
 private:
  std::mutex mtx_;
  vec_int_t insts_;
  std::vector<vecl_t> vec_labels_list_;

 public:
  const vec_int_t& insts() const noexcept { return insts_; }
  const std::vector<vecl_t>& vec_labels_list() const noexcept {
    return vec_labels_list_;
  }

 private:
  bool Load(const std::string& dir, int thread_num);
  bool LoadEntry(const vec_str_t& files, int thread_id);

 private:
  InstFileLoader() = default;

 public:
  static std::unique_ptr<InstFileLoader> Create(const std::string& dir,
                                                int thread_num);
};

}  // namespace embedx
