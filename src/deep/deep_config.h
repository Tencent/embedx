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
#include <string>

namespace embedx {

class DeepConfig {
 private:
  std::string node_config_;
  std::string item_feat_;
  std::string inst_file_;
  std::string freq_file_;

  int negative_sampler_type_ = 0;
  int thread_num_ = 1;

 public:
  const std::string& node_config() const noexcept { return node_config_; }
  const std::string& item_feature() const noexcept { return item_feat_; }
  const std::string& inst_file() const noexcept { return inst_file_; }
  const std::string& freq_file() const noexcept { return freq_file_; }

  int negative_sampler_type() const noexcept { return negative_sampler_type_; }
  int thread_num() const noexcept { return thread_num_; }

 public:
  // data
  void set_node_config(const std::string& path) noexcept {
    node_config_ = path;
  }
  void set_item_feature(const std::string& path) noexcept { item_feat_ = path; }
  void set_inst_file(const std::string& path) noexcept { inst_file_ = path; }
  void set_freq_file(const std::string& path) noexcept { freq_file_ = path; }

  // type
  void set_negative_sampler_type(int type) noexcept {
    negative_sampler_type_ = type;
  }

  // performance
  void set_thread_num(int thread_num) noexcept { thread_num_ = thread_num; }
};

}  // namespace embedx
