// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)

#pragma once
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/deep_config.h"
#include "src/io/loader/freq_file_loader.h"
#include "src/io/loader/inst_file_loader.h"
#include "src/io/loader/loader.h"

namespace embedx {

class DeepData {
 private:
  std::unique_ptr<Loader> item_feature_loader_;
  std::unique_ptr<InstFileLoader> inst_file_loader_;
  std::unique_ptr<FreqFileLoader> freq_file_loader_;

 public:
  static std::unique_ptr<DeepData> Create(const DeepConfig& config);

 public:
  const vec_pair_t* FindItemFeature(int_t item) const {
    return item_feature_loader_->storage()->FindNeighbor(item);
  }

  const vec_int_t& insts() const noexcept { return inst_file_loader_->insts(); }
  const std::vector<vecl_t>& vec_labels_list() const noexcept {
    return inst_file_loader_->vec_labels_list();
  }

  int ns_size() const noexcept { return freq_file_loader_->ns_size(); }
  const id_name_t& id_name_map() const noexcept {
    return freq_file_loader_->id_name_map();
  }
  const std::vector<vec_int_t>& nodes_list() const noexcept {
    return freq_file_loader_->nodes_list();
  }
  const std::vector<vec_float_t>& freqs_list() const noexcept {
    return freq_file_loader_->freqs_list();
  }

 private:
  bool Build(const DeepConfig& config);

 private:
  DeepData() = default;
};

}  // namespace embedx
