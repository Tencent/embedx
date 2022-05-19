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
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/deep_config.h"

namespace embedx {

class DeepClientImpl;

class DeepClient {
 private:
  std::unique_ptr<DeepClientImpl> impl_;

 public:
  explicit DeepClient(std::unique_ptr<DeepClientImpl>&& impl);
  ~DeepClient();

 public:
  // negative sampler
  bool SharedSampleNegative(int count, const vec_int_t& nodes,
                            const vec_int_t& excluded_nodes,
                            std::vector<vec_int_t>* sampled_nodes_list) const;

  // feature
  bool LookupItemFeature(const vec_int_t& items,
                         std::vector<vec_pair_t>* item_feats) const;

  // instance sampler
  bool SampleInstance(int count, vec_int_t* insts,
                      std::vector<vecl_t>* vec_labels_list) const;
};

enum class DeepClientEnum : int { LOCAL = 0 };

std::unique_ptr<DeepClient> NewDeepClient(const DeepConfig& config,
                                          DeepClientEnum type);

}  // namespace embedx
