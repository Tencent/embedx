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

class DeepClientImpl {
 public:
  virtual ~DeepClientImpl() = default;

 public:
  virtual bool Init(const DeepConfig& deep_config) = 0;

 public:
  // negative sampler
  virtual bool SharedSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const = 0;

  // feature
  virtual bool LookupItemFeature(const vec_int_t& items,
                                 std::vector<vec_pair_t>* item_feats) const = 0;
  // instance sampler
  virtual bool SampleInstance(int count, vec_int_t* insts,
                              std::vector<vecl_t>* vec_labels_list) const = 0;
};

std::unique_ptr<DeepClientImpl> NewLocalDeepClientImpl(
    const DeepConfig& deep_config);

}  // namespace embedx
