// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)

#include "src/deep/data_op/instance_sampler_op/instance_sampler_op.h"

#include "src/common/random.h"
#include "src/deep/data_op/deep_op_registry.h"

namespace embedx {
namespace deep_op {

bool InstanceSampler::Run(int count, vec_int_t* insts,
                          std::vector<vecl_t>* vec_labels_list) const {
  insts->clear();
  vec_labels_list->clear();

  // do random sampling
  while ((int)insts->size() < count) {
    auto k = int(ThreadLocalRandom() * insts_->size());
    insts->emplace_back((*insts_)[k]);
    vec_labels_list->emplace_back((*vec_labels_list_)[k]);
  }

  return (int)insts->size() == count && (int)vec_labels_list->size() == count;
}

REGISTER_LOCAL_DEEP_OP("InstanceSampler", InstanceSampler);

}  // namespace deep_op
}  // namespace embedx
