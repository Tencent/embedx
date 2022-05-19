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
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/data_op/deep_op.h"
#include "src/deep/data_op/deep_op_resource.h"

namespace embedx {
namespace deep_op {

class InstanceSampler : public LocalDeepOp {
 private:
  const vec_int_t* insts_ = nullptr;
  const std::vector<vecl_t>* vec_labels_list_ = nullptr;

 public:
  ~InstanceSampler() override = default;

 public:
  bool Run(int count, vec_int_t* insts,
           std::vector<vecl_t>* vec_labels_list) const;

 private:
  bool Init(const LocalDeepOpResource* resource) override {
    insts_ = &resource->deep_data()->insts();
    vec_labels_list_ = &resource->deep_data()->vec_labels_list();
    return insts_ != nullptr && vec_labels_list_ != nullptr;
  }
};

}  // namespace deep_op
}  // namespace embedx
