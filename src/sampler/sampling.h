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

#include "src/common/data_types.h"

namespace embedx {

class Sampling {
 public:
  virtual ~Sampling() = default;

 public:
  virtual int_t Next() const noexcept = 0;
  virtual int_t Next(int begin, int end) const noexcept = 0;
};

enum class SamplingEnum : int {
  UNIFORM = 0,
  ALIAS = 1,
  WORD2VEC = 2,
  PARTIAL_SUM = 3
};

std::unique_ptr<Sampling> NewSampling(const vec_float_t* probs,
                                      SamplingEnum type);

}  // namespace embedx
