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

#include <deepx_core/dx_log.h>

#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"
#include "src/common/random.h"
#include "src/sampler/sampling.h"

namespace embedx {

class UniformSampling : public Sampling {
 private:
  int table_size_ = 0;

 public:
  static std::unique_ptr<Sampling> Create(const vec_float_t* probs);

 public:
  int_t Next() const noexcept override;
  int_t Next(int begin, int end) const noexcept override;

 private:
  bool Init(const vec_float_t& probs);
};

std::unique_ptr<Sampling> UniformSampling::Create(const vec_float_t* probs) {
  std::unique_ptr<Sampling> sampling(new UniformSampling);
  if (!dynamic_cast<UniformSampling*>(sampling.get())->Init(*probs)) {
    DXERROR("Failed to init uniform sampling.");
    sampling.reset();
  }
  return sampling;
}

int_t UniformSampling::Next() const noexcept { return Next(0, table_size_); }

int_t UniformSampling::Next(int begin, int end) const noexcept {
  return (int_t)(ThreadLocalRandom() * (end - begin)) + begin;
}

bool UniformSampling::Init(const vec_float_t& probs) {
  table_size_ = (int)probs.size();
  return table_size_ != 0;
}

std::unique_ptr<Sampling> NewUniformSampling(const vec_float_t* probs) {
  return UniformSampling::Create(probs);
}

}  // namespace embedx
