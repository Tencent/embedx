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

#include <algorithm>  // std::lower_bound
#include <memory>     // std::unique_ptr

#include "src/common/data_types.h"
#include "src/common/random.h"
#include "src/sampler/sampling.h"

namespace embedx {

class PartialSumSampling : public Sampling {
 private:
  vec_float_t partial_sum_table_;

 public:
  static std::unique_ptr<Sampling> Create(const vec_float_t& probs);

 public:
  int_t Next() const noexcept override;
  int_t Next(int begin, int end) const noexcept override;

 private:
  bool Init(const vec_float_t& probs);
};

std::unique_ptr<Sampling> PartialSumSampling::Create(const vec_float_t& probs) {
  std::unique_ptr<Sampling> sampling(new PartialSumSampling);
  if (!dynamic_cast<PartialSumSampling*>(sampling.get())->Init(probs)) {
    DXERROR("Failed to init word2vec sampling.");
    sampling.reset();
  }
  return sampling;
}

int_t PartialSumSampling::Next() const noexcept {
  return Next(0, (int)partial_sum_table_.size());
}

int_t PartialSumSampling::Next(int begin, int end) const noexcept {
  float_t sum_begin = begin > 0 ? partial_sum_table_[begin - 1] : 0.;
  float_t sum_end = partial_sum_table_[end - 1];
  float_t random = ThreadLocalRandom() * (sum_end - sum_begin) + sum_begin;
  auto index = std::lower_bound(partial_sum_table_.begin() + begin,
                                partial_sum_table_.begin() + end, random) -
               partial_sum_table_.begin();
  return (int_t)index;
}

bool PartialSumSampling::Init(const vec_float_t& probs) {
  partial_sum_table_.clear();
  partial_sum_table_.resize(probs.size());
  float_t sum = 0;
  for (size_t i = 0; i < probs.size(); ++i) {
    sum += probs[i];
    partial_sum_table_[i] = sum;
  }
  return sum != 0;
}

std::unique_ptr<Sampling> NewPartialSumSampling(const vec_float_t* probs) {
  return PartialSumSampling::Create(*probs);
}

}  // namespace embedx
