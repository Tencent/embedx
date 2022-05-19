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

#include <deepx_core/dx_log.h>

#include <memory>  // std::unique_ptr

#include "src/common/data_types.h"
#include "src/common/random.h"
#include "src/sampler/sampling.h"

namespace embedx {

class AliasSampling : public Sampling {
 private:
  vec_float_t alias_probs_;
  vec_int_t alias_tables_;

 public:
  static std::unique_ptr<Sampling> Create(const vec_float_t& probs);

 public:
  int_t Next() const noexcept override;
  int_t Next(int begin, int end) const noexcept override;

 private:
  // Always return true.
  bool Init(const vec_float_t& probs);

  void Clear() noexcept {
    alias_probs_.clear();
    alias_tables_.clear();
  }

  void Resize(size_t size) {
    alias_probs_.resize(size, 0);
    alias_tables_.resize(size, 0);
  }
};

std::unique_ptr<Sampling> AliasSampling::Create(const vec_float_t& probs) {
  std::unique_ptr<Sampling> sampling(new AliasSampling);
  if (!dynamic_cast<AliasSampling*>(sampling.get())->Init(probs)) {
    DXERROR("Failed to init alias sampling.");
    sampling.reset();
  }
  return sampling;
}

int_t AliasSampling::Next() const noexcept {
  size_t table_size = alias_probs_.size();
  auto k = int_t(ThreadLocalRandom() * table_size);
  if (ThreadLocalRandom() < alias_probs_[k]) {
    return k;
  } else {
    return alias_tables_[k];
  }
}

int_t AliasSampling::Next(int /*begin*/, int /*end*/) const noexcept {
  DXERROR("Next with range was not implemented in AliasSampling.");
  return 0;
}

bool AliasSampling::Init(const vec_float_t& probs) {
  size_t table_size = probs.size();

  Clear();
  Resize(table_size);

  vec_int_t smaller;
  vec_int_t larger;
  for (size_t i = 0; i < probs.size(); ++i) {
    alias_probs_[i] = table_size * probs[i];
    if (alias_probs_[i] < 1.0) {
      smaller.emplace_back(i);
    } else {
      larger.emplace_back(i);
    }
  }

  while (smaller.size() > 0 && larger.size() > 0) {
    const auto s = smaller.back();
    smaller.pop_back();
    const auto l = larger.back();
    larger.pop_back();

    alias_tables_[s] = l;
    alias_probs_[l] += alias_probs_[s] - (float_t)1.0;
    if (alias_probs_[l] < 1.0) {
      smaller.emplace_back(l);
    } else {
      larger.emplace_back(l);
    }
  }
  return true;
}

std::unique_ptr<Sampling> NewAliasSampling(const vec_float_t* probs) {
  return AliasSampling::Create(*probs);
}

}  // namespace embedx
