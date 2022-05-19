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

#include <algorithm>  // std::stable_sort
#include <cmath>
#include <memory>   // std::unique_ptr
#include <utility>  // std::pair, std::make_pair

#include "src/common/data_types.h"
#include "src/common/random.h"
#include "src/sampler/sampling.h"

namespace embedx {

class Word2vecSampling : public Sampling {
  class Table {
   private:
    static constexpr size_t MAX_TABLE_SIZE = 1000000000;
    // ratio of valid frequency's size
    static constexpr int TABLE_RATIO = 10;
    static constexpr float_t FREQUENCY_POWER = (float_t)0.75;

    size_t table_size_ = 0;
    vec_int_t sample_tables_;

   public:
    // Set id to sampled_tables_ by id's frequency,
    // the higher frequency, the higher probability to be choosen.
    bool Init(const vec_float_t& freqs) {
      DXINFO("Initing sampling table...");

      size_t freq_size = freqs.size();
      if (freq_size > MAX_TABLE_SIZE) {
        DXERROR("The freq_size: %zu must be less than or equal to %zu.",
                freq_size, MAX_TABLE_SIZE);
        return false;
      }

      vec_pair_t valid_index_freqs;
      float_t acc = 0;
      for (size_t i = 0; i < freq_size; ++i) {
        if (freqs[i] != 0) {
          acc += std::pow(freqs[i], FREQUENCY_POWER);
          valid_index_freqs.emplace_back(std::make_pair((id_t)i, freqs[i]));
        }
      }

      std::stable_sort(
          valid_index_freqs.begin(), valid_index_freqs.end(),
          [](const pair_t& a, const pair_t& b) { return a.second > b.second; });

      table_size_ = valid_index_freqs.size() * TABLE_RATIO;
      if (table_size_ > MAX_TABLE_SIZE) {
        table_size_ = MAX_TABLE_SIZE;
      }
      if (table_size_ <= 1u) {
        DXERROR("The table_size_: %zu must be greater than 1.", table_size_);
        return false;
      }
      sample_tables_.resize(table_size_);

      auto iter = valid_index_freqs.begin();
      float_t p = std::pow(iter->second, FREQUENCY_POWER) / acc;
      for (size_t i = 0; i < table_size_; ++i) {
        sample_tables_[i] = iter->first;
        if (((float_t)i / table_size_) > p) {
          if (++iter == valid_index_freqs.end()) {
            --iter;
          }
          p += std::pow(iter->second, FREQUENCY_POWER) / acc;
        }
      }

      DXINFO("Done.");
      return true;
    }

    void clear() noexcept {
      table_size_ = 0;
      sample_tables_.clear();
    }

    int_t Next() const noexcept {
      auto k = (int_t)(ThreadLocalRandom() * table_size_);
      return sample_tables_[k];
    }
  };

  Table table_;

 public:
  static std::unique_ptr<Sampling> Create(const vec_float_t& probs);

 public:
  int_t Next() const noexcept override;
  int_t Next(int begin, int end) const noexcept override;

 private:
  void Clear() noexcept { table_.clear(); }

  bool Init(const vec_float_t& freqs);
};

std::unique_ptr<Sampling> Word2vecSampling::Create(const vec_float_t& probs) {
  std::unique_ptr<Sampling> sampling(new Word2vecSampling);
  if (!dynamic_cast<Word2vecSampling*>(sampling.get())->Init(probs)) {
    DXERROR("Failed to init word2vec sampling.");
    sampling.reset();
  }
  return sampling;
}

int_t Word2vecSampling::Next() const noexcept { return table_.Next(); }

int_t Word2vecSampling::Next(int /*begin*/, int /*end*/) const noexcept {
  DXERROR("Next with range was not implemented in Word2vecSampling.");
  return 0;
}

bool Word2vecSampling::Init(const vec_float_t& freqs) {
  if (freqs.empty()) {
    DXERROR("Frequency tables are empty!");
    return false;
  }
  Clear();
  return table_.Init(freqs);
}

std::unique_ptr<Sampling> NewWord2vecSampling(const vec_float_t* probs) {
  return Word2vecSampling::Create(*probs);
}

}  // namespace embedx
