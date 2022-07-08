// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//

#pragma once
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"

namespace embedx {

struct SwingConfig {
  float_t alpha = 0;
  int cache_thld = 10;
  int sample_thld = 5000;
};

class Swing {
 private:
  const GraphClient& graph_client_;
  SwingConfig config_;

  // cache
  index_map_t cached_index_;
  std::vector<vec_int_t> cached_inter_items_;
  vec_float_t cached_scores_;
  int index_ = 0;

 public:
  Swing(const GraphClient* graph_client, const SwingConfig& config);

 public:
  // Large Scale Product Graph Construction for Recommendation in E-commerce
  // Xiaoyong Yang, Yadong Zhu etc.
  bool ComputeItemScore(const vec_int_t& item_nodes,
                        const std::vector<vec_pair_t>& item_contexts,
                        std::vector<vec_pair_t>* item_scores);

 private:
  bool PrepareUserAdj(const std::vector<vec_pair_t>& item_contexts,
                      adj_list_t* user_adj_list);
  void ComputeSingleItemScore(int_t item, const vec_pair_t& item_context,
                              const adj_list_t& user_adj_list,
                              vec_pair_t* item_scores);
  void AccumulateItemScore(const std::vector<vec_int_t>& vec_inter_items,
                           const vec_float_t& scores, vec_pair_t* item_scores);
  bool FindUserPair(int_t u1, int_t u2) const;
};

}  // namespace embedx
