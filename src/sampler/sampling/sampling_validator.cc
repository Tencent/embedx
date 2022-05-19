// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yong Zhou (zhouyongnju@gmail.com)
//

#include "src/sampler/sampling/sampling_validator.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::find_if, std::min
#include <cinttypes>  // PRIu64
#include <cmath>      // std::sqrt

#include "src/common/random.h"

namespace embedx {
namespace {

// 97.5% quantile ->  1.96
// 99.5% quantile ->  2.575
// 99.75% quantile -> 2.81
constexpr float_t QUANTILE = 2.81;

//  The MAX_FAILURE_TIME and NUM_TO_TEST should be carefully adapted, which
//  will affect the false positive rate.
constexpr int MAX_FAILURE_TIME = 1;
constexpr int ROUND_TO_VALID = 5;

bool CheckDistribution(const vec_pair_t& distribution) {
  set_int_t nodes;
  for (const auto& pair : distribution) {
    if (nodes.count(pair.first)) {
      DXERROR("Duplicate node: %" PRIu64 " in distribution.", pair.first);
      return false;
    } else {
      nodes.emplace(pair.first);
    }
  }

  return !nodes.empty();
}

void DrawKHypothesis(int k, const vec_pair_t& ground_truth_distribution,
                     vec_pair_t* distribution_to_validate) {
  distribution_to_validate->clear();

  while (k--) {
    auto idx = int(ThreadLocalRandom() * ground_truth_distribution.size());
    auto next = ground_truth_distribution[idx];
    auto it = std::find_if(
        distribution_to_validate->begin(), distribution_to_validate->end(),
        [next](const pair_t& pair) { return pair.first == next.first; });
    if (it == distribution_to_validate->end()) {
      distribution_to_validate->emplace_back(next);
    }
  }
}

void CountNodes(const vec_int_t& nodes, const vec_pair_t& distribution,
                index_map_t* node_count_map) {
  node_count_map->clear();
  node_count_map->reserve(distribution.size());
  for (const auto& pair : distribution) {
    node_count_map->emplace(pair.first, 0);
  }

  for (auto node : nodes) {
    auto it = node_count_map->find(node);
    if (it != node_count_map->end()) {
      it->second += 1;
    }
  }
}

// Let Phi(Z_a) be the CDF of the variable x. Find the smallest Z_{a/2}, such
// that Phi(Z_{a/2}) >= 1 - a/2.
// mean : N * prob
// confidence interval : [N * prob - Z_{a/2} * sqrt(N * prob * (1-prob)),
// N * prob + Z_{a/2} * sqrt(N * prob * (1-prob))]
bool HypothesisTesting(int trails, const vec_pair_t& distribution_to_validate,
                       const index_map_t& node_count_map) {
  int_t count = 0;
  float_t prob = 0;
  float_t mean = 0;
  float_t interval = 0;
  int failure_time = 0;
  for (const auto& entry : distribution_to_validate) {
    auto it = node_count_map.find(entry.first);
    if (it == node_count_map.end()) {
      count = 0;
    } else {
      count = it->second;
    }
    prob = entry.second;
    mean = trails * prob;
    interval = QUANTILE * std::sqrt(trails * prob * (1 - prob));

    if (count < mean - interval || count > mean + interval) {
      DXINFO("The sampling result of node: %" PRIu64 "(%" PRIu64
             ") is out of the expect range [%f,%f]",
             entry.first, count, mean - interval, mean + interval);
      failure_time += 1;
    }
  }

  return failure_time <= MAX_FAILURE_TIME;
}

}  // namespace

bool SamplingValidator::Test(const vec_pair_t& ground_truth_distribution,
                             const vec_int_t& sampled_nodes) {
  if (!CheckDistribution(ground_truth_distribution)) {
    return false;
  }

  int k = std::min(ROUND_TO_VALID, (int)ground_truth_distribution.size());

  vec_pair_t distribution_to_validate;
  DrawKHypothesis(k, ground_truth_distribution, &distribution_to_validate);

  index_map_t node_count_map;
  CountNodes(sampled_nodes, distribution_to_validate, &node_count_map);

  return HypothesisTesting(sampled_nodes.size(), distribution_to_validate,
                           node_count_map);
}

}  // namespace embedx
