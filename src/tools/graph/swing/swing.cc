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

#include "src/tools/graph/swing/swing.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::find_if, std::sort
#include <cmath>      // std::sqrt
#include <utility>    // std::move

#include "src/common/random.h"

namespace embedx {
namespace {

constexpr size_t MAX_NUM_PAIR = 50000000;  // magic number

int_t EncodeTwoEntry(int_t k1, int_t k2) {
  return (k1 + k2) * (k1 + k2 + 1) + k2;
}

void SampleContext(const std::vector<vec_pair_t>& contexts, int sample_thld,
                   std::vector<vec_pair_t>* sampled_contexts) {
  sampled_contexts->resize(contexts.size());
  for (size_t i = 0; i < contexts.size(); ++i) {
    DXCHECK(contexts[i].size() > 0);
    (*sampled_contexts)[i].clear();

    if ((int)contexts[i].size() <= sample_thld) {
      (*sampled_contexts)[i] = contexts[i];
    } else {
      float_t p = 1.0 * sample_thld / contexts[i].size();
      for (const auto& entry : contexts[i]) {
        if (ThreadLocalRandom() <= p) {
          (*sampled_contexts)[i].emplace_back(entry);
        }
      }
    }
  }
}

void GetContextUniqNode(const std::vector<vec_pair_t>& contexts,
                        vec_int_t* nodes) {
  set_int_t node_set;
  for (const auto& context : contexts) {
    for (const auto& entry : context) {
      node_set.emplace(entry.first);
    }
  }

  nodes->clear();
  nodes->assign(node_set.begin(), node_set.end());
}

void ContextRemoveIf(vec_pair_t* context, int_t key) {
  for (;;) {
    auto it =
        std::find_if(context->begin(), context->end(),
                     [key](const pair_t& pair) { return pair.first == key; });
    if (it != context->end()) {
      context->erase(it);
    } else {
      break;
    }
  }
}

void GetIntersection(const vec_pair_t& items1, const vec_pair_t& items2,
                     vec_int_t* inter_items) {
  set_int_t item_set1;
  for (const auto& item : items1) {
    item_set1.emplace(item.first);
  }
  set_int_t item_set2;
  for (const auto& item : items2) {
    item_set2.emplace(item.first);
  }

  inter_items->clear();
  if (item_set1.size() < item_set2.size()) {
    for (const auto& item : item_set1) {
      if (item_set2.count(item)) {
        inter_items->emplace_back(item);
      }
    }
  } else {
    for (const auto& item : item_set2) {
      if (item_set1.count(item)) {
        inter_items->emplace_back(item);
      }
    }
  }
}

}  // namespace

Swing::Swing(const GraphClient* graph_client, const SwingConfig& config)
    : graph_client_(*graph_client), config_(config) {
  cached_index_.clear();
  cached_index_.reserve(MAX_NUM_PAIR);
  cached_inter_items_.clear();
  cached_inter_items_.reserve(MAX_NUM_PAIR);
  cached_scores_.clear();
  cached_scores_.reserve(MAX_NUM_PAIR);
  index_ = 0;
}

bool Swing::ComputeItemScore(const vec_int_t& item_nodes,
                             const std::vector<vec_pair_t>& item_contexts,
                             std::vector<vec_pair_t>* item_scores) {
  DXASSERT(item_nodes.size() == item_contexts.size());

  // prepare user data
  std::vector<vec_pair_t> sampled_item_contexts;
  SampleContext(item_contexts, config_.sample_thld, &sampled_item_contexts);

  adj_list_t user_adj_list;
  if (!PrepareUserAdj(sampled_item_contexts, &user_adj_list)) {
    return false;
  }

  // compute swing
  item_scores->clear();
  for (size_t i = 0; i < sampled_item_contexts.size(); ++i) {
    vec_pair_t single_item_scores;
    ComputeSingleItemScore(item_nodes[i], sampled_item_contexts[i],
                           user_adj_list, &single_item_scores);
    item_scores->emplace_back(std::move(single_item_scores));
  }
  return true;
}

bool Swing::PrepareUserAdj(const std::vector<vec_pair_t>& item_contexts,
                           adj_list_t* user_adj_list) {
  vec_int_t user_nodes;
  set_int_t user_node_set;
  GetContextUniqNode(item_contexts, &user_nodes);
  for (size_t i = 0; i < user_nodes.size(); ++i) {
    for (size_t j = i + 1; j < user_nodes.size(); ++j) {
      if (!FindUserPair(user_nodes[i], user_nodes[j])) {
        user_node_set.emplace(user_nodes[i]);
        user_node_set.emplace(user_nodes[j]);
      }
    }
  }

  user_nodes.clear();
  user_nodes.assign(user_node_set.begin(), user_node_set.end());
  std::vector<vec_pair_t> user_contexts;
  if (!graph_client_.LookupContext(user_nodes, &user_contexts)) {
    return false;
  }

  user_adj_list->clear();
  for (size_t i = 0; i < user_nodes.size(); ++i) {
    user_adj_list->emplace(user_nodes[i], user_contexts[i]);
  }

  return !user_adj_list->empty();
}

void Swing::ComputeSingleItemScore(int_t item, const vec_pair_t& item_context,
                                   const adj_list_t& user_adj_list,
                                   vec_pair_t* item_scores) {
  std::vector<vec_int_t> vec_inter_items;
  vec_float_t scores;

  for (size_t i = 0; i < item_context.size(); ++i) {
    auto u = item_context[i].first;
    for (size_t j = i + 1; j < item_context.size(); ++j) {
      auto v = item_context[j].first;

      int_t key = u > v ? EncodeTwoEntry(u, v) : EncodeTwoEntry(v, u);
      auto it = cached_index_.find(key);
      if (it != cached_index_.end()) {
        // search
        auto index = it->second;
        auto& inter_items = cached_inter_items_[index];
        auto score = cached_scores_[index];
        vec_inter_items.emplace_back(inter_items);
        scores.emplace_back(score);
      } else {
        // get intersection
        auto context_u = user_adj_list.at(u);
        ContextRemoveIf(&context_u, item);
        auto w_u = 1.0 / std::sqrt(context_u.size());
        auto context_v = user_adj_list.at(v);
        ContextRemoveIf(&context_v, item);
        auto w_v = 1.0 / std::sqrt(context_v.size());
        vec_int_t inter_items;
        GetIntersection(context_u, context_v, &inter_items);
        if (inter_items.empty()) {
          continue;
        }
        float_t score = w_u * w_v * 1.0 / (config_.alpha + inter_items.size());
        vec_inter_items.emplace_back(inter_items);
        scores.emplace_back(score);

        // cache
        if (cached_index_.size() <= MAX_NUM_PAIR &&
            context_u.size() < (size_t)config_.cache_thld &&
            context_v.size() < (size_t)config_.cache_thld) {
          cached_index_.emplace(key, index_);
          cached_inter_items_.emplace_back(std::move(inter_items));
          cached_scores_.emplace_back(score);
          index_ += 1;
        }
      }
    }
  }

  AccumulateItemScore(vec_inter_items, scores, item_scores);
}

void Swing::AccumulateItemScore(const std::vector<vec_int_t>& vec_inter_items,
                                const vec_float_t& scores,
                                vec_pair_t* item_scores) {
  weight_map_t item_score_map;
  for (size_t i = 0; i < vec_inter_items.size(); ++i) {
    for (auto item : vec_inter_items[i]) {
      auto it = item_score_map.find(item);
      if (it == item_score_map.end()) {
        item_score_map.emplace(item, scores[i]);
      } else {
        it->second += scores[i];
      }
    }
  }

  item_scores->clear();
  item_scores->assign(item_score_map.begin(), item_score_map.end());
  std::sort(
      item_scores->begin(), item_scores->end(),
      [](const pair_t& a, const pair_t& b) { return a.second > b.second; });
}

bool Swing::FindUserPair(int_t u1, int_t u2) const {
  int_t key = u1 > u2 ? EncodeTwoEntry(u1, u2) : EncodeTwoEntry(u2, u1);
  return cached_index_.find(key) != cached_index_.end();
}

}  // namespace embedx
