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

#include "src/model/instance_reader_util.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::min, std::max

#include "src/io/io_util.h"

namespace embedx {
namespace inst_util {

void GenerateIthContext(int i, int left_window, int right_window,
                        const vec_int_t& seq, vec_int_t* context) {
  context->clear();

  int j, j_max;

  j = std::max(i - left_window, 0);
  j_max = std::min(i, (int)seq.size());
  // left side  of context window
  for (; j < j_max; ++j) {
    context->emplace_back(seq[j]);
  }

  // right side  of context window
  j = std::min(i + 1, (int)seq.size());
  j_max = std::min(i + 1 + right_window, (int)seq.size());
  for (; j < j_max; ++j) {
    context->emplace_back(seq[j]);
  }
}

void GenerateContext(int left, int right, const vec_int_t& seq,
                     std::vector<vec_int_t>* contexts_list) {
  contexts_list->clear();
  contexts_list->resize(seq.size());
  for (size_t i = 0; i < seq.size(); ++i) {
    GenerateIthContext(i, left, right, seq, &(*contexts_list)[i]);
  }
}

void CreateIndexing(const vec_int_t& nodes, Indexing* indexing) {
  indexing->Clear();
  for (auto& node : nodes) {
    indexing->Add(node);
  }
}

void CreateIndexings(const vec_set_t& level_nodes,
                     std::vector<Indexing>* indexings) {
  indexings->clear();
  indexings->resize(level_nodes.size());
  int k = 0;
  for (int i = 0; i < (int)level_nodes.size(); ++i) {
    (*indexings)[i].Clear();
    for (auto node : level_nodes[i]) {
      (*indexings)[i].Emplace(node, k);
      k += 1;
    }
  }
}

void RemoveDuplicateItems(const vec_int_t& pos_items,
                          const std::vector<vec_int_t>& neg_items_list,
                          vec_int_t* unique_items) {
  set_int_t item_set;
  item_set.insert(pos_items.begin(), pos_items.end());
  for (const auto& neg_items : neg_items_list) {
    item_set.insert(neg_items.begin(), neg_items.end());
  }

  unique_items->clear();
  unique_items->assign(item_set.begin(), item_set.end());
}

void CheckLengthValid(const std::vector<vec_int_t>& seqs) {
  DXCHECK(seqs.size() == 1u);
  const auto& seq = seqs[0];
  DXCHECK(seq.size() > 1u);
}

void ParseUserAndItemFrom(deepx_core::Instance* inst, const std::string& name,
                          uint16_t user_group, uint16_t item_group,
                          vec_int_t* user_nodes, vec_int_t* item_nodes) {
  auto* x_ptr = &inst->get_or_insert<csr_t>(name);
  user_nodes->clear();
  item_nodes->clear();

  CSR_FOR_EACH_ROW(*x_ptr, i) {
    CSR_FOR_EACH_COL(*x_ptr, i) {
      auto feat_id = CSR_COL(*x_ptr);
      auto group = io_util::GetNodeType(feat_id);
      if (group == user_group) {
        user_nodes->emplace_back(feat_id);
      }
      if (group == item_group) {
        item_nodes->emplace_back(feat_id);
      }
    }
    if ((int)user_nodes->size() != (i + 1)) {
      DXTHROW_INVALID_ARGUMENT(
          "Invalid user_nodes, the size of user_node: %zu must be %d.",
          user_nodes->size(), i + 1);
    }
    if ((int)item_nodes->size() != (i + 1)) {
      DXTHROW_INVALID_ARGUMENT(
          "Invalid item_nodes, the size of item_nodes: %zu must be %d.",
          item_nodes->size(), i + 1);
    }
  }
}

void GenerateSeqFrom(const GraphClient* client, const vec_int_t& user_nodes,
                     const vec_int_t& item_nodes,
                     const std::vector<int>& walk_lengths,
                     const WalkerInfo& walker_info,
                     std::vector<vec_int_t>* seqs) {
  vec_int_t tmp_nodes;
  tmp_nodes.insert(tmp_nodes.end(), user_nodes.begin(), user_nodes.end());
  tmp_nodes.insert(tmp_nodes.end(), item_nodes.begin(), item_nodes.end());

  client->StaticTraverse(tmp_nodes, walk_lengths, walker_info, seqs);
}

void ParseSeqTo(const std::vector<vec_int_t>& seqs, int window,
                uint16_t user_group, vec_int_t* src_nodes,
                vec_int_t* dst_nodes) {
  // src_nodes: users
  // dst_nodes: items
  src_nodes->clear();
  dst_nodes->clear();

  vec_int_t ctx_nodes;
  for (const auto& seq : seqs) {
    for (size_t i = 0; i < seq.size(); ++i) {
      GenerateIthContext((int)i, window, window, seq, &ctx_nodes);
      auto tgt_node = seq[i];
      auto tgt_group = io_util::GetNodeType(tgt_node);
      for (auto ctx_node : ctx_nodes) {
        auto ctx_group = io_util::GetNodeType(ctx_node);
        if (tgt_group != ctx_group) {
          if (tgt_group == user_group) {
            src_nodes->emplace_back(tgt_node);
            dst_nodes->emplace_back(ctx_node);
          } else {
            src_nodes->emplace_back(ctx_node);
            dst_nodes->emplace_back(tgt_node);
          }
        }
      }
    }
  }
}

}  // namespace inst_util
}  // namespace embedx
