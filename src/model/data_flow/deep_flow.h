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
#include <deepx_core/graph/tensor_map.h>  // Instance

#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/io/indexing.h"
#include "src/io/io_util.h"

namespace embedx {

using ::deepx_core::Instance;

class DeepFlow : public deepx_core::DataType {
 public:
  virtual ~DeepFlow() = default;

 public:
  void FillNodeOrIndex(Instance* inst, const std::string& name,
                       const vec_int_t& nodes, const Indexing* indexing) const;
  void FillNodeFeature(Instance* inst, const std::string& name,
                       const vec_int_t* nodes_ptr,
                       const std::vector<vec_pair_t>& node_feats_list) const;

 public:
  template <class IndexingFunc>
  void FillEdge(Instance* inst, const std::string& user_name,
                const std::string& item_name, const vec_int_t& pos_items,
                const std::vector<vec_int_t>& self_training_items_list,
                IndexingFunc&& indexing_func) const {
    auto* user_ptr = &inst->get_or_insert<csr_t>(user_name);
    auto* item_ptr = &inst->get_or_insert<csr_t>(item_name);
    user_ptr->clear();
    item_ptr->clear();

    for (size_t i = 0; i < pos_items.size(); ++i) {
      for (auto& items : self_training_items_list) {
        for (auto item : items) {
          user_ptr->emplace(i, 1);
          user_ptr->add_row();
          item_ptr->emplace(indexing_func(item), 1);
          item_ptr->add_row();
        }
      }
    }
  }

  template <class PosIndexingFunc, class NegIndexingFunc>
  void FillEdgeAndLabel(Instance* inst, const std::string& user_name,
                        const std::string& item_name, const std::string& y_name,
                        const vec_int_t& pos_items,
                        const std::vector<vec_int_t>& neg_items_list,
                        PosIndexingFunc&& pos_indexing_func,
                        NegIndexingFunc&& neg_indexing_func) const {
    auto* user_ptr = &inst->get_or_insert<csr_t>(user_name);
    auto* item_ptr = &inst->get_or_insert<csr_t>(item_name);
    auto* y_ptr = &inst->get_or_insert<tsr_t>(y_name);
    user_ptr->clear();
    item_ptr->clear();
    y_ptr->clear();
    y_ptr->resize((int)pos_items.size(), 1);

    int k = 0;
    for (size_t i = 0; i < pos_items.size(); ++i) {
      // (user, pos_item)
      auto pos = pos_indexing_func(pos_items[i]);
      user_ptr->emplace(i, 1);
      user_ptr->add_row();
      item_ptr->emplace(pos, 1);
      item_ptr->add_row();
      // sampled softmax: [pos, neg, neg, neg, ...]
      // Y should be set as the position of pos items (0 in this case)
      y_ptr->data(k++) = 0;

      // (user, neg_item)
      auto ns = io_util::GetNodeType(pos_items[i]);
      for (auto neg_item : neg_items_list[ns]) {
        auto neg = neg_indexing_func(neg_item);
        user_ptr->emplace(i, 1);
        user_ptr->add_row();
        item_ptr->emplace(neg, 1);
        item_ptr->add_row();
      }
    }
  }
};

}  // namespace embedx
