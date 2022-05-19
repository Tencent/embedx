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
#include "src/graph/client/graph_client.h"
#include "src/io/indexing.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {
namespace inst_util {

// Generate nodes based on the size of the left and
// right windows in the sequence
void GenerateIthContext(int i, int left, int right, const vec_int_t& seq,
                        vec_int_t* context);
void GenerateContext(int left, int right, const vec_int_t& seq,
                     std::vector<vec_int_t>* contexts_list);

void CreateIndexing(const vec_int_t& nodes, Indexing* indexing);
void CreateIndexings(const vec_set_t& level_nodes,
                     std::vector<Indexing>* indexings);

void RemoveDuplicateItems(const vec_int_t& pos_items,
                          const std::vector<vec_int_t>& neg_items_list,
                          vec_int_t* unique_items);
void CheckLengthValid(const std::vector<vec_int_t>& seqs);

void ParseUserAndItemFrom(deepx_core::Instance* inst, const std::string& name,
                          uint16_t user_group, uint16_t item_group,
                          vec_int_t* user_nodes, vec_int_t* item_nodes);

void GenerateSeqFrom(const GraphClient* client, const vec_int_t& user_nodes,
                     const vec_int_t& item_nodes,
                     const std::vector<int>& walk_lengths,
                     const WalkerInfo& walker_info,
                     std::vector<vec_int_t>* seqs);

void ParseSeqTo(const std::vector<vec_int_t>& seqs, int window,
                uint16_t user_group, vec_int_t* src_nodes,
                vec_int_t* dst_nodes);

}  // namespace inst_util
}  // namespace embedx
