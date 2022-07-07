// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/sampler/random_walker/static_random_walker_impl.h"

#include <deepx_core/dx_log.h>

#include <utility>  // std::pair

#include "src/io/io_util.h"
#include "src/sampler/random_walker/random_walker_util.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

std::unique_ptr<RandomWalkerImpl> StaticRandomWalkerImpl::Create(
    const SamplerBuilder* sampler_builder) {
  std::unique_ptr<RandomWalkerImpl> random_walker_impl;
  random_walker_impl.reset(new StaticRandomWalkerImpl(sampler_builder));
  return random_walker_impl;
}

void StaticRandomWalkerImpl::Traverse(const vec_int_t& cur_nodes,
                                      const std::vector<int>& walk_lens,
                                      const WalkerInfo& walker_info,
                                      std::vector<vec_int_t>* seqs,
                                      PrevInfo* /*prev_info*/) const {
  if (walker_info.meta_path.empty()) {
    Traverse(cur_nodes, walk_lens, seqs);
  } else {
    MetaPathTraverse(cur_nodes, walk_lens, walker_info, seqs);
  }
}

void StaticRandomWalkerImpl::Traverse(const vec_int_t& cur_nodes,
                                      const std::vector<int>& walk_lens,
                                      std::vector<vec_int_t>* seqs) const {
  seqs->clear();
  seqs->resize(cur_nodes.size());
  int_t next_node;
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    auto cur_node = cur_nodes[i];
    for (int j = 0; j < walk_lens[i]; ++j) {
      if (neighbor_sampler_builder_.Next(cur_node, &next_node)) {
        (*seqs)[i].emplace_back(next_node);
        cur_node = next_node;
      } else {
        break;
      }
    }
  }
}

void StaticRandomWalkerImpl::MetaPathTraverse(
    const vec_int_t& cur_nodes, const std::vector<int>& walk_lens,
    const WalkerInfo& walker_info, std::vector<vec_int_t>* seqs) const {
  seqs->clear();
  seqs->resize(cur_nodes.size());
  int_t next_node;
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    auto cur_node = cur_nodes[i];
    auto cur_index = walker_info.walker_length - walk_lens[i];

    for (int j = cur_index; j < walker_info.walker_length; ++j) {
      if (MetaPathNext(walker_info.meta_path, cur_node, j, &next_node)) {
        (*seqs)[i].emplace_back(next_node);
        cur_node = next_node;
      } else {
        break;
      }
    }
  }
}

bool StaticRandomWalkerImpl::MetaPathNext(const meta_path_t& meta_path,
                                          int_t cur_node, int cur_index,
                                          int_t* next_node) const {
  DXASSERT(cur_index >= 0);

  auto* context =
      neighbor_sampler_builder_.sampler_source().FindContext(cur_node);
  if (context == nullptr) {
    return false;
  }

  uint16_t expected_next_type = meta_path[(cur_index + 1) % meta_path.size()];

  std::pair<int, int> bound;
  if (!random_walker_util::FindBound(*context, expected_next_type, &bound)) {
    return false;
  }

  return neighbor_sampler_builder_.Next(cur_node, bound.first, bound.second,
                                        next_node);
}

std::unique_ptr<RandomWalkerImpl> NewStaticRandomWalkerImpl(
    const SamplerBuilder* sampler_builder) {
  return StaticRandomWalkerImpl::Create(sampler_builder);
}

}  // namespace embedx
