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
                                      const WalkerInfo& /*walker_info*/,
                                      std::vector<vec_int_t>* seqs,
                                      PrevInfo* /*prev_info*/) const {
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

std::unique_ptr<RandomWalkerImpl> NewStaticRandomWalkerImpl(
    const SamplerBuilder* sampler_builder) {
  return StaticRandomWalkerImpl::Create(sampler_builder);
}

}  // namespace embedx
