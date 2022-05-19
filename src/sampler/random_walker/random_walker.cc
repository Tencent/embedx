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

#include "src/sampler/random_walker.h"

#include <utility>  // std::move

#include "src/sampler/random_walker/random_walker_impl.h"

namespace embedx {

RandomWalker::RandomWalker(std::unique_ptr<RandomWalkerImpl>&& impl) {
  impl_ = std::move(impl);
}

RandomWalker::~RandomWalker() {}

void RandomWalker::Traverse(const vec_int_t& cur_nodes,
                            const std::vector<int>& walk_lens,
                            const WalkerInfo& walker_info,
                            std::vector<vec_int_t>* seqs,
                            PrevInfo* prev_info) const {
  impl_->Traverse(cur_nodes, walk_lens, walker_info, seqs, prev_info);
}

std::unique_ptr<RandomWalker> NewRandomWalker(
    const SamplerBuilder* sampler_builder, RandomWalkerEnum type) {
  std::unique_ptr<RandomWalker> random_walker;
  switch (type) {
    case RandomWalkerEnum::STATIC:
      random_walker.reset(
          new RandomWalker(NewStaticRandomWalkerImpl(sampler_builder)));
      break;
    default:
      DXERROR("Need type: STATIC(0), got type: %d.", (int)type);
      break;
  }
  return random_walker;
}

}  // namespace embedx
