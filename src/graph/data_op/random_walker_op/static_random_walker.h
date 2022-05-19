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
#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/proto/random_walker_proto.h"
#include "src/sampler/random_walker.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {
namespace graph_op {

class StaticRandomWalker : public LocalGSOp {
 private:
  std::unique_ptr<RandomWalker> random_walker_;

 public:
  ~StaticRandomWalker() override = default;

 public:
  bool Run(const vec_int_t& cur_nodes, const std::vector<int>& walk_lens,
           const WalkerInfo&, std::vector<vec_int_t>* seqs) const;
  int HandleRpc(const StaticRandomWalkerRequest& req,
                StaticRandomWalkerResponse* resp) const;

 private:
  bool Init(const LocalGSOpResource* resource) override {
    random_walker_ = NewRandomWalker(resource->neighbor_sampler_builder(),
                                     RandomWalkerEnum::STATIC);
    return random_walker_ != nullptr;
  }
};

}  // namespace graph_op
}  // namespace embedx
