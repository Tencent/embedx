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
#include "src/graph/proto/graph_service_proto.h"
#include "src/sampler/neighbor_sampler.h"

namespace embedx {
namespace graph_op {

class RandomNeighborSampler : public LocalGSOp {
 private:
  std::unique_ptr<NeighborSampler> neighbor_sampler_;

 public:
  ~RandomNeighborSampler() override = default;

 public:
  bool Run(int count, const vec_int_t& nodes,
           std::vector<vec_int_t>* neighbor_nodes_list) const;

  int HandleRpc(const RandomNeighborSamplerRequest& req,
                RandomNeighborSamplerResponse* resp) const;

 private:
  bool Init(const LocalGSOpResource* resource) override {
    neighbor_sampler_ =
        NewNeighborSampler(resource->neighbor_sampler_builder());
    return neighbor_sampler_ != nullptr;
  }
};

}  // namespace graph_op
}  // namespace embedx
