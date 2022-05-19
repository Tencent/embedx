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
#include "src/graph/data_op/feature_lookuper_op/feature.h"
#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_resource.h"
#include "src/graph/proto/graph_service_proto.h"

namespace embedx {
namespace graph_op {

class FeatureLookuper : public LocalGSOp {
 private:
  std::unique_ptr<Feature> feature_;

 public:
  ~FeatureLookuper() override = default;

 public:
  bool Run(const vec_int_t& nodes, std::vector<vec_pair_t>* node_feats,
           std::vector<vec_pair_t>* neigh_feats) const;
  int HandleRpc(const FeatureLookuperRequest& req,
                FeatureLookuperResponse* resp) const;

 private:
  bool Init(const LocalGSOpResource* resource) override {
    feature_ = NewFeature(resource->graph());
    return feature_ != nullptr;
  }
};

}  // namespace graph_op
}  // namespace embedx
