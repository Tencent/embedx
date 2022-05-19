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

#include "src/graph/graph_builder.h"

#include <deepx_core/dx_log.h>

namespace embedx {

void GraphBuilder::InitLoader(int shard_num, int shard_id, int store_type) {
  context_loader_ = NewContextLoader(shard_num, shard_id, store_type);
  node_feat_loader_ = NewFeatureLoader(shard_num, shard_id, store_type);
  neigh_feat_loader_ = NewFeatureLoader(shard_num, shard_id, store_type);
}

/************************************************************************/
/* Build graph */
/************************************************************************/
bool GraphBuilder::BuildContext(const std::string& context, int thread_num) {
  DXINFO("Building graph context...");

  if (context.empty()) {
    DXERROR("Context files are empty.");
    return false;
  }

  context_loader_->Clear();
  context_loader_->Reserve(estimated_size_);
  if (!context_loader_->Load(context, thread_num)) {
    DXERROR("Failed to load context files.");
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool GraphBuilder::BuildNodeFeature(const std::string& node_feature,
                                    int thread_num) {
  DXINFO("Building graph feature...");

  if (!node_feature.empty()) {
    node_feat_loader_->Clear();
    node_feat_loader_->Reserve(estimated_size_);
    if (!node_feat_loader_->Load(node_feature, thread_num)) {
      DXERROR("Failed to load node feature.");
      return false;
    }
  }

  return true;
}

bool GraphBuilder::BuildNeighborFeature(const std::string& neighbor_feature,
                                        int thread_num) {
  if (!neighbor_feature.empty()) {
    neigh_feat_loader_->Clear();
    neigh_feat_loader_->Reserve(estimated_size_);
    if (!neigh_feat_loader_->Load(neighbor_feature, thread_num)) {
      DXERROR("Failed to load neighbor feature.");
      return false;
    }
  }

  DXINFO("Done.");
  return true;
}

std::unique_ptr<GraphBuilder> GraphBuilder::Create(const GraphConfig& config) {
  std::unique_ptr<GraphBuilder> builder;
  builder.reset(new GraphBuilder());

  builder->set_estimated_size(config.estimated_size());
  builder->InitLoader(config.shard_num(), config.shard_id(),
                      config.store_type());

  if (!builder->BuildContext(config.node_graph(), config.thread_num()) ||
      !builder->BuildNodeFeature(config.node_feature(), config.thread_num()) ||
      !builder->BuildNeighborFeature(config.neighbor_feature(),
                                     config.thread_num())) {
    DXERROR("Failed to create graph builder.");
    builder.reset();
  }

  return builder;
}

}  // namespace embedx
