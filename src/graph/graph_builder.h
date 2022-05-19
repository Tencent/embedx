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
#include <string>

#include "src/common/data_types.h"
#include "src/graph/graph_config.h"
#include "src/io/loader/loader.h"
#include "src/io/storage/storage.h"

namespace embedx {

class GraphBuilder {
 private:
  uint64_t estimated_size_ = 1000000;  // magic number
  std::unique_ptr<Loader> context_loader_;
  std::unique_ptr<Loader> node_feat_loader_;
  std::unique_ptr<Loader> neigh_feat_loader_;

 public:
  static std::unique_ptr<GraphBuilder> Create(const GraphConfig& config);

 public:
  const Storage* context_storage() const noexcept {
    return context_loader_->storage();
  }
  const Storage* node_feature_storage() const noexcept {
    return node_feat_loader_->storage();
  }
  const Storage* neigh_feature_storage() const noexcept {
    return neigh_feat_loader_->storage();
  }

 private:
  void set_estimated_size(uint64_t size) noexcept { estimated_size_ = size; }
  void InitLoader(int shard_num, int shard_id, int store_type);
  bool BuildContext(const std::string& context, int thread_num);
  bool BuildNodeFeature(const std::string& node_feature, int thread_num);
  bool BuildNeighborFeature(const std::string& neighbor_feature,
                            int thread_num);

 private:
  GraphBuilder() = default;
};

}  // namespace embedx
