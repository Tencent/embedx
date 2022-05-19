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
#include <deepx_core/dx_log.h>
#include <deepx_core/ps/rpc_client.h>

#include "src/common/data_types.h"
#include "src/graph/data_op/gs_op_resource.h"

namespace embedx {
namespace graph_op {

class LocalGSOp {
 public:
  virtual ~LocalGSOp() = default;

 public:
  virtual bool Init(const LocalGSOpResource* resource) = 0;
};

class DistGSOp {
 protected:
  deepx_core::TcpConnections* conns_ = nullptr;
  const DistGSOpResource* resource_ = nullptr;
  int shard_num_ = 0;

 public:
  virtual ~DistGSOp() = default;

 public:
  bool Init(const DistGSOpResource* resource, int shard_num) {
    if (resource->rpc_connector()->conns() == nullptr) {
      DXERROR("The connections of DistGSOpResource is nullptr.");
      return false;
    }
    if (shard_num <= 0) {
      DXERROR("Number of shard: %d must be greater than 0.", shard_num);
      return false;
    }

    resource_ = resource;
    conns_ = resource_->rpc_connector()->conns();
    shard_num_ = shard_num;
    return true;
  }

 protected:
  int ModShard(int_t node) const noexcept { return node % shard_num_; }
};

}  // namespace graph_op
}  // namespace embedx
