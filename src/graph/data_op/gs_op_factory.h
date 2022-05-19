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

#pragma once
#include <deepx_core/ps/rpc_client.h>

#include <string>

#include "src/graph/data_op/gs_op.h"
#include "src/graph/data_op/gs_op_registry.h"
#include "src/graph/data_op/gs_op_resource.h"

namespace embedx {
namespace graph_op {

class LocalGSOpFactory {
 protected:
  LocalGSOpRegistry* op_registry_ = nullptr;
  const LocalGSOpResource* resource_ = nullptr;

 public:
  static LocalGSOpFactory* GetInstance();

  virtual bool Init(const LocalGSOpResource* resource) = 0;

  virtual LocalGSOp* LookupOrCreate(const std::string& name) = 0;

 protected:
  LocalGSOpFactory();
  virtual ~LocalGSOpFactory() = default;
};

class DistGSOpFactory {
 protected:
  DistGSOpRegistry* op_registry_ = nullptr;
  const DistGSOpResource* resource_ = nullptr;
  int shard_num_ = 0;

 public:
  static DistGSOpFactory* GetInstance();

  virtual bool Init(const DistGSOpResource* resource, int shard_num) = 0;

  virtual DistGSOp* LookupOrCreate(const std::string& name) = 0;

 protected:
  DistGSOpFactory();
  virtual ~DistGSOpFactory() = default;
};

}  // namespace graph_op
}  // namespace embedx
