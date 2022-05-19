// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//

#pragma once
#include <string>

#include "src/deep/data_op/deep_op_registry.h"
#include "src/deep/data_op/deep_op_resource.h"

namespace embedx {
namespace deep_op {

class LocalDeepOpFactory {
 protected:
  LocalDeepOpRegistry* op_registry_ = nullptr;
  const LocalDeepOpResource* resource_ = nullptr;

 public:
  static LocalDeepOpFactory* GetInstance();

  virtual bool Init(const LocalDeepOpResource* resource) = 0;

  virtual LocalDeepOp* LookupOrCreate(const std::string& name) = 0;

 protected:
  LocalDeepOpFactory();
  virtual ~LocalDeepOpFactory() = default;
};

}  // namespace deep_op
}  // namespace embedx
