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
#include "src/common/class_registry.h"
#include "src/deep/data_op/deep_op.h"

namespace embedx {
namespace deep_op {

using LocalDeepOpRegistry = ClassRegistry<LocalDeepOp>;

}  // namespace deep_op
}  // namespace embedx

#define REGISTER_LOCAL_DEEP_OP(OpName, OpClass)                         \
  static ::embedx::ClassFactoryRegister<::embedx::deep_op::LocalDeepOp, \
                                        OpClass>                        \
      register_##OpClass(OpName)
