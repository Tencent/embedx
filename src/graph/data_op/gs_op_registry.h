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
#include "src/common/class_registry.h"
#include "src/graph/data_op/gs_op.h"

namespace embedx {
namespace graph_op {

using LocalGSOpRegistry = ClassRegistry<LocalGSOp>;
using DistGSOpRegistry = ClassRegistry<DistGSOp>;

}  // namespace graph_op
}  // namespace embedx

#define REGISTER_LOCAL_GS_OP(OpName, OpClass)                          \
  static ::embedx::ClassFactoryRegister<::embedx::graph_op::LocalGSOp, \
                                        OpClass>                       \
      register_##OpClass(OpName)

#define REGISTER_DIST_GS_OP(OpName, OpClass)                                   \
  static ::embedx::ClassFactoryRegister<::embedx::graph_op::DistGSOp, OpClass> \
      register_##OpClass(OpName)
