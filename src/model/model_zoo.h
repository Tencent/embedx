// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#pragma once
#include <deepx_core/common/any_map.h>
#include <deepx_core/common/group_config.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/variable_scope.h>

#include <memory>  // std::unique_ptr

namespace embedx {

/************************************************************************/
/* ModelZoo */
/************************************************************************/
class ModelZoo {
 public:
  virtual ~ModelZoo() = default;
  virtual const char* class_name() const noexcept = 0;
  virtual bool InitConfig(const deepx_core::AnyMap& config) = 0;
  virtual bool InitConfig(const deepx_core::StringMap& config) = 0;
  virtual bool InitGraph(deepx_core::Graph* graph) const = 0;
};

/************************************************************************/
/* ModelZoo functions */
/************************************************************************/
std::unique_ptr<ModelZoo> NewModelZoo(const std::string& name);

}  // namespace embedx
