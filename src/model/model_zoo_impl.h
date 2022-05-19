// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#pragma once
#include <string>
#include <vector>

#include "src/model/model_zoo.h"

namespace embedx {

#define MODEL_ZOO_REGISTER(class_name, name) \
  CLASS_FACTORY_REGISTER(ModelZoo, class_name, name)
#define MODEL_ZOO_NEW(name) CLASS_FACTORY_NEW(ModelZoo, name)
#define MODEL_ZOO_NAMES() CLASS_FACTORY_NAMES(ModelZoo)
#define DEFINE_MODEL_ZOO_LIKE(clazz_name) \
  const char* class_name() const noexcept override { return #clazz_name; }

/************************************************************************/
/* ModelZooImpl */
/************************************************************************/
class ModelZooImpl : public ModelZoo {
 protected:
  std::vector<deepx_core::GroupConfigItem3> items_;
  int item_is_fm_ = 0;
  int item_m_ = 0;
  int item_k_ = 0;
  int item_mk_ = 0;
  int has_w_ = 0;
  int sparse_ = 0;

 public:
  bool InitConfig(const deepx_core::AnyMap& config) override;
  bool InitConfig(const deepx_core::StringMap& config) override;

 protected:
  virtual bool PreInitConfig() { return true; }
  virtual bool InitConfigKV(const std::string& k, const std::string& v) = 0;
  virtual bool PostInitConfig() { return true; }
};

}  // namespace embedx
