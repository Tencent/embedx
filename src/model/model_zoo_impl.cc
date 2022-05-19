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

#include "src/model/model_zoo_impl.h"

#include <deepx_core/dx_log.h>

namespace embedx {

/************************************************************************/
/* ModelZooImpl */
/************************************************************************/
bool ModelZooImpl::InitConfig(const deepx_core::AnyMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const auto& v = entry.second.to_ref<std::string>();
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }
  return true;
}

bool ModelZooImpl::InitConfig(const deepx_core::StringMap& config) {
  if (!PreInitConfig()) {
    return false;
  }

  for (const auto& entry : config) {
    const std::string& k = entry.first;
    const std::string& v = entry.second;
    if (!InitConfigKV(k, v)) {
      return false;
    }
  }

  if (!PostInitConfig()) {
    return false;
  }
  return true;
}

bool ModelZooImpl::InitConfigKV(const std::string& k, const std::string& v) {
  if (k == "config" || k == "group_config") {
    if (!GuessGroupConfig(v, &items_, nullptr)) {
      return false;
    }
    item_is_fm_ = IsFMGroupConfig(items_) ? 1 : 0;
    item_m_ = (int)items_.size();
    if (item_is_fm_) {
      item_k_ = items_.front().embedding_col;
    } else {
      item_k_ = 0;
    }
    item_mk_ = GetTotalEmbeddingCol(items_);
  } else if (k == "w" || k == "has_w") {
    has_w_ = std::stoi(v);
  } else if (k == "sparse") {
    sparse_ = std::stoi(v);
  } else {
    return false;
  }
  return true;
}

/************************************************************************/
/* ModelZoo functions */
/************************************************************************/
std::unique_ptr<ModelZoo> NewModelZoo(const std::string& name) {
  std::unique_ptr<ModelZoo> model_zoo(MODEL_ZOO_NEW(name));
  if (!model_zoo) {
    DXERROR("Invalid model name: %s.", name.c_str());
    DXERROR("Model name can be: ");
    for (const auto& _name : MODEL_ZOO_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return model_zoo;
}

}  // namespace embedx
