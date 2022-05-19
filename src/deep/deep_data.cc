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

#include "src/deep/deep_data.h"

#include <deepx_core/dx_log.h>

#include <string>

namespace embedx {

std::unique_ptr<DeepData> DeepData::Create(const DeepConfig& config) {
  std::unique_ptr<DeepData> deep_data;
  deep_data.reset(new DeepData);

  if (!deep_data->Build(config)) {
    DXERROR("Failed to create deep data.");
    deep_data.reset();
  }

  return deep_data;
}

bool DeepData::Build(const DeepConfig& config) {
  // item feature
  if (!config.item_feature().empty()) {
    item_feature_loader_ = NewFeatureLoader();
    if (!item_feature_loader_->Load(config.item_feature(),
                                    config.thread_num())) {
      DXERROR("Failed to load item feature from %s.",
              config.item_feature().c_str());
      return false;
    }
  }

  // inst file
  if (!config.inst_file().empty()) {
    inst_file_loader_ =
        InstFileLoader::Create(config.inst_file(), config.thread_num());
    if (!inst_file_loader_) {
      return false;
    }
  }

  // freq file
  if (!config.freq_file().empty()) {
    freq_file_loader_ = FreqFileLoader::Create(
        config.node_config(), config.freq_file(), config.thread_num());
    if (!freq_file_loader_) {
      return false;
    }
  }

  if (!item_feature_loader_ && !inst_file_loader_ && !freq_file_loader_) {
    DXERROR("Failed to build deep data, check if config file is empty.");
    return false;
  }

  return true;
}

}  // namespace embedx
