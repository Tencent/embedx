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

#include "src/model/embed_instance_reader.h"

#include <deepx_core/dx_log.h>

namespace embedx {

bool EmbedInstanceReader::InitGraphClient(const GraphClient* graph_client) {
  if (graph_client == nullptr) {
    DXERROR("Graph_client is nullptr.");
    return false;
  }
  graph_client_ = graph_client;
  return true;
}

bool EmbedInstanceReader::InitDeepClient(const DeepClient* deep_client) {
  if (deep_client == nullptr) {
    DXERROR("Deep_client is nullptr.");
    return false;
  }
  deep_client_ = deep_client;
  return true;
}

std::unique_ptr<EmbedInstanceReader> NewEmbedInstanceReader(
    const std::string& name) {
  std::unique_ptr<EmbedInstanceReader> instance_reader(
      (EmbedInstanceReader*)INSTANCE_READER_NEW(name));
  if (!instance_reader) {
    DXERROR("Invalid instance reader name: %s.", name.c_str());
    DXERROR("Instance reader name can be: ");
    for (const auto& _name : INSTANCE_READER_NAMES()) {
      DXERROR("  %s", _name.c_str());
    }
  }
  return instance_reader;
}

}  // namespace embedx
