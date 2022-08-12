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
#include <deepx_core/graph/instance_reader.h>
#include <deepx_core/graph/instance_reader_impl.h>
#include <deepx_core/graph/tensor_map.h>  // Instance

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/deep/client/deep_client.h"
#include "src/graph/client/graph_client.h"
#include "src/io/line_parser.h"

namespace embedx {

using ::deepx_core::Instance;
using ::deepx_core::InstanceReader;
using ::deepx_core::InstanceReaderImpl;

class EmbedInstanceReader : public InstanceReaderImpl {
 protected:
  const GraphClient* graph_client_ = nullptr;
  const DeepClient* deep_client_ = nullptr;

  LineParser line_parser_;

 public:
  virtual bool InitGraphClient(const GraphClient* graph_client);
  virtual bool InitDeepClient(const DeepClient* deep_client);
  virtual void PostInit(const std::string& /*node_config*/) {}

  bool Open(const std::string& file) override {
    return line_parser_.Open(file);
  }

 protected:
  template <typename ValueType>
  bool NextInstanceBatch(Instance* inst, int batch,
                         std::vector<ValueType>* values) {
    if (!line_parser_.NextBatch<ValueType>(batch, values)) {
      line_parser_.Close();
      inst->clear_batch();
      return false;
    }
    return true;
  }

  // InstanceReaderImpl
  void InitX(Instance* /* inst */) override {}
  void InitXBatch(Instance* /*inst*/) override {}
  bool ParseLine() override { return true; }
};

std::unique_ptr<EmbedInstanceReader> NewEmbedInstanceReader(
    const std::string& name);

}  // namespace embedx
