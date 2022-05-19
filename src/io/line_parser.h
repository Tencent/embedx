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
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>

#include <sstream>  // std::istringstream
#include <string>
#include <vector>

#include "src/io/value.h"

namespace embedx {

class LineParser {
 private:
  std::string line_;
  std::istringstream iss_;
  deepx_core::AutoInputFileStream ifs_;

 public:
  bool Open(const std::string& file) {
    ifs_.Close();
    if (!ifs_.Open(file)) {
      DXERROR("Failed to open file: %s.", file.c_str());
      return false;
    }
    return true;
  }
  void Close() noexcept { ifs_.Close(); }

 public:
  template <typename ValueType>
  bool NextBatch(int batch, std::vector<ValueType>* values) {
    values->clear();
    ValueType value;

    for (;;) {
      if (!GetLine(ifs_, line_)) {
        break;
      }

      if (ParseValue(line_, &value)) {
        values->emplace_back(value);
        if (values->size() == (size_t)batch) {
          break;
        }
      }
    }

    return !values->empty();
  }

 private:
  bool ParseValue(const std::string& line, NodeValue* node);
  bool ParseValue(const std::string& line, EdgeValue* value);
  bool ParseValue(const std::string& line, SeqValue* value);
  bool ParseValue(const std::string& line, AdjValue* value);
  bool ParseValue(const std::string& line, NodeAndLabelValue* value);
  bool ParseValue(const std::string& line, EdgeAndLabelValue* value);
};

}  // namespace embedx
