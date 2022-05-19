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

#include <sstream>  // std::ostringstream
#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

class MainUtil {
 public:
  virtual ~MainUtil() = default;

 public:
  virtual bool Init() = 0;
  bool RunSingleWorker(const std::string& in, const std::string& out,
                       int worker_num, int worker_id);
  bool RunMultiThread(const std::string& in, const std::string& out,
                      int thread_num);

 protected:
  virtual const char* task_name() const noexcept = 0;
  virtual bool RunEntry(int entry_id, const vec_str_t& in_files,
                        const std::string& out) = 0;
  bool DumpText(deepx_core::AutoOutputFileStream& ofs /* NOLINT */,
                std::ostringstream& oss /* NOLINT*/, const vec_int_t& nodes,
                const std::vector<vec_pair_t>& pairs_list) const;
};

}  // namespace embedx
