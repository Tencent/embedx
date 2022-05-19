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

#include "src/io/loader/loader.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::min

#include "src/io/io_util.h"

namespace embedx {

bool Loader::Load(const std::string& path, int thread_num) {
  vec_str_t files;
  if (!io_util::ListFile(path, &files)) {
    return false;
  }

  thread_num = std::min(thread_num, (int)files.size());
  if (!io_util::ParallelProcess<std::string>(
          files,
          [this](const vec_str_t& files, int thread_id) {
            return LoadEntry(files, thread_id);
          },
          thread_num)) {
    DXERROR("Failed to load files.");
    return false;
  }

  return true;
}

}  // namespace embedx
