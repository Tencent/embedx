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

#include "src/tools/graph/main_util.h"

#include <algorithm>  // std::min

#include "src/io/io_util.h"

namespace embedx {
namespace {

void MakeDir(const std::string& out) {
  if (!deepx_core::AutoFileSystem::Exists(out)) {
    if (!deepx_core::AutoFileSystem::MakeDir(out)) {
      DXINFO("Failed to make dir.");
    }
  } else {
    DXINFO("Dir: %s exists.", out.c_str());
  }
}

}  // namespace

bool MainUtil::RunSingleWorker(const std::string& in, const std::string& out,
                               int worker_num, int worker_id) {
  DXINFO("Runing %s...", task_name());

  vec_str_t in_files;
  if (!io_util::ListFile(in, &in_files)) {
    return false;
  }

  // partition
  vec_str_t worker_files;
  for (size_t i = 0; i < in_files.size(); ++i) {
    if (i % worker_num == (size_t)worker_id) {
      worker_files.emplace_back(in_files[i]);
    }
  }

  MakeDir(out);
  if (!RunEntry(worker_id, worker_files, out)) {
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool MainUtil::RunMultiThread(const std::string& in, const std::string& out,
                              int thread_num) {
  DXINFO("Runing %s...", task_name());

  vec_str_t in_files;
  if (!io_util::ListFile(in, &in_files)) {
    return false;
  }

  MakeDir(out);

  thread_num = std::min(thread_num, (int)in_files.size());
  if (!io_util::ParallelProcess<std::string>(
          in_files,
          [this, &out](const vec_str_t& in_files, int thread_id) {
            return RunEntry(thread_id, in_files, out);
          },
          thread_num)) {
    DXERROR("Failed to run %s.", task_name());
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool MainUtil::DumpText(deepx_core::AutoOutputFileStream& ofs /* NOLINT */,
                        std::ostringstream& oss /* NOLINT*/,
                        const vec_int_t& nodes,
                        const std::vector<vec_pair_t>& pairs_list) const {
  for (size_t i = 0; i < nodes.size(); ++i) {
    oss.clear();
    oss.str("");
    oss << nodes[i];

    for (const auto& pair : pairs_list[i]) {
      oss << " " << pair.first << ":" << pair.second;
    }
    oss << "\n";

    std::string s = oss.str();
    ofs.Write(s.data(), s.size());
    if (!ofs) {
      DXERROR("Failed to dump file.");
      return false;
    }
  }

  return true;
}

}  // namespace embedx
