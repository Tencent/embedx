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

#include "src/io/loader/inst_file_loader.h"

#include <deepx_core/dx_log.h>

#include <algorithm>  // std::min

#include "src/io/io_util.h"
#include "src/io/line_parser.h"
#include "src/io/value.h"

namespace embedx {
namespace {

constexpr int BATCH = 128;

}  // namespace

bool InstFileLoader::Load(const std::string& dir, int thread_num) {
  DXINFO("Loading files from dir: %s.", dir.c_str());

  vec_str_t files;
  if (!io_util::ListFile(dir, &files)) {
    DXERROR("Failed to load files from dir: %s.", dir.c_str());
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

  DXINFO("Done.");
  return true;
}

bool InstFileLoader::LoadEntry(const vec_str_t& files, int thread_id) {
  std::vector<NodeAndLabelValue> label_values;
  LineParser line_parser;

  for (const auto& file : files) {
    DXINFO("Thread: %d is processing file: %s.", thread_id, file.c_str());
    if (!line_parser.Open(file)) {
      DXERROR("Failed to open file: %s.", file.c_str());
      return false;
    }

    // (node, label)
    while (line_parser.NextBatch<NodeAndLabelValue>(BATCH, &label_values)) {
      vec_int_t batch_nodes = Collect<NodeAndLabelValue, int_t>(
          label_values, &NodeAndLabelValue::node);
      std::vector<vecl_t> batch_labels_list =
          Collect<NodeAndLabelValue, vecl_t>(label_values,
                                             &NodeAndLabelValue::labels);

      std::lock_guard<std::mutex> guard(mtx_);
      insts_.insert(insts_.end(), batch_nodes.begin(), batch_nodes.end());
      for (auto& batch_labels : batch_labels_list) {
        vec_labels_list_.emplace_back(batch_labels);
      }
    }
  }

  DXINFO("Done.");
  return true;
}

std::unique_ptr<InstFileLoader> InstFileLoader::Create(const std::string& dir,
                                                       int thread_num) {
  std::unique_ptr<InstFileLoader> loader;
  loader.reset(new InstFileLoader());

  if (!loader->Load(dir, thread_num)) {
    DXERROR("Failed to create inst file loader.");
    loader.reset();
  }
  return loader;
}

}  // namespace embedx
