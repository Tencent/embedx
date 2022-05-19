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

#include "src/io/loader/freq_file_loader.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

#include "src/io/io_util.h"
#include "src/io/line_parser.h"
#include "src/io/value.h"

namespace embedx {
namespace {

constexpr int BATCH = 128;

}  // namespace

bool FreqFileLoader::LoadConfig(const std::string& config_file) {
  if (!io_util::LoadConfig(config_file, &ns_size_, &id_name_map_)) {
    DXERROR("Failed to load config file.");
    return false;
  }
  return true;
}

bool FreqFileLoader::LoadFreq(const std::string& dir, int thread_num) {
  DXINFO("Loading files from dir: %s.", dir.c_str());

  vec_str_t freq_files;
  if (!io_util::ListFile(dir, &freq_files)) {
    DXERROR("Failed to list files from dir: %s.", dir.c_str());
    return false;
  }

  nodes_list_.clear();
  freqs_list_.clear();
  nodes_list_.resize(ns_size_);
  freqs_list_.resize(ns_size_);

  thread_num = std::min(thread_num, (int)freq_files.size());
  if (!io_util::ParallelProcess<std::string>(
          freq_files,
          [this](const vec_str_t& freq_files, int thread_id) {
            return LoadFreqEntry(freq_files, thread_id);
          },
          thread_num)) {
    DXERROR("Failed to load files.");
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool FreqFileLoader::LoadFreqEntry(const vec_str_t& freq_files, int thread_id) {
  LineParser line_parser;
  std::vector<NodeValue> node_freqs;

  for (const auto& file : freq_files) {
    DXINFO("Thread: %d is processing file: %s.", thread_id, file.c_str());

    if (!line_parser.Open(file)) {
      DXERROR("Failed to open file: %s.", file.c_str());
      return false;
    }

    // (node, frequency)
    while (line_parser.NextBatch<NodeValue>(BATCH, &node_freqs)) {
      for (auto& node_freq : node_freqs) {
        {
          auto node = node_freq.node;
          auto freq = node_freq.weight;
          std::lock_guard<std::mutex> guard(mtx_);
          auto ns_id = io_util::GetNodeType(node);
          if (id_name_map_.find(ns_id) == id_name_map_.end()) {
            DXERROR("Couldn't find node: %" PRIu64
                    " namespace id: %d in the config file.",
                    node, (int)ns_id);
            return false;
          }

          if (freq <= 0) {
            DXERROR("The frequency: %f of node: %" PRIu64
                    " must be greater than 0.",
                    freq, node);
            return false;
          }
          nodes_list_[ns_id].emplace_back(node);
          freqs_list_[ns_id].emplace_back(freq);
        }
      }
    }
  }

  DXINFO("Done.");
  return true;
}

std::unique_ptr<FreqFileLoader> FreqFileLoader::Create(
    const std::string& config, const std::string& dir, int thread_num) {
  std::unique_ptr<FreqFileLoader> loader;
  loader.reset(new FreqFileLoader());

  if (!loader->LoadConfig(config) || !loader->LoadFreq(dir, thread_num)) {
    DXERROR("Failed to create frequency file loader.");
    loader.reset();
  }

  return loader;
}

}  // namespace embedx
