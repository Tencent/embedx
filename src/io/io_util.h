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
#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {
namespace io_util {

bool ListFile(const std::string& dir, vec_str_t* files);

uint16_t GetNodeType(int_t node);
void ParseMaxNodeType(int max_num, const vec_int_t& nodes,
                      std::unordered_set<uint16_t>* ns_ids);

bool LoadConfig(const std::string& file, uint16_t* ns_size,
                id_name_t* id_name_map);

template <class T>
bool ParallelProcess(
    const std::vector<T>& in,
    const std::function<bool(const std::vector<T>&, int)>& processor,
    int thread_num) {
  std::vector<std::vector<T>> partitioned_in;
  partitioned_in.resize(thread_num);
  for (size_t i = 0; i < in.size(); ++i) {
    int j = i % thread_num;
    partitioned_in[j].emplace_back(in[i]);
  }

  std::vector<std::thread> threads;
  std::atomic_bool success{true};
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(
        std::thread([i, &partitioned_in, &processor, &success]() {
          if (!processor(partitioned_in[i], i)) {
            success = false;
          }
        }));
  }
  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
  return success;
}

}  // namespace io_util
}  // namespace embedx
