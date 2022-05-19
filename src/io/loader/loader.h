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
#include <memory>  // std::unique_ptr
#include <string>

#include "src/common/data_types.h"
#include "src/io/storage/storage.h"

namespace embedx {

class Loader {
 public:
  Loader() = default;
  virtual ~Loader() = default;

 public:
  virtual void Clear() noexcept = 0;
  virtual void Reserve(uint64_t estimated_size) = 0;
  virtual const Storage* storage() const noexcept = 0;

 public:
  virtual bool Load(const std::string& path, int thread_num);
  virtual bool PartOfShard(int_t node, int shard_num, int shard_id) const {
    return node % shard_num == (size_t)shard_id;
  }

 protected:
  virtual bool LoadEntry(const vec_str_t& files, int thread_id) = 0;
};

std::unique_ptr<Loader> NewContextLoader(int shard_num = 1, int shard_id = 0,
                                         int store_type = 0);
std::unique_ptr<Loader> NewFeatureLoader(int shard_num = 1, int shard_id = 0,
                                         int store_type = 0);
std::unique_ptr<Loader> NewEdgeLoader(int shard_num = 1, int shard_id = 0,
                                      int store_type = 0);

}  // namespace embedx
