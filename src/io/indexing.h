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

#include "src/common/data_types.h"

namespace embedx {

class Indexing {
 private:
  index_map_t index_map_;

 public:
  void Reserve(uint64_t estimated_size);
  void Clear() noexcept;
  void Add(int_t node);
  void Emplace(int_t k, int v);
  int Get(int_t node) const;
  bool Find(int_t node) const;
  size_t Size() const noexcept;
};

}  // namespace embedx
