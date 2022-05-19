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
#include "src/io/value.h"

namespace embedx {

class Storage {
 public:
  Storage() = default;
  virtual ~Storage() = default;

 public:
  virtual void Clear() noexcept = 0;
  virtual void Reserve(uint64_t estimated_size) = 0;
  virtual void Lock() = 0;
  virtual void UnLock() = 0;
  virtual bool InsertContext(AdjValue*) { return true; }
  virtual bool InsertFeature(AdjValue*) { return true; }
  virtual bool InsertEdge(EdgeValue*) { return true; }

 public:
  virtual size_t Size() const noexcept = 0;
  virtual bool Empty() const noexcept = 0;
  virtual const vec_int_t& Keys() const noexcept = 0;

 public:
  virtual const vec_pair_t* FindNeighbor(int_t node) const = 0;
  virtual std::string Print(int_t node) const = 0;
  virtual int GetInDegree(int_t dst_node) const = 0;
  virtual int GetOutDegree(int_t src_node) const = 0;
};

std::unique_ptr<Storage> NewContextStorage(int store_type);
std::unique_ptr<Storage> NewFeatureStorage(int store_type);
std::unique_ptr<Storage> NewEdgeStorage(int store_type);

}  // namespace embedx
