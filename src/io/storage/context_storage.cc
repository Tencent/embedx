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

#include <memory>  // std::unique_ptr
#include <mutex>
#include <string>

#include "src/common/data_types.h"
#include "src/io/storage/adjacency.h"
#include "src/io/storage/storage.h"
#include "src/io/value.h"

namespace embedx {

class ContextStorage : public Storage {
 private:
  std::unique_ptr<Adjacency> adj_;
  std::mutex mtx_;

 public:
  explicit ContextStorage(int store_type) {
    adj_ = NewAdjacency((AdjacencyEnum)store_type);
  }
  ~ContextStorage() override = default;

 public:
  void Clear() noexcept override { adj_->Clear(); }
  void Reserve(uint64_t estimated_size) override {
    adj_->Reserve(estimated_size);
  }
  void Lock() override { mtx_.lock(); }
  void UnLock() override { mtx_.unlock(); }
  bool InsertContext(AdjValue* value) override {
    return adj_->AddContext(value);
  }

 public:
  size_t Size() const noexcept override { return adj_->Size(); }
  bool Empty() const noexcept override { return adj_->Empty(); }
  const vec_int_t& Keys() const noexcept override { return adj_->Keys(); }

 public:
  const vec_pair_t* FindNeighbor(int_t node) const override {
    return adj_->FindNeighbor(node);
  }
  std::string Print(int_t node) const override { return adj_->Print(node); }
  int GetInDegree(int_t dst_node) const override {
    return adj_->GetInDegree(dst_node);
  }
  int GetOutDegree(int_t src_node) const override {
    return adj_->GetOutDegree(src_node);
  }
};

std::unique_ptr<Storage> NewContextStorage(int store_type) {
  std::unique_ptr<Storage> context_store;
  context_store.reset(new ContextStorage(store_type));
  return context_store;
}

}  // namespace embedx
