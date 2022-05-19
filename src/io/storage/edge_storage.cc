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

#include <deepx_core/dx_log.h>

#include <memory>  // std::unique_ptr
#include <mutex>
#include <string>

#include "src/common/data_types.h"
#include "src/io/storage/edge_vector.h"
#include "src/io/storage/storage.h"

namespace embedx {

class EdgeStorage : public Storage {
 private:
  std::mutex mtx_;
  std::unique_ptr<EdgeVector> edge_vector_;

 public:
  explicit EdgeStorage(int store_type) {
    edge_vector_ = NewEdgeVector(store_type);
  }
  ~EdgeStorage() override = default;

 public:
  void Clear() noexcept override { edge_vector_->Clear(); }
  void Reserve(uint64_t estimated_size) override {
    edge_vector_->Reserve(estimated_size);
  }
  void Lock() override { mtx_.lock(); }
  void UnLock() override { mtx_.unlock(); }
  bool InsertEdge(EdgeValue* value) override {
    return edge_vector_->Add(value);
  }

 public:
  size_t Size() const noexcept override { return edge_vector_->Size(); }
  bool Empty() const noexcept override { return edge_vector_->Empty(); }
  const vec_int_t& Keys() const noexcept override {
    return edge_vector_->src_node_list();
  }

 public:
  // Not Implemented
  const vec_pair_t* FindNeighbor(int_t /*node*/) const override {
    DXERROR("FindNeighbor was not implemented in the edge storage.");
    return nullptr;
  }
  std::string Print(int_t edge_id) const override {
    return edge_vector_->Print(edge_id);
  }
  int GetInDegree(int_t dst_node) const override {
    return edge_vector_->GetInDegree(dst_node);
  }
  int GetOutDegree(int_t src_node) const override {
    return edge_vector_->GetOutDegree(src_node);
  }
};

std::unique_ptr<Storage> NewEdgeStorage(int store_type) {
  std::unique_ptr<Storage> edge_store;
  edge_store.reset(new EdgeStorage(store_type));
  return edge_store;
}

}  // namespace embedx
