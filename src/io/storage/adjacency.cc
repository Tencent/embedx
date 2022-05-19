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

#include "src/io/storage/adjacency.h"

#include <deepx_core/dx_log.h>

#include <utility>  // std::move

#include "src/io/storage/adjacency_impl.h"

namespace embedx {

Adjacency::Adjacency(std::unique_ptr<AdjacencyImpl>&& impl) {
  impl_ = std::move(impl);
}

Adjacency::~Adjacency() {}

void Adjacency::Clear() noexcept { impl_->Clear(); }

void Adjacency::Reserve(uint64_t estimated_size) {
  impl_->Reserve(estimated_size);
}

bool Adjacency::AddContext(AdjValue* value) { return impl_->AddContext(value); }

bool Adjacency::AddFeature(AdjValue* value) { return impl_->AddFeature(value); }

size_t Adjacency::Size() const noexcept { return impl_->Size(); }

bool Adjacency::Empty() const noexcept { return impl_->Empty(); }

const vec_int_t& Adjacency::Keys() const noexcept { return impl_->Keys(); }

const vec_pair_t* Adjacency::FindNeighbor(int_t node) const {
  return impl_->FindNeighbor(node);
}

std::string Adjacency::Print(int_t node) const { return impl_->Print(node); }

int Adjacency::GetInDegree(int_t dst_node) const {
  return impl_->GetInDegree(dst_node);
}

int Adjacency::GetOutDegree(int_t src_node) const {
  return impl_->GetOutDegree(src_node);
}

std::unique_ptr<Adjacency> NewAdjacency(AdjacencyEnum type) {
  std::unique_ptr<Adjacency> adjacency;
  switch (type) {
    case AdjacencyEnum::ADJ_LIST:
      adjacency.reset(new Adjacency(NewAdjListImpl()));
      break;
    case AdjacencyEnum::ADJ_MATRIX:
      adjacency.reset(new Adjacency(NewAdjMatrixImpl()));
      break;
    default:
      DXERROR("Need type: ADJ_LIST(0) || ADJ_MATRIX(1), got type: %d.",
              (int)type);
      break;
  }

  return adjacency;
}

}  // namespace embedx
