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

#include "src/graph/data_op/gs_op_factory.h"

#include <deepx_core/dx_log.h>

#include <mutex>
#include <unordered_map>

namespace embedx {
namespace graph_op {

class CreateOnceGSOpFactory : public LocalGSOpFactory {
 private:
  std::mutex mtx_;
  std::unordered_map<std::string, LocalGSOp*> map_;

 public:
  CreateOnceGSOpFactory() : LocalGSOpFactory() {}

  ~CreateOnceGSOpFactory() override {
    for (auto& entry : map_) {
      delete entry.second;
    }
  }

  bool Init(const LocalGSOpResource* resource) override {
    if (resource == nullptr) {
      DXERROR("LocaGSOpResource is nullptr.");
      return false;
    }

    resource_ = resource;

    std::unique_lock<std::mutex> _(mtx_);
    for (auto entry : map_) {
      if (!entry.second->Init(resource_)) {
        return false;
      }
    }
    return true;
  }

  LocalGSOp* LookupOrCreate(const std::string& name) override {
    std::unique_lock<std::mutex> _(mtx_);
    auto it = map_.find(name);
    if (it == map_.end()) {
      auto creator = op_registry_->Lookup(name);
      if (creator == nullptr) {
        DXTHROW_RUNTIME_ERROR("Couldn't lookup a creator named: %s.",
                              name.c_str());
      }

      LocalGSOp* gs_op = (*creator)();
      DXCHECK(gs_op->Init(resource_));

      map_[name] = gs_op;
      DXINFO("Create: %s.", name.c_str());
    }
    return map_[name];
  }
};

LocalGSOpFactory::LocalGSOpFactory()
    : op_registry_(LocalGSOpRegistry::GetInstance()) {}

LocalGSOpFactory* LocalGSOpFactory::GetInstance() {
  static CreateOnceGSOpFactory factory;
  return &factory;
}

class CreateOnceDistGSOpFactory : public DistGSOpFactory {
 private:
  std::mutex mtx_;
  std::unordered_map<std::string, DistGSOp*> map_;

 public:
  CreateOnceDistGSOpFactory() : DistGSOpFactory() {}
  ~CreateOnceDistGSOpFactory() override {
    for (auto& entry : map_) {
      delete entry.second;
    }
  }

  bool Init(const DistGSOpResource* resource, int shard_num) override {
    resource_ = resource;
    shard_num_ = shard_num;

    std::unique_lock<std::mutex> _(mtx_);
    for (auto entry : map_) {
      if (!entry.second->Init(resource_, shard_num_)) {
        return false;
      }
    }
    return true;
  }

  DistGSOp* LookupOrCreate(const std::string& name) override {
    std::unique_lock<std::mutex> _(mtx_);
    auto it = map_.find(name);
    if (it == map_.end()) {
      auto creator = op_registry_->Lookup(name);
      if (creator == nullptr) {
        DXTHROW_RUNTIME_ERROR("Couldn't lookup a creator named: %s.",
                              name.c_str());
      }

      DistGSOp* gs_op = (*creator)();
      DXCHECK(gs_op->Init(resource_, shard_num_));

      map_[name] = gs_op;
      DXINFO("Create: %s.", name.c_str());
    }
    return map_[name];
  }
};

DistGSOpFactory::DistGSOpFactory()
    : op_registry_(DistGSOpRegistry::GetInstance()) {}
DistGSOpFactory* DistGSOpFactory::GetInstance() {
  static CreateOnceDistGSOpFactory factory;
  return &factory;
}

}  // namespace graph_op
}  // namespace embedx
