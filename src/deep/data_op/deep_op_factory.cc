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

#include "src/deep/data_op/deep_op_factory.h"

#include <deepx_core/dx_log.h>

#include <mutex>
#include <unordered_map>

namespace embedx {
namespace deep_op {

class CreateOnceDeepOpFactory : public LocalDeepOpFactory {
 private:
  std::mutex mtx_;
  std::unordered_map<std::string, LocalDeepOp*> map_;

 public:
  CreateOnceDeepOpFactory() : LocalDeepOpFactory() {}

  ~CreateOnceDeepOpFactory() override {
    for (auto& entry : map_) {
      delete entry.second;
    }
  }

  bool Init(const LocalDeepOpResource* resource) override {
    resource_ = resource;

    std::unique_lock<std::mutex> _(mtx_);
    for (auto& entry : map_) {
      if (!entry.second->Init(resource_)) {
        return false;
      }
    }
    return true;
  }

  LocalDeepOp* LookupOrCreate(const std::string& name) override {
    std::unique_lock<std::mutex> _(mtx_);
    auto it = map_.find(name);
    if (it == map_.end()) {
      auto creator = op_registry_->Lookup(name);
      if (creator == nullptr) {
        DXTHROW_RUNTIME_ERROR("Couldn't lookup a creator named: %s.",
                              name.c_str());
      }

      LocalDeepOp* data_op = (*creator)();
      DXCHECK(data_op->Init(resource_));

      map_[name] = data_op;
      DXINFO("Create: %s!", name.c_str());
    }
    return map_[name];
  }
};

LocalDeepOpFactory::LocalDeepOpFactory()
    : op_registry_(LocalDeepOpRegistry::GetInstance()) {}

LocalDeepOpFactory* LocalDeepOpFactory::GetInstance() {
  static CreateOnceDeepOpFactory factory;
  return &factory;
}

}  // namespace deep_op
}  // namespace embedx
