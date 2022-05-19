// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#pragma once
#include <deepx_core/dx_log.h>

#include <functional>
#include <mutex>
#include <string>
#include <type_traits>  //std::enable_if, std::is_default_constructible, std::is_convertible
#include <unordered_map>

namespace embedx {

template <typename T>
class ClassRegistry {
 public:
  using ClassCreator = std::function<T*()>;

 private:
  std::mutex mtx_;
  std::unordered_map<std::string, ClassCreator> creator_map_;

 public:
  static ClassRegistry* GetInstance() {
    static ClassRegistry registry;
    return &registry;
  }

  ClassRegistry(const ClassRegistry&) = delete;
  ClassRegistry& operator=(const ClassRegistry&) = delete;

  void Register(const std::string& name, ClassCreator creator);
  ClassCreator* Lookup(const std::string& name);

 private:
  ClassRegistry() = default;
  ~ClassRegistry() = default;
};

template <typename T>
void ClassRegistry<T>::Register(const std::string& name, ClassCreator creator) {
  std::unique_lock<std::mutex> _(mtx_);
  if (creator_map_.find(name) != creator_map_.end()) {
    DXTHROW_RUNTIME_ERROR("Duplicate registered name: %s.", name.c_str());
  }
  creator_map_[name] = creator;
}

template <typename T>
auto ClassRegistry<T>::Lookup(const std::string& name) -> ClassCreator* {
  auto it = creator_map_.find(name);
  if (it == creator_map_.end()) {
    DXERROR("No creator named: %s.", name.c_str());
    return nullptr;
  }
  return &(it->second);
}

template <typename T, typename U,
          typename =
              typename std::enable_if<std::is_default_constructible<U>::value &&
                                      std::is_convertible<U*, T*>::value>::type>
class ClassFactoryRegister {
 private:
  static T* Create() { return new U; }

 public:
  explicit ClassFactoryRegister(const std::string& name) {
    ClassRegistry<T>::GetInstance()->Register(name, &Create);
  }
};

}  // namespace embedx
