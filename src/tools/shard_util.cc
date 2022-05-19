// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//

#include <deepx_core/graph/shard.h>
#include <deepx_core/tensor/data_type.h>

#include "src/tools/shard_func_name.h"

namespace embedx {
namespace {

/************************************************************************/
/* ShardFunc */
/************************************************************************/
class ShardFunc : public deepx_core::DataType {
 public:
  static int SRMShardFunc(int_t feature_id, int shard_size) noexcept {
    return (int)((uint64_t)feature_id % UINT64_C(9973) %  // magic number
                 (uint64_t)shard_size);
  }
};

/************************************************************************/
/* ShardFuncRegister */
/************************************************************************/
class ShardFuncRegister {
 public:
  ShardFuncRegister() {
    deepx_core::Shard::RegisterShardFunc(MOD9973_NAME, nullptr,
                                         &ShardFunc::SRMShardFunc);
  }
} shard_func_register;

}  // namespace
}  // namespace embedx
