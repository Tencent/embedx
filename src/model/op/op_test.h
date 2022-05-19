// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#pragma once
// include all headers needed by operator UTs
#include <deepx_core/dx_gtest.h>
#include <deepx_core/graph/op.h>
#include <deepx_core/graph/tensor_map.h>
#include <deepx_core/tensor/data_type.h>

#include <functional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

namespace deepx_core {

using param_initializer_t =
    std::function<void(std::default_random_engine& engine, TensorMap* param)>;
using inst_initializer_t = std::function<void(deepx_core::Instance* inst)>;

void CheckOpForward(GraphNode* node, int on_heap,
                    const DataType::tsr_t& expected_forward,
                    const param_initializer_t& pre_param_initializer = nullptr,
                    const param_initializer_t& post_param_initializer = nullptr,
                    const inst_initializer_t& inst_initializer = nullptr);
void CheckOpBackward(
    GraphNode* node, int on_heap,
    const param_initializer_t& pre_param_initializer = nullptr,
    const param_initializer_t& post_param_initializer = nullptr,
    const inst_initializer_t& inst_initializer = nullptr);

void CheckOpOverwrittenParam(
    GraphNode* node, const std::string& name, int on_heap,
    const DataType::srm_t& expected_overwritten_param,
    const param_initializer_t& pre_param_initializer = nullptr,
    const param_initializer_t& post_param_initializer = nullptr,
    const inst_initializer_t& inst_initializer = nullptr);

}  // namespace deepx_core
