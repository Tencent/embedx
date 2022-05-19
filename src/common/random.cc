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

#include "src/common/random.h"

#include <chrono>
#include <random>  // std::uniform_real_distribution

namespace embedx {
namespace {

static thread_local std::default_random_engine e(
    std::chrono::system_clock::now().time_since_epoch().count());

static thread_local std::uniform_real_distribution<double> u(0, 1);

}  // namespace

double ThreadLocalRandom() { return u(e); }

}  // namespace embedx
