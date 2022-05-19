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
#include <gflags/gflags.h>

// dist and parallel
DECLARE_int32(dist);
DECLARE_string(gs_addrs);
DECLARE_int32(gs_shard_num);
DECLARE_int32(gs_shard_id);
DECLARE_int32(gs_worker_num);
DECLARE_int32(gs_worker_id);

// data
DECLARE_string(node_graph);
DECLARE_string(node_config);
DECLARE_string(node_feature);
DECLARE_string(neighbor_feature);

// sampler type
DECLARE_int32(negative_sampler_type);
DECLARE_int32(neighbor_sampler_type);
DECLARE_int32(random_walker_type);

// perf
DECLARE_int32(batch_node);
DECLARE_int32(gs_thread_num);

// cache
DECLARE_double(cache_thld);
DECLARE_int32(cache_type);
DECLARE_int32(max_node_per_rpc);

// output
DECLARE_string(out);
DECLARE_string(success_out);
