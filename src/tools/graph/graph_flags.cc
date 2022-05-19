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
#include "src/tools/graph/graph_flags.h"

// dist and parallel
DEFINE_int32(dist, 0, "0 for local conditions, otherwise distributed.");
DEFINE_string(
    gs_addrs, "",
    "Graph server ip/port, format like:127.0.0.1:8000;127.0.0.1:8001.");
DEFINE_int32(gs_shard_num, 1, "Graph data shard number.");
DEFINE_int32(gs_shard_id, 0, "Current shard id.");
DEFINE_int32(gs_worker_num, -1, "How many worker used to process graph data.");
DEFINE_int32(gs_worker_id, -1, "Worker id of distributed graph server.");

// data
DEFINE_string(node_graph, "", "Node graph folder.");
DEFINE_string(node_config, "",
              "Configuration information of heterogeneous nodes.");
DEFINE_string(node_feature, "", "Node feature folder.");
DEFINE_string(neighbor_feature, "",
              "Neighbor feature folder, this can be empty.");

// sampler type
DEFINE_int32(
    negative_sampler_type, 0,
    "Negative sampler method, for now support: 0 uniform | 1 frequency(alias) "
    "| 2 frequency(word2vec) | 3 frequency(partial_sum).");
DEFINE_int32(
    neighbor_sampler_type, 0,
    "Neighbor sampler method, for now support: 0 uniform | 1 frequency(alias) "
    "| 2 frequency(word2vec) | 3 frequency(partial_sum).");
DEFINE_int32(random_walker_type, 0,
             "Random walker method, for now support: 0 uniform | 1 frequency.");

// cache
DEFINE_double(cache_thld, 0.0,
              "cache_thld represents percentage in random cache and degree "
              "cache, important factors in importance cache.");
DEFINE_int32(cache_type, 1,
             "0 refers to random cache, 1 refers to degree cache, 2 refers to "
             "importance cache.");
DEFINE_int32(max_node_per_rpc, 2000,
             "Limit the number of Nodes in one rpc request.");

// perf
DEFINE_int32(batch_node, 128, "Batch nodes.");
DEFINE_int32(gs_thread_num, 1, "How many thread used to parse graph data.");

// out
DEFINE_string(out, "", "Output folder or file.");
DEFINE_string(success_out, "",
              "The hdfs dir for saving success files, each graph server will "
              "generate a success file when server is ready.");
