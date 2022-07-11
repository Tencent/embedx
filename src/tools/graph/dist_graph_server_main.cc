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

#include <deepx_core/common/misc.h>
#include <deepx_core/common/stream.h>
#include <gflags/gflags.h>

#include "src/graph/graph_config.h"
#include "src/graph/server/dist_graph_server.h"
#include "src/tools/graph/graph_flags.h"

namespace embedx {
namespace {

void SetGraphConfig(GraphConfig* graph_config) {
  graph_config->set_ip_ports(FLAGS_gs_addrs);
  graph_config->set_shard_num(FLAGS_gs_shard_num);
  graph_config->set_shard_id(FLAGS_gs_shard_id);
  graph_config->set_thread_num(FLAGS_gs_thread_num);

  graph_config->set_node_graph(FLAGS_node_graph);
  graph_config->set_node_config(FLAGS_node_config);
  graph_config->set_node_feature(FLAGS_node_feature);
  graph_config->set_neighbor_feature(FLAGS_neighbor_feature);

  graph_config->set_negative_sampler_type(FLAGS_negative_sampler_type);
  graph_config->set_neighbor_sampler_type(FLAGS_neighbor_sampler_type);
  graph_config->set_random_walker_type(FLAGS_random_walker_type);

  graph_config->set_cache_thld(FLAGS_cache_thld);
  graph_config->set_cache_type(FLAGS_cache_type);
  graph_config->set_max_node_per_rpc(FLAGS_max_node_per_rpc);

  graph_config->set_success_out(FLAGS_success_out);
}

/************************************************************************/
/* main */
/************************************************************************/
void CheckFlags() {
  DXCHECK(!FLAGS_gs_addrs.empty());
  DXCHECK(FLAGS_gs_shard_num > 0);
  DXCHECK(FLAGS_gs_shard_id >= 0);
  DXCHECK(FLAGS_gs_thread_num > 0);

  DXCHECK(!FLAGS_node_graph.empty());

  DXCHECK(FLAGS_negative_sampler_type == 0 ||
          FLAGS_negative_sampler_type == 1 ||
          FLAGS_negative_sampler_type == 2 || FLAGS_negative_sampler_type == 3);
  DXCHECK(FLAGS_neighbor_sampler_type == 0 ||
          FLAGS_neighbor_sampler_type == 1 ||
          FLAGS_neighbor_sampler_type == 2 || FLAGS_neighbor_sampler_type == 3);
  DXCHECK(FLAGS_random_walker_type == 0 || FLAGS_random_walker_type == 1);

  DXCHECK(FLAGS_cache_thld >= 0);
  DXCHECK(FLAGS_cache_type == 0 || FLAGS_cache_type == 1 ||
          FLAGS_cache_type == 2);
  DXCHECK(FLAGS_max_node_per_rpc > 0);

  if (!FLAGS_success_out.empty()) {
    deepx_core::AutoFileSystem fs;
    deepx_core::CanonicalizePath(&FLAGS_success_out);
    DXCHECK(!FLAGS_success_out.empty());
    DXCHECK(fs.Open(FLAGS_success_out));
    DXCHECK(!deepx_core::IsStdinStdoutPath(FLAGS_success_out));
    (void)deepx_core::AutoFileSystem::MakeDir(FLAGS_success_out);
  }
}

int main(int argc, char* argv[]) {
  google::SetUsageMessage("Usage: [Options]");
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  GraphConfig graph_config;
  SetGraphConfig(&graph_config);

  DistGraphServer server;
  DXCHECK(server.Start(graph_config));

  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char* argv[]) { return embedx::main(argc, argv); }
