// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <gflags/gflags.h>

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"
#include "src/sampler/random_walker_data_types.h"
#include "src/tools/graph/graph_flags.h"

namespace embedx {
namespace {

void SharedSampleNegativeTEST(const GraphClient* graph_client,
                              int test_number) {
  int count = 10;
  vec_int_t nodes = {0, 9};
  vec_int_t excluded_nodes = {1, 10};
  std::vector<vec_int_t> sampled_nodes_list;

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(graph_client->SharedSampleNegative(count, nodes, excluded_nodes,
                                               &sampled_nodes_list));
    DXCHECK(sampled_nodes_list.size() == 1u);
    DXCHECK(sampled_nodes_list[0].size() == (size_t)count);
    // exclude
    DXCHECK(std::find(sampled_nodes_list[0].begin(),
                      sampled_nodes_list[0].end(),
                      1) == sampled_nodes_list[0].end());
    DXCHECK(std::find(sampled_nodes_list[0].begin(),
                      sampled_nodes_list[0].end(),
                      1) == sampled_nodes_list[0].end());
  }
}

void IndepSampleNegativeTEST(const GraphClient* graph_client, int test_number) {
  int count = 10;
  vec_int_t nodes = {0, 9};
  vec_int_t excluded_nodes = {1, 10};
  std::vector<vec_int_t> sampled_nodes_list;

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(graph_client->IndepSampleNegative(count, nodes, excluded_nodes,
                                              &sampled_nodes_list));
    DXCHECK(sampled_nodes_list.size() == 2u);
    // exclude
    for (size_t i = 0; i < nodes.size(); ++i) {
      DXCHECK(std::find(sampled_nodes_list[i].begin(),
                        sampled_nodes_list[i].end(),
                        1) == sampled_nodes_list[i].end());
      DXCHECK(std::find(sampled_nodes_list[i].begin(),
                        sampled_nodes_list[i].end(),
                        10) == sampled_nodes_list[i].end());
    }
  }
}

void RandomSampleNeighborTEST(const GraphClient* graph_client,
                              int test_number) {
  int count = 3;
  vec_int_t nodes = {0, 9};
  std::vector<vec_int_t> neighbor_nodes_list;

  vec_int_t candidates_0 = {12, 11, 10};
  vec_int_t candidates_9 = {8, 7, 6};

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(
        graph_client->RandomSampleNeighbor(count, nodes, &neighbor_nodes_list));
    DXCHECK(nodes.size() == neighbor_nodes_list.size());

    DXCHECK(candidates_0.size() == neighbor_nodes_list[0].size());
    for (auto node : neighbor_nodes_list[0]) {
      DXCHECK(std::find(candidates_0.begin(), candidates_0.end(), node) !=
              candidates_0.end());
    }

    DXCHECK(candidates_9.size() == neighbor_nodes_list[1].size());
    for (auto node : neighbor_nodes_list[1]) {
      DXCHECK(std::find(candidates_9.begin(), candidates_9.end(), node) !=
              candidates_9.end());
    }
  }
}

void StaticTraverseTEST(const GraphClient* graph_client, int test_number) {
  vec_int_t cur_nodes = {0, 9};
  std::vector<int> walk_lens = {3, 3};
  WalkerInfo walker_info;
  std::vector<vec_int_t> seqs;

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(
        graph_client->StaticTraverse(cur_nodes, walk_lens, walker_info, &seqs));
    DXCHECK(cur_nodes.size() == seqs.size());
    for (size_t j = 0; j < seqs.size(); ++j) {
      DXCHECK(seqs[j].size() == (size_t)walk_lens[j]);
    }
  }

  cur_nodes = {15};
  walk_lens = {3};
  DXCHECK(
      !graph_client->StaticTraverse(cur_nodes, walk_lens, walker_info, &seqs));
}

void LookupFeatureTEST(const GraphClient* graph_client, int test_number) {
  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> node_feats;
  std::vector<vec_pair_t> neigh_feats;

  DXCHECK(graph_client->LookupFeature(nodes, &node_feats, &neigh_feats));
  for (int i = 0; i < test_number; ++i) {
    // node feature list
    DXCHECK(node_feats[0].size() == 2u);
    DXCHECK(node_feats[1].size() == 2u);
    // node(12) has no features, insert an empty feature
    DXCHECK(node_feats[2].size() == 1u);
    // node(13) does not exist in graph, insert an empty feature
    DXCHECK(node_feats[3].size() == 1u);

    // neighbor feature list
    DXCHECK(neigh_feats[0].size() == 2u);
    DXCHECK(neigh_feats[1].size() == 2u);
    // node(12) has no features, insert an empty feature
    DXCHECK(neigh_feats[2].size() == 1u);
    // node(13) does not exist in graph
    DXCHECK(neigh_feats[3].size() == 1u);
  }
}

void LookupNodeFeatureTEST(const GraphClient* graph_client, int test_number) {
  vec_int_t nodes = {10, 11, 12, 13};
  std::vector<vec_pair_t> node_feats;

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(graph_client->LookupNodeFeature(nodes, &node_feats));
    // node feature list
    DXCHECK(node_feats[0].size() == 2u);
    DXCHECK(node_feats[1].size() == 2u);
    // node(12) has no features, insert an empty feature
    DXCHECK(node_feats[2].size() == 1u);
    // node(13) does not exist in graph, insert an empty feature
    DXCHECK(node_feats[3].size() == 1u);
  }
}

void LookupContextTEST(const GraphClient* graph_client, int test_number) {
  vec_int_t nodes = {0, 1, 2};
  std::vector<vec_pair_t> contexts;

  for (int i = 0; i < test_number; ++i) {
    DXCHECK(graph_client->LookupContext(nodes, &contexts));
    DXCHECK(nodes.size() == contexts.size());

    DXCHECK(contexts[0].size() == 3u);
    DXCHECK(contexts[1].size() == 3u);
    DXCHECK(contexts[2].size() == 3u);
  }
}

class GraphClientTest {
 private:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;
  int NUMBER_TEST = 10;

 public:
  virtual ~GraphClientTest() = default;

 public:
  bool Init(const std::string& ip_ports) {
    graph_config_.set_ip_ports(ip_ports);
    graph_client_ = NewGraphClient(graph_config_, GraphClientEnum::DIST);
    return graph_client_ != nullptr;
  }

  void RunTest() {
    SharedSampleNegativeTEST(graph_client_.get(), NUMBER_TEST);
    IndepSampleNegativeTEST(graph_client_.get(), NUMBER_TEST);
    RandomSampleNeighborTEST(graph_client_.get(), NUMBER_TEST);
    StaticTraverseTEST(graph_client_.get(), NUMBER_TEST);
    LookupFeatureTEST(graph_client_.get(), NUMBER_TEST);
    LookupNodeFeatureTEST(graph_client_.get(), NUMBER_TEST);
    LookupContextTEST(graph_client_.get(), NUMBER_TEST);
  }
};

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  DXCHECK(!FLAGS_gs_addrs.empty());

  GraphClientTest gc_test;
  if (!gc_test.Init(FLAGS_gs_addrs)) {
    return -1;
  }

  gc_test.RunTest();
  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
