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
#include <gflags/gflags.h>

#include <memory>  // std::unique_ptr

#include "src/common/random.h"
#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/io/line_parser.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/graph/main_util.h"

// average_feature_main
DEFINE_int32(sample_num, 100000000, "How many neighbor nodes to sample.");

namespace embedx {
namespace {

void SampleNode(int count, const vec_pair_t& context, vec_int_t* nodes) {
  nodes->clear();

  if (context.size() <= (size_t)count) {
    for (auto& entry : context) {
      nodes->emplace_back(entry.first);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      auto k = int(ThreadLocalRandom() * context.size());
      nodes->emplace_back(context[k].first);
    }
  }
}

void AverageFeature(const std::vector<vec_pair_t>& feats_list,
                    vec_pair_t* average_feats) {
  average_feats->clear();

  for (const auto& feats : feats_list) {
    for (const auto& feat : feats) {
      auto feat_id = feat.first;
      auto feat_value = feat.second;
      auto it = std::find_if(
          average_feats->begin(), average_feats->end(),
          [feat_id](const pair_t& pair) { return pair.first == feat_id; });
      auto mean_value = feat_value / feats_list.size();
      if (it != average_feats->end()) {
        it->second += mean_value;
      } else {
        average_feats->emplace_back(feat_id, mean_value);
      }
    }
  }
}

class AverageFeatureMain : public MainUtil {
 private:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;

  int batch_node_;
  int count_;
  std::string entry_flag_;

 public:
  ~AverageFeatureMain() override = default;

  bool Init() override {
    if (FLAGS_dist) {
      graph_config_.set_ip_ports(FLAGS_gs_addrs);
    } else {
      graph_config_.set_node_graph(FLAGS_node_graph);
      graph_config_.set_node_feature(FLAGS_node_feature);
      graph_config_.set_node_config(FLAGS_node_config);
      graph_config_.set_thread_num(FLAGS_gs_thread_num);
    }

    graph_client_ = NewGraphClient(graph_config_, (GraphClientEnum)FLAGS_dist);
    // if (!graph_client_->Init(config_)) {
    if (!graph_client_) {
      return false;
    }

    // used by RunEntry
    batch_node_ = FLAGS_batch_node;
    count_ = FLAGS_sample_num;
    entry_flag_ = FLAGS_dist ? "Worker" : "Thread";
    return true;
  }

 private:
  const char* task_name() const noexcept override { return "AverageFeature"; }

  bool RunEntry(int entry_id, const vec_str_t& files,
                const std::string& out) override {
    DXINFO("%s id: %d is processing ...", entry_flag_.c_str(), entry_id);

    LineParser line_parser;
    deepx_core::AutoOutputFileStream ofs;
    std::ostringstream oss;
    for (const auto& file : files) {
      DXINFO("Processed file: %s", file.c_str());
      if (!line_parser.Open(file)) {
        return false;
      }

      auto out_file = deepx_core::GetOutputPredictFile(out, file);
      if (!ofs.Open(out_file)) {
        return false;
      }

      // averaging feature
      std::vector<NodeValue> values;
      vec_int_t nodes;
      std::vector<vec_pair_t> node_feats;
      while (line_parser.NextBatch<NodeValue>(batch_node_, &values)) {
        nodes = Collect<NodeValue, int_t>(values, &NodeValue::node);
        AverageBatchFeature(nodes, &node_feats);

        if (!MainUtil::DumpText(ofs, oss, nodes, node_feats)) {
          return false;
        }
      }
    }

    DXINFO("Done.");
    return true;
  }

  bool AverageBatchFeature(const vec_int_t& nodes,
                           std::vector<vec_pair_t>* average_feats) const {
    average_feats->clear();
    average_feats->resize(nodes.size());

    std::vector<vec_pair_t> contexts;
    if (!graph_client_->LookupContext(nodes, &contexts)) {
      return false;
    }

    vec_int_t remained_nodes;
    std::vector<vec_pair_t> node_feats;
    for (size_t i = 0; i < contexts.size(); ++i) {
      // Too many neighbor nodes lead to too many features.
      // Therefore, it is necessary to sample neighbor nodes
      SampleNode(count_, contexts[i], &remained_nodes);
      if (!graph_client_->LookupNodeFeature(remained_nodes, &node_feats)) {
        return false;
      }

      AverageFeature(node_feats, &(*average_feats)[i]);
    }
    return true;
  }
};

/************************************************************************/
/* main */
/************************************************************************/
void CheckFlags() {
  if (FLAGS_dist) {
    DXCHECK(!FLAGS_gs_addrs.empty());
    DXCHECK(FLAGS_gs_worker_num > 0);
    DXCHECK(FLAGS_gs_worker_id >= 0);
  } else {
    DXCHECK(FLAGS_gs_thread_num > 0);
    DXCHECK(!FLAGS_node_feature.empty());
  }
  DXCHECK(!FLAGS_node_graph.empty());

  DXCHECK(FLAGS_sample_num > 0);
  DXCHECK(FLAGS_batch_node > 0);
  DXCHECK(!FLAGS_out.empty());
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  std::unique_ptr<MainUtil> main(new AverageFeatureMain);
  if (!main->Init()) {
    return -1;
  }

  if (FLAGS_dist) {
    if (!main->RunSingleWorker(FLAGS_node_graph, FLAGS_out, FLAGS_gs_worker_num,
                               FLAGS_gs_worker_id)) {
      return -1;
    }
  } else {
    if (!main->RunMultiThread(FLAGS_node_graph, FLAGS_out,
                              FLAGS_gs_thread_num)) {
      return -1;
    }
  }
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
